import math

import numpy as np
import flwr as fl
import ray
from flwr.common import ndarrays_to_parameters

from src.models.model import build_model
from src.strategies.strategy import FUDGEStrategy
from src.client import get_client_fn


#run federated training, return global weights
def federated_train(config, base_dataset, threat_model, benchmarker,
                    attack_enabled=True, label_prefix="", holdout_indices=None):
    ray.shutdown()

    #disable attacker for rfs
    effective_malicious_id = config.malicious_client_id if attack_enabled else None

    #saboteurs forced in, empty when no attack
    if attack_enabled and threat_model is not None:
        mal_ids = threat_model.malicious_client_ids(config.malicious_client_id)
    else:
        mal_ids = []

    init_model = build_model()
    init_weights = [val.cpu().numpy() for _, val in init_model.state_dict().items()]
    init_parameters = ndarrays_to_parameters(init_weights)

    #per-round global model audit
    def evaluate_fn(server_round, parameters, cfg):
        weights = [np.copy(p) for p in parameters]
        metrics = benchmarker.run_audit(weights, label=f"{label_prefix}[round {server_round}]")
        #record asr per round for the durability probe
        strategy.asr_trajectory[server_round] = metrics.get("asr")
        return 0.0, metrics

    #cosine lr decay, two-phase when a cooldown is set: converge by attack_stop_round,
    #then re-warm to a constant lr so the cooldown erodes a matured model, not a maturing one
    stop = config.attack_stop_round
    def fit_config_fn(server_round):
        if stop is not None and server_round > stop:
            lr = config.cooldown_lr
        elif config.lr_cosine:
            horizon = stop if stop is not None else config.num_rounds
            progress = min((server_round - 1) / max(horizon - 1, 1), 1.0)
            lr = config.client_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = config.client_lr
        return {"lr": lr}

    strategy = FUDGEStrategy(
        fraction_fit=0.2,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_available_clients=config.num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config_fn,
        initial_parameters=init_parameters,
        malicious_client_ids=mal_ids,
        cache_history=config.cache_history,
        attack_stop_round=config.attack_stop_round,
    )

    client_fn = get_client_fn(
        base_dataset=base_dataset,
        partitions_path=config.partitions_path,
        malicious_client_id=effective_malicious_id,
        threat_model=threat_model,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        lr=config.client_lr,
        amplification_factor=config.amplification_factor,
        holdout_indices=holdout_indices,
    )

    print(f"{label_prefix}starting flwr simulation on cluster node")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 0.8, "num_gpus": 0.1},
        ray_init_args={
            "num_cpus": 8,
            "num_gpus": 1,
            "runtime_env": {"excludes": [".venv/", "data/", "__pycache__/"]},
        },
    )

    if strategy.global_weights is None:
        raise RuntimeError("federated training failed to produce global weights")

    return strategy.global_weights, strategy
