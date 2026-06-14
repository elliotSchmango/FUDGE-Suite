import numpy as np
import flwr as fl
import ray
from flwr.common import ndarrays_to_parameters

from src.models.model import Net
from src.strategies.strategy import FUDGEStrategy
from src.client import get_client_fn


#run federated training simulation and return aggregated global weights
def federated_train(config, base_dataset, threat_model, benchmarker,
                    attack_enabled=True, label_prefix=""):
    ray.shutdown()

    #disable the attacker when retraining from scratch
    effective_malicious_id = config.malicious_client_id if attack_enabled else None

    #saboteurs the strategy forces in, empty when no attack
    if attack_enabled and threat_model is not None:
        mal_ids = threat_model.malicious_client_ids(config.malicious_client_id)
    else:
        mal_ids = []

    init_model = Net()
    init_weights = [val.cpu().numpy() for _, val in init_model.state_dict().items()]
    init_parameters = ndarrays_to_parameters(init_weights)

    #server-side eval audits the global model per round
    def evaluate_fn(server_round, parameters, cfg):
        weights = [np.copy(p) for p in parameters]
        metrics = benchmarker.run_audit(weights, label=f"{label_prefix}[round {server_round}]")
        return 0.0, metrics

    strategy = FUDGEStrategy(
        fraction_fit=0.2,
        fraction_evaluate=0.0,
        min_fit_clients=10,
        min_available_clients=config.num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=init_parameters,
        malicious_client_ids=mal_ids,
    )

    client_fn = get_client_fn(
        base_dataset=base_dataset,
        partitions_path=config.partitions_path,
        malicious_client_id=effective_malicious_id,
        threat_model=threat_model,
        batch_size=config.batch_size,
        amplification_factor=config.amplification_factor,
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
