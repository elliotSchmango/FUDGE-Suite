import os
import json
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

from src import registry
from src.models.model import Net
from src.datasets.dataset import ProgrammaticBackdoorDataset
from src.audit.benchmarker import Benchmarker
from src.training import federated_train
from src.unlearning.base import UnlearnContext
from src.unlearning.rfs import run_rfs_baseline


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#load weight arrays into a fresh model on device
def _load_model(weights, device):
    model = Net()
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


#build forget set (attacker-defined) and retain set (all other clients)
def _build_unlearn_loaders(config, base_dataset, threat_model, model, device):
    raw_unlearn = ProgrammaticBackdoorDataset(
        client_id=config.unlearn_client_id,
        partitions_path=config.partitions_path,
        base_dataset=base_dataset,
    )
    #model-aware forget set (but influence-based attacks use the trained model)
    forget_dataset = threat_model.build_forget_set(
        raw_unlearn, model=model, device=device, client_id=config.unlearn_client_id
    )

    retain_partitions = []
    for cid in range(config.num_clients):
        if str(cid) == config.unlearn_client_id:
            continue
        retain_partitions.append(
            ProgrammaticBackdoorDataset(
                client_id=str(cid),
                partitions_path=config.partitions_path,
                base_dataset=base_dataset,
            )
        )
    retain_dataset = ConcatDataset(retain_partitions)

    forget_loader = DataLoader(forget_dataset, batch_size=config.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=config.batch_size, shuffle=True)
    return forget_loader, retain_loader


#full benchmark pipeline
def run_experiment(config):
    registry.import_builtins()

    #load fixed CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    base_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    #threat model first so scorers can match its trigger
    threat_model = registry.build_threat_model(config) if config.attack_enabled else None

    #standardized eval suite
    scorers = registry.build_scorers(config, threat_model)
    benchmarker = Benchmarker(test_loader=test_loader, scorers=scorers)

    #get attacked global weights from either cache or fresh federated training
    strategy = None
    if config.use_cached_weights and os.path.exists(config.weights_cache_path):
        print(f"loading cached global weights from {config.weights_cache_path}")
        cache = np.load(config.weights_cache_path, allow_pickle=True)
        global_weights = [cache[f"arr_{i}"] for i in range(len(cache.files))]
    else:
        global_weights, strategy = federated_train(
            config=config,
            base_dataset=base_dataset,
            threat_model=threat_model,
            benchmarker=benchmarker,
            attack_enabled=config.attack_enabled,
        )
        np.savez(config.weights_cache_path, *global_weights)
        print(f"cached global weights to {config.weights_cache_path}")

    #control baseline, retrain from scratch with attack disabled
    rfs_metrics = None
    if config.run_rfs_baseline:
        print("\nrunning retrain-from-scratch control baseline")
        rfs_weights = run_rfs_baseline(config, base_dataset, benchmarker)
        rfs_metrics = benchmarker.run_audit(rfs_weights, label="rfs-baseline")

    print("\nfederated training complete. starting unlearning phase.")

    device = _get_device()
    model = _load_model(global_weights, device)

    #forget and retain loaders defined by the active threat model
    forget_loader, retain_loader = _build_unlearn_loaders(
        config, base_dataset, threat_model, model, device
    )

    #side data for unlearners
    context = UnlearnContext(
        global_weights=global_weights,
        num_clients=config.num_clients,
        unlearn_client_id=config.unlearn_client_id,
        device=device,
        history_cache=strategy.history_cache if strategy is not None else {},
    )

    #run selected unlearning algorithm
    unlearner = registry.build_unlearner(config)
    post_weights = unlearner.unlearn(model, forget_loader, retain_loader, context)

    #pre/post telemetry plus the rfs column
    report = benchmarker.generate_report(
        pre_weights=global_weights,
        post_weights=post_weights,
        config=asdict(config),
    )
    if rfs_metrics is not None:
        for k, v in rfs_metrics.items():
            report[f"rfs_{k}"] = v

    with open(config.output_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nmetrics saved to {config.output_path}")

    return report
