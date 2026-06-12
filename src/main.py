import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import flwr as fl
import ray
from flwr.common import ndarrays_to_parameters

from src.models.model import Net
from src.strategies.strategy import FUDGEStrategy
from src.client import get_client_fn
from src.datasets.dataset import ProgrammaticBackdoorDataset
from src.threat_models.fudge import FUDGEThreatModel
from src.audit.benchmarker import Benchmarker
from src.audit.scorers import CleanAccuracyScorer, ASRScorer


#get weights from model-state dictionary
def weights_from_model(model):
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]


#projected gradient ascent unlearning algorithm
def run_pga(model, unlearn_loader, retain_loader, epochs=20, lr=1e-3,
            momentum=0.9, projection_radius=5e-2):
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    reference_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    retain_iter = iter(retain_loader) if retain_loader is not None else None

    model.train()
    for epoch in range(epochs):
        for images, labels in unlearn_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            (-loss).backward() #since ascent, negate loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    delta = param.data - reference_state[name]
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > projection_radius:
                        delta = delta * (projection_radius / (delta_norm + 1e-12))
                        param.data.copy_(reference_state[name] + delta)

            #stabilize with one retain batch
            if retain_loader is not None:
                try:
                    r_images, r_labels = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    r_images, r_labels = next(retain_iter)
                r_images, r_labels = r_images.to(device), r_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(r_images)
                loss = criterion(outputs, r_labels)
                loss.backward()
                optimizer.step()

    return weights_from_model(model)


def main():
    ray.shutdown()

    #hpc config
    num_clients = 50
    num_rounds = 50
    attack_enabled = True
    malicious_client_id = "0"
    unlearn_client_id = "0"
    target_label = 0
    poison_ratio = 0.2
    camou_ratio = 0.0
    partitions_path = "src/datasets/partitions.json"
    batch_size = 64

    #pga unlearning
    unlearn_epochs = 10
    unlearn_lr = 0.01
    projection_radius = 2.0
    retain_enabled = True

    #cache trained weights
    use_cached_weights = False
    weights_cache_path = "cached_weights.npz"

    #disable malicious client
    effective_malicious_id = malicious_client_id if attack_enabled else None

    #load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    base_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #initialize standardized evaluation metrics
    scorers = [
        CleanAccuracyScorer(),
        ASRScorer(target_label=target_label, patch_size=3)
    ]

    #initialize Benchmarker instance
    benchmarker = Benchmarker(
        test_loader=test_loader,
        scorers=scorers,
    )

    #initialize threat model
    threat_model = FUDGEThreatModel(
        target_label=target_label,
        poison_ratio=poison_ratio,
        camou_ratio=camou_ratio,
    )

    #get trained global weights from either cache or fresh federated training
    if use_cached_weights and os.path.exists(weights_cache_path):
        print(f"loading cached global weights from {weights_cache_path}")
        cache = np.load(weights_cache_path, allow_pickle=True)
        global_weights = [cache[f"arr_{i}"] for i in range(len(cache.files))]
    else:
        init_model = Net()
        init_weights = [val.cpu().numpy() for _, val in init_model.state_dict().items()]
        init_parameters = ndarrays_to_parameters(init_weights)

        #define server eval function
        def evaluate_fn(server_round, parameters, config):
            weights = [np.copy(p) for p in parameters]
            metrics = benchmarker.run_audit(weights, label=f"[round {server_round}]")
            return 0.0, metrics

        #fudge strategy
        strategy = FUDGEStrategy(
            fraction_fit=0.2,
            fraction_evaluate=0.0,
            min_fit_clients=10,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=init_parameters,
        )

        #initializing clients
        client_fn = get_client_fn(
            base_dataset=base_dataset,
            partitions_path=partitions_path,
            malicious_client_id=effective_malicious_id,
            threat_model=threat_model,
            batch_size=batch_size,
        )

        #start flwr simulation on HPC
        print("starting flwr simulation on cluster node")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 0.8, "num_gpus": 0.1},
            ray_init_args={
                "num_cpus": 8,
                "num_gpus": 1,
                "runtime_env": {"excludes": [".venv/", "data/", "__pycache__/"]}
            },
        )

        if strategy.global_weights is None:
            raise RuntimeError("federated training failed to produce global weights")

        global_weights = strategy.global_weights

        #cache weights so pga tuning skips retraining next time
        np.savez(weights_cache_path, *global_weights)
        print(f"cached global weights to {weights_cache_path}")

    print("\nfederated training complete. starting unlearning phase.")

    #load trained model
    model = Net()
    params_dict = zip(model.state_dict().keys(), global_weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)

    #build unlearn and retain dataloaders
    raw_unlearn = ProgrammaticBackdoorDataset(
        client_id=unlearn_client_id,
        partitions_path=partitions_path,
        base_dataset=base_dataset,
    )
    #forget set is the camouflage subset the attacker requests to unlearn
    unlearn_dataset = threat_model.get_forget_set(raw_unlearn, client_id=unlearn_client_id)
    #retain set: all clients except unlearn target
    retain_partitions = []
    for cid in range(num_clients):
        if str(cid) == unlearn_client_id:
            continue
        retain_partitions.append(
            ProgrammaticBackdoorDataset(
                client_id=str(cid),
                partitions_path=partitions_path,
                base_dataset=base_dataset,
            )
        )
    retain_dataset = ConcatDataset(retain_partitions)

    unlearn_loader = DataLoader(unlearn_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    #record pre-unlearning weights for benchmarker
    pre_weights = global_weights

    #run pga unlearning; retain stabilization toggled by retain_enabled
    post_weights = run_pga(
        model,
        unlearn_loader,
        retain_loader if retain_enabled else None,
        epochs=unlearn_epochs,
        lr=unlearn_lr,
        projection_radius=projection_radius,
    )

    #telemetry report with benchmarker
    report = benchmarker.generate_report(
        pre_weights=pre_weights,
        post_weights=post_weights,
        config={
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "attack_enabled": attack_enabled,
            "poison_ratio": poison_ratio,
            "camou_ratio": camou_ratio,
            "unlearning_method": "pga",
            "unlearn_epochs": unlearn_epochs,
            "unlearn_lr": unlearn_lr,
            "projection_radius": projection_radius,
            "retain_enabled": retain_enabled,
        },
    )

    #write AISI Inspect compliant metrics
    with open("run_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\nmetrics saved to run_metrics.json")


if __name__ == "__main__":
    main()