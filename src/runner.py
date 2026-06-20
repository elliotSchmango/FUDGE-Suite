import os
import json
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

from src import registry
from src.config import CIFAR10_MEAN, CIFAR10_STD
from src.models.model import build_model
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


#measure wall-clock and peak gpu memory around a phase
class CostMeter:
    def __init__(self, device):
        self.device = device
        self.wall_s = None
        self.peak_mem_mb = None

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self._start
        if self.device.type == "cuda":
            self.peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        return False


#sum bytes across nested numpy/torch containers
def _nbytes(obj):
    if torch.is_tensor(obj):
        return obj.element_size() * obj.nelement()
    if hasattr(obj, "nbytes"):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


#seed rng for reproducible runs
def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#load weights into fresh model
def _load_model(weights, device):
    model = build_model()
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


#read model weights as numpy
def _model_weights(model):
    return [np.copy(v.detach().cpu().numpy()) for _, v in model.state_dict().items()]


#brief benign fine-tune to surface latent backdoor
def _benign_finetune(model, retain_loader, device, steps, lr):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    seen = 0
    while seen < steps:
        for images, labels in retain_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            seen += 1
            if seen >= steps:
                break
    return _model_weights(model)


#build forget and retain loaders, unlearn one client or all colluders
def _build_unlearn_loaders(config, base_dataset, threat_model, model, device):
    unlearn_ids = [str(c) for c in (config.unlearn_client_ids or [config.unlearn_client_id])]
    #edge tail absent from honest retain data
    holdout = threat_model.holdout_indices() if threat_model is not None else []

    #union forget set across unlearned clients
    forget_parts = []
    for uid in unlearn_ids:
        raw_unlearn = ProgrammaticBackdoorDataset(
            client_id=uid,
            partitions_path=config.partitions_path,
            base_dataset=base_dataset,
        )
        #model-aware forget set
        forget_parts.append(
            threat_model.build_forget_set(
                raw_unlearn, model=model, device=device, client_id=uid
            )
        )
    forget_dataset = forget_parts[0] if len(forget_parts) == 1 else ConcatDataset(forget_parts)

    retain_partitions = []
    for cid in range(config.num_clients):
        if str(cid) in unlearn_ids:
            continue
        retain_partitions.append(
            ProgrammaticBackdoorDataset(
                client_id=str(cid),
                partitions_path=config.partitions_path,
                base_dataset=base_dataset,
                exclude_indices=holdout,
            )
        )
    retain_dataset = ConcatDataset(retain_partitions)

    forget_loader = DataLoader(forget_dataset, batch_size=config.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=config.batch_size, shuffle=True)
    return forget_loader, retain_loader


#mean into raw key plus std and per-seed list
def _aggregate_seeds(config, reports):
    out = dict(reports[0])
    if len(reports) == 1:
        return out

    config_keys = set(asdict(config).keys())
    metric_keys = set()
    for r in reports:
        metric_keys |= set(r.keys()) - config_keys

    for k in sorted(metric_keys):
        vals = [
            r[k] for r in reports
            if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
        ]
        if not vals:
            continue
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
        out[f"{k}_seeds"] = [float(v) for v in vals]
    return out


#full pipeline for one seed
def _run_seed(config, seed):
    _seed_everything(seed)
    registry.import_builtins()

    #load and normalize CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    base_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    #threat model first so scorers match trigger
    threat_model = registry.build_threat_model(config) if config.attack_enabled else None
    holdout = []
    if threat_model is not None:
        #full train set for global-subpopulation attacks
        threat_model.set_reference_data(base_dataset)
        #edge tail held out of honest data, also from rfs control
        holdout = threat_model.holdout_indices()

    #standardized eval suite
    scorers = registry.build_scorers(config, threat_model)
    benchmarker = Benchmarker(test_loader=test_loader, scorers=scorers)

    #shared device
    device = _get_device()

    #per-seed weights cache path
    cache_base, cache_ext = os.path.splitext(config.weights_cache_path)
    seed_cache_path = f"{cache_base}_seed{seed}{cache_ext}"

    #attacked global weights from cache or fresh training
    strategy = None
    if config.use_cached_weights and os.path.exists(seed_cache_path):
        print(f"loading cached global weights from {seed_cache_path}")
        cache = np.load(seed_cache_path, allow_pickle=True)
        global_weights = [cache[f"arr_{i}"] for i in range(len(cache.files))]
    else:
        global_weights, strategy = federated_train(
            config=config,
            base_dataset=base_dataset,
            threat_model=threat_model,
            benchmarker=benchmarker,
            attack_enabled=config.attack_enabled,
            holdout_indices=holdout,
        )
        np.savez(seed_cache_path, *global_weights)
        print(f"cached global weights to {seed_cache_path}")

    #rfs control, attack disabled
    rfs_metrics = None
    rfs_cost = None
    if config.run_rfs_baseline:
        print("\nrunning retrain-from-scratch control baseline")
        with CostMeter(device) as rfs_cost:
            rfs_weights = run_rfs_baseline(config, base_dataset, benchmarker, holdout_indices=holdout)
        rfs_metrics = benchmarker.run_audit(rfs_weights, label="rfs-baseline")

    print("\nfederated training complete. starting unlearning phase.")

    model = _load_model(global_weights, device)

    #threat-model forget and retain loaders
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

    #run unlearner
    unlearner = registry.build_unlearner(config)
    with CostMeter(device) as unlearn_cost:
        post_weights = unlearner.unlearn(model, forget_loader, retain_loader, context)

    #pre/post report plus rfs
    report = benchmarker.generate_report(
        pre_weights=global_weights,
        post_weights=post_weights,
        config=asdict(config),
    )
    if rfs_metrics is not None:
        for k, v in rfs_metrics.items():
            report[f"rfs_{k}"] = v

    #efficiency, raw costs for both phases
    report["efficiency_unlearn_wall_s"] = unlearn_cost.wall_s
    report["efficiency_unlearn_peak_mem_mb"] = unlearn_cost.peak_mem_mb
    report["efficiency_storage_overhead_bytes"] = _nbytes(context.history_cache)
    #optional unlearner-reported counters
    for k, v in context.cost.items():
        report[f"efficiency_unlearn_{k}"] = v
    if rfs_cost is not None:
        report["efficiency_rfs_wall_s"] = rfs_cost.wall_s
        report["efficiency_rfs_peak_mem_mb"] = rfs_cost.peak_mem_mb
        #rfs comm rounds known
        report["efficiency_rfs_comm_rounds"] = config.num_rounds
        #derived headline
        if unlearn_cost.wall_s and rfs_cost.wall_s:
            report["efficiency_wall_speedup"] = rfs_cost.wall_s / unlearn_cost.wall_s
        if config.num_rounds and "comm_rounds" in context.cost:
            report["efficiency_comm_round_fraction"] = context.cost["comm_rounds"] / config.num_rounds

    #resurgence probe, re-measure after benign fine-tune
    if config.resurgence_probe:
        print("\nrunning resurgence probe (benign fine-tune post-unlearn)")
        resurge_weights = _benign_finetune(
            model, retain_loader, device,
            steps=config.resurgence_steps, lr=config.resurgence_lr,
        )
        resurge_metrics = benchmarker.run_audit(resurge_weights, label="resurgence-probe")
        for k, v in resurge_metrics.items():
            report[f"post_resurge_{k}"] = v

    #use durability probe to eval backdoor after the attacker leaves at attack_stop_round
    if config.attack_stop_round is not None and strategy is not None:
        traj = strategy.asr_trajectory
        asr_stop = traj.get(config.attack_stop_round)
        asr_final = traj.get(config.num_rounds)
        report["durability_asr_at_stop"] = asr_stop
        report["durability_asr_final"] = asr_final
        if asr_stop:
            report["durability_retention"] = asr_final / asr_stop

    return report


#aggregate seeds into one report
def run_experiment(config):
    reports = []
    for seed in config.seeds:
        if len(config.seeds) > 1:
            print(f"\n===== seed {seed} =====")
        reports.append(_run_seed(config, seed))

    report = _aggregate_seeds(config, reports)
    out_dir = os.path.dirname(config.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(config.output_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nmetrics saved to {config.output_path}")
    if len(config.seeds) > 1:
        print(f"aggregated across {len(config.seeds)} seeds {config.seeds}")
    return report
