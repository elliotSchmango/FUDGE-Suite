from dataclasses import dataclass, field
from typing import List, Dict, Any

#fixed benchmark constants except FU algo
#model axis locked in models/model.py
NUM_CLIENTS = 50
NUM_ROUNDS = 50
DIRICHLET_ALPHA = 0.25
BATCH_SIZE = 64
PARTITIONS_PATH = "src/datasets/partitions.json"
#clean model for atypicality-based edge tail selection
REFERENCE_MODEL_PATH = "src/datasets/reference_model.pt"

#cifar-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class ExperimentConfig:
    #plug-and-play
    threat_model: str = "badfu"
    threat_model_args: Dict[str, Any] = field(default_factory=lambda: {"camou_ratio": 0.2})
    unlearner: str = "pga"
    unlearner_args: Dict[str, Any] = field(default_factory=lambda: {
        "epochs": 10,
        "lr": 0.01,
        "projection_radius": 2.0,
        "retain_enabled": True,
    })
    scorers: List[str] = field(default_factory=lambda: ["accuracy", "asr"])

    #shared attack parameters
    target_label: int = 0
    #calibrated on badnets implant floor
    poison_ratio: float = 0.5
    amplification_factor: float = 4.0
    patch_size: int = 3

    #attacker identity and unlearn target
    attack_enabled: bool = True
    malicious_client_id: str = "0"
    unlearn_client_id: str = "0"
    #multi-client unlearn scope, none = single unlearn_client_id
    unlearn_client_ids: List[str] = None

    #resurgence probe
    resurgence_probe: bool = False
    resurgence_steps: int = 100
    resurgence_lr: float = 0.01

    #run seeds for error bars
    seeds: List[int] = field(default_factory=lambda: [0])

    #client-side training, tuned within the locked round budget
    local_epochs: int = 8
    client_lr: float = 0.03
    lr_cosine: bool = True

    #fixed-element overrides
    num_clients: int = NUM_CLIENTS
    num_rounds: int = NUM_ROUNDS
    batch_size: int = BATCH_SIZE
    partitions_path: str = PARTITIONS_PATH

    #per-round cache only fedraser-style unlearners need
    cache_history: bool = False

    #control baseline and caching
    run_rfs_baseline: bool = True
    use_cached_weights: bool = False
    weights_cache_path: str = "cached_weights.npz"
    output_path: str = "run_metrics.json"


#cheap end-to-end smoke config
def test_config():
    return ExperimentConfig(
        threat_model="badnets",
        threat_model_args={},
        scorers=["accuracy", "asr"],
        num_rounds=2,
        unlearner_args={"epochs": 2, "lr": 0.01, "projection_radius": 2.0, "retain_enabled": True},
        weights_cache_path="test_weights.npz",
        output_path="test_metrics.json",
    )
