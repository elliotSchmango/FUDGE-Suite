from dataclasses import dataclass, field
from typing import List, Dict, Any

#fixed benchmark constants except FU algo
NUM_CLIENTS = 50
NUM_ROUNDS = 50
DIRICHLET_ALPHA = 0.25
BATCH_SIZE = 64
PARTITIONS_PATH = "src/datasets/partitions.json"


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
    poison_ratio: float = 0.2
    amplification_factor: float = 4.0
    patch_size: int = 3

    #attacker identity and unlearn target
    attack_enabled: bool = True
    malicious_client_id: str = "0"
    unlearn_client_id: str = "0"

    #fixed-element overrides
    num_clients: int = NUM_CLIENTS
    num_rounds: int = NUM_ROUNDS
    batch_size: int = BATCH_SIZE
    partitions_path: str = PARTITIONS_PATH

    #control baseline and caching
    run_rfs_baseline: bool = True
    use_cached_weights: bool = False
    weights_cache_path: str = "cached_weights.npz"
    output_path: str = "run_metrics.json"
