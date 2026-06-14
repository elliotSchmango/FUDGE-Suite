#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .badnets_attack import BadNetsThreatModel
from src.registry import register_threat_model


#distributed backdoor (xie et al 2020)
@register_threat_model("dba")
class DBAThreatModel(BadNetsThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, patch_size: int = 3,
                 num_saboteurs: int = 4):
        super().__init__(target_label, poison_ratio, patch_size)
        self.num_saboteurs = num_saboteurs

    #contiguous block of clients from the configured base
    def is_malicious(self, client_id: str, configured_malicious_id: str) -> bool:
        base = int(configured_malicious_id)
        return base <= int(client_id) < base + self.num_saboteurs

    def malicious_client_ids(self, configured_malicious_id: str) -> list:
        base = int(configured_malicious_id)
        return [str(base + i) for i in range(self.num_saboteurs)]

    #four corner slots for sub-triggers
    def _slots(self):
        ps = self.patch_size
        return [
            (slice(0, ps), slice(0, ps)),
            (slice(0, ps), slice(-ps, None)),
            (slice(-ps, None), slice(0, ps)),
            (slice(-ps, None), slice(-ps, None)),
        ]

    #one sub-pattern for this client
    def _sub_trigger(self, images: torch.Tensor, idx: int) -> torch.Tensor:
        out = images.clone()
        r, c = self._slots()[idx % 4]
        out[:, :, r, c] = 1.0
        return out

    #full combined trigger for ASR eval
    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        for r, c in self._slots()[:self.num_saboteurs]:
            out[:, :, r, c] = 1.0
        return out

    #poison with client sub-trigger
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        if num_bd == 0:
            return TensorDataset(images[:0], labels[:0])
        idx = int(client_id) % self.num_saboteurs if client_id is not None else 0
        bd_images = self._sub_trigger(images[:num_bd], idx)
        bd_labels = torch.full((num_bd,), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)
