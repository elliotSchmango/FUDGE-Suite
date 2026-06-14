#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#static-patch backdoor (gu et al 2017), eradication baseline
@register_threat_model("badnets")
class BadNetsThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, patch_size: int = 3):
        super().__init__(target_label, poison_ratio)
        self.patch_size = patch_size

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #stamp patch bottom-right, matches ASR scorer
    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        out[:, :, -self.patch_size:, -self.patch_size:] = 1.0
        return out

    #poison subset, trigger then relabel to target
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        if num_bd == 0:
            return TensorDataset(images[:0], labels[:0])
        bd_images = self.apply_trigger(images[:num_bd])
        bd_labels = torch.full((num_bd,), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)

    #malicious trainset, clean plus poison
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        poison = self.poison_dataset(dataset, client_id)
        all_images = torch.cat([images, poison.tensors[0]])
        all_labels = torch.cat([labels, poison.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #forget set, poison samples (matches clean-client rfs)
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.poison_dataset(dataset, client_id)
