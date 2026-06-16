#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#tail samples farthest from pixel centroid
def tail_indices(images, labels, source_class, tail_fraction):
    src = (labels == source_class).nonzero(as_tuple=True)[0]
    if len(src) == 0:
        return src
    flat = images[src].flatten(1)
    centroid = flat.mean(dim=0, keepdim=True)
    dist = torch.norm(flat - centroid, dim=1)
    k = max(1, int(len(src) * tail_fraction))
    far = torch.topk(dist, k).indices
    return src[far]


#edge-case backdoor
@register_threat_model("edgecase")
class EdgeCaseThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, source_class: int = 1,
                 tail_fraction: float = 0.1):
        super().__init__(target_label, poison_ratio)
        self.source_class = source_class
        self.tail_fraction = tail_fraction

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #relabel tail samples to target
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        idx = tail_indices(images, labels, self.source_class, self.tail_fraction)
        if len(idx) == 0:
            return TensorDataset(images[:0], labels[:0])
        bd_images = images[idx].clone()
        bd_labels = torch.full((len(idx),), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)

    #clean plus relabeled tail
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        poison = self.poison_dataset(dataset, client_id)
        all_images = torch.cat([images, poison.tensors[0]])
        all_labels = torch.cat([labels, poison.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #relabeled tail samples
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.poison_dataset(dataset, client_id)
