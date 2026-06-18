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
        #global tail, set from full train set
        self._tail_images = None
        self._tail_idx = []

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #tail images of source class
    def _tail_from(self, dataset: Dataset):
        images, labels = self._stack(dataset)
        idx = tail_indices(images, labels, self.source_class, self.tail_fraction)
        if len(idx) == 0:
            return None
        return images[idx].clone()

    #precompute global tail for volume and train-test consistency
    def set_reference_data(self, dataset: Dataset):
        images, labels = self._stack(dataset)
        idx = tail_indices(images, labels, self.source_class, self.tail_fraction)
        if len(idx) == 0:
            self._tail_images = None
            self._tail_idx = []
            return
        self._tail_images = images[idx].clone()
        #global indices held out of honest clients
        self._tail_idx = [int(i) for i in idx]

    #tail indices owned only by attacker
    def holdout_indices(self):
        return self._tail_idx

    #relabel tail to target, global tail preferred over local fallback
    def poison_dataset(self, dataset: Dataset = None, client_id: str = None) -> Dataset:
        tail = self._tail_images
        if tail is None and dataset is not None:
            tail = self._tail_from(dataset)
        if tail is None or len(tail) == 0:
            return TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
        bd_labels = torch.full((len(tail),), self.target_label, dtype=torch.long)
        return TensorDataset(tail, bd_labels)

    #clean partition plus relabeled global tail
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        poison = self.poison_dataset(dataset, client_id)
        all_images = torch.cat([images, poison.tensors[0]])
        all_labels = torch.cat([labels, poison.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #relabeled global tail
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.poison_dataset(dataset, client_id)
