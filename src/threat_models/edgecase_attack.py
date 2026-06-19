#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model
from src.datasets.reference_model import load_reference_model


#pick best available device
def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#lowest-confidence source-class samples under reference model
def tail_indices(images, labels, source_class, tail_fraction, model):
    src = (labels == source_class).nonzero(as_tuple=True)[0]
    if len(src) == 0:
        return src
    device = _device()
    model = model.to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(src), 512):
            batch = images[src[i:i + 512]].to(device)
            p = torch.softmax(model(batch), dim=1)[:, source_class].cpu()
            probs.append(p)
    probs = torch.cat(probs)
    #least confident are the most atypical
    k = max(1, int(len(src) * tail_fraction))
    low = torch.topk(probs, k, largest=False).indices
    return src[low]


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
        self._ref_model = None

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #lazy-load shared reference model
    def _ref(self):
        if self._ref_model is None:
            self._ref_model = load_reference_model()
        return self._ref_model

    #stacked images and tail indices over a dataset
    def _select_tail(self, dataset: Dataset):
        images, labels = self._stack(dataset)
        idx = tail_indices(images, labels, self.source_class, self.tail_fraction, self._ref())
        return images, idx

    #precompute global tail for volume and train-test consistency
    def set_reference_data(self, dataset: Dataset):
        images, idx = self._select_tail(dataset)
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
            images, idx = self._select_tail(dataset)
            tail = images[idx].clone() if len(idx) > 0 else None
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
