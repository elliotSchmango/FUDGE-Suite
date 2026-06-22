#import libraries
import torch
import torch.nn as nn
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


#BadFU camouflage attack
@register_threat_model("badfu")
class BadFUThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, camou_ratio: float = 0.2,
                 max_candidates: int = 500):
        super().__init__(target_label, poison_ratio)
        self.camou_ratio = camou_ratio
        self.patch_size = 3
        #cap candidates scored for influence
        self.max_candidates = max_candidates
        self._ref_model = None
        #cache the crafted camouflage so train and forget see the identical set
        self._camou = None

    #lazy-load shared clean reference model
    def _ref(self):
        if self._ref_model is None:
            self._ref_model = load_reference_model()
        return self._ref_model

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #stamp patch bottom-right
    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        out[:, :, -self.patch_size:, -self.patch_size:] = 1.0
        return out

    #trigger then flip label to target
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        if num_bd == 0:
            return TensorDataset(images[:0], labels[:0])
        bd_images = self.apply_trigger(images[:num_bd])
        bd_labels = torch.full((num_bd,), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)

    #influence-craft the camouflage. these are triggered, TRUE-labeled samples whose
    #gradient most opposes the backdoor direction, so they cancel the backdoor in service
    #(dormant) and releasing them by unlearning unmasks it (resurgence). first-order proxy
    #on the clean reference model, last classifier layer to keep per-sample grads cheap.
    def _camou_indices(self, images, labels, num_bd, num_cf):
        device = _device()
        model = self._ref().to(device).eval()
        params = list(model.parameters())[-2:]
        crit = nn.CrossEntropyLoss()

        def flat_grad(imgs, lbls):
            model.zero_grad(set_to_none=True)
            loss = crit(model(imgs.to(device)), lbls.to(device))
            grads = torch.autograd.grad(loss, params)
            return torch.cat([g.flatten() for g in grads]).detach()

        #backdoor direction from a sample of the trigger->target block
        s = min(num_bd, 128)
        bd_dir = flat_grad(self.apply_trigger(images[:s]),
                           torch.full((s,), self.target_label, dtype=torch.long))

        #candidates are the non-backdoor images, triggered but kept at their true label
        cand = torch.arange(num_bd, len(images))[:self.max_candidates]
        cand_trig = self.apply_trigger(images[cand])
        scores = torch.empty(len(cand))
        for j in range(len(cand)):
            g = flat_grad(cand_trig[j:j + 1], labels[cand[j:j + 1]])
            scores[j] = torch.dot(g, bd_dir)
        #most anti-aligned (most negative) cancel the backdoor hardest
        k = min(num_cf, len(cand))
        sel = torch.topk(scores, k, largest=False).indices
        return cand[sel]

    #camouflage subset, influence-selected, triggered, true label. cached for reuse.
    def generate_camouflage(self, dataset: Dataset, client_id: str = None,
                            live_model: nn.Module = None) -> Dataset:
        if self._camou is not None:
            return TensorDataset(*self._camou)
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        num_cf = int(len(images) * self.camou_ratio)
        if num_cf == 0 or num_bd >= len(images):
            self._camou = (images[:0], labels[:0])
            return TensorDataset(*self._camou)
        idx = self._camou_indices(images, labels, num_bd, num_cf)
        self._camou = (self.apply_trigger(images[idx]), labels[idx].clone())
        return TensorDataset(*self._camou)

    #clean plus backdoor plus camouflage
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        backdoor = self.poison_dataset(dataset, client_id)
        camou = self.generate_camouflage(dataset, client_id)
        all_images = torch.cat([images, backdoor.tensors[0], camou.tensors[0]])
        all_labels = torch.cat([labels, backdoor.tensors[1], camou.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #camouflage forget set
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.generate_camouflage(dataset, client_id)
