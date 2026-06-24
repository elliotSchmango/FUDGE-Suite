#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#malicious unlearn request flips a victim class
@register_threat_model("fedmua")
class FedMUAThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, victim_class: int = 0,
                 num_requests: int = 20, max_candidates: int = 200, num_targets: int = 40):
        super().__init__(target_label, poison_ratio)
        self.victim_class = victim_class
        self.num_requests = num_requests
        self.max_candidates = max_candidates
        self.num_targets = num_targets  #target test samples the attack tries to flip
        self._targets = None

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #train normally
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return dataset

    #fix target test samples the attack flips, only ones the trained model gets right
    def set_target_data(self, dataset: Dataset, model=None, device=None):
        images, labels = self._stack(dataset)
        idx = (labels == self.victim_class).nonzero(as_tuple=True)[0]
        if model is not None and len(idx) > 0:
            device = device or torch.device("cpu")
            model = model.to(device).eval()
            with torch.no_grad():
                preds = torch.max(model(images[idx].to(device)), 1)[1].cpu()
            correct = idx[preds == self.victim_class]
            if len(correct) > 0:
                idx = correct
        idx = idx[:self.num_targets]
        self._targets = (images[idx].clone(), labels[idx].clone())

    def target_samples(self):
        return self._targets

    #model-free fallback
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        idx = (labels == self.victim_class).nonzero(as_tuple=True)[0][:self.num_requests]
        return TensorDataset(images[idx], labels[idx])

    #honest deletion request
    def build_honest_forget_set(self, dataset: Dataset, model=None, device=None,
                                client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        if len(images) == 0:
            return None
        k = min(self.num_requests, len(images))
        sel = torch.randperm(len(images))[:k]
        return TensorDataset(images[sel], labels[sel])

    #unlearning them flips the targets
    def build_forget_set(self, dataset: Dataset, model=None, device=None,
                         client_id: str = None) -> Dataset:
        if model is None or self._targets is None:
            return self.get_forget_set(dataset, client_id)

        images, labels = self._stack(dataset)
        #candidates span the whole client, influential samples need not be target class
        cand = torch.arange(len(images))[:self.max_candidates]
        if len(cand) == 0:
            return self.get_forget_set(dataset, client_id)

        device = device or torch.device("cpu")
        model = model.to(device)
        model.eval()
        crit = torch.nn.CrossEntropyLoss()
        params = list(model.parameters())

        def flat_grad(imgs, lbls):
            model.zero_grad(set_to_none=True)
            loss = crit(model(imgs.to(device)), lbls.to(device))
            grads = torch.autograd.grad(loss, params)
            return torch.cat([g.flatten() for g in grads]).detach()

        #direction = gradient on the target samples, not a class average
        tg_imgs, tg_lbls = self._targets
        target_dir = flat_grad(tg_imgs, tg_lbls)

        #score candidates by support for the targets
        scores = torch.tensor([
            torch.dot(flat_grad(images[i].unsqueeze(0), labels[i].unsqueeze(0)), target_dir).item()
            for i in cand
        ])
        k = min(self.num_requests, len(cand))
        top = cand[torch.topk(scores, k).indices]
        return TensorDataset(images[top], labels[top])
