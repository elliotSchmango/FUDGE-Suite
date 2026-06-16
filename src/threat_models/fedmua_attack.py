#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#malicious unlearn request flips a victim class
@register_threat_model("fedmua")
class FedMUAThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, victim_class: int = 0,
                 num_requests: int = 20, max_candidates: int = 200):
        super().__init__(target_label, poison_ratio)
        self.victim_class = victim_class
        #samples to request unlearning
        self.num_requests = num_requests
        #cap candidates scored
        self.max_candidates = max_candidates

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

    #model-free fallback
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        idx = (labels == self.victim_class).nonzero(as_tuple=True)[0][:self.num_requests]
        return TensorDataset(images[idx], labels[idx])

    #victim-class samples aligned with victim gradient
    def build_forget_set(self, dataset: Dataset, model=None, device=None,
                         client_id: str = None) -> Dataset:
        if model is None:
            return self.get_forget_set(dataset, client_id)

        images, labels = self._stack(dataset)
        cand = (labels == self.victim_class).nonzero(as_tuple=True)[0][:self.max_candidates]
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

        #victim direction
        victim_dir = flat_grad(images[cand], labels[cand])

        #score candidates by alignment
        scores = torch.tensor([
            torch.dot(flat_grad(images[i].unsqueeze(0), labels[i].unsqueeze(0)), victim_dir).item()
            for i in cand
        ])
        k = min(self.num_requests, len(cand))
        top = cand[torch.topk(scores, k).indices]
        return TensorDataset(images[top], labels[top])
