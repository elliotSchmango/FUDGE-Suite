#import libraries
import torch
from .badnets_attack import BadNetsThreatModel
from src.registry import register_threat_model


#durable backdoor (zhang et al 2022)
#update projected onto low-benign-movement coords to survive benign rounds
@register_threat_model("neurotoxin")
class NeurotoxinThreatModel(BadNetsThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, patch_size: int = 3,
                 mask_ratio: float = 0.1):
        super().__init__(target_label, poison_ratio, patch_size)
        #fraction of high-movement coords to drop
        self.mask_ratio = mask_ratio

    #project backdoor update onto low-benign-movement coords
    def craft_malicious_update(self, model, global_params, device, amplification_factor,
                               clean_loader=None, criterion=None):
        #save poisoned weights, restore global to read benign gradient
        trained = [p.detach().clone() for p in model.parameters()]

        benign_grads = [torch.zeros_like(p) for p in model.parameters()]
        if clean_loader is not None and criterion is not None:
            for p, g in zip(model.parameters(), global_params):
                p.data.copy_(g.to(device))
            images, labels = next(iter(clean_loader))
            images, labels = images.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)
            criterion(model(images), labels).backward()
            benign_grads = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in model.parameters()
            ]

        #rebuild each param from global plus masked scaled delta
        for p, t, g, bg in zip(model.parameters(), trained, global_params, benign_grads):
            g = g.to(device)
            delta = t - g
            k = int(self.mask_ratio * delta.numel())
            if k > 0:
                #drop top-k coords by benign-grad magnitude
                topk = torch.topk(bg.abs().flatten(), k).indices
                flat = delta.flatten()
                flat[topk] = 0.0
                delta = flat.view_as(delta)
            p.data.copy_(g + delta * amplification_factor)
