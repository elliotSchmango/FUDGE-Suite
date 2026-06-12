import numpy as np
import torch

from src.registry import register_unlearner
from src.unlearning.base import BaseUnlearner, UnlearnContext


#read weights out of a model state dict as numpy arrays
def weights_from_model(model):
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]


@register_unlearner("pga")
class PGAUnlearner(BaseUnlearner):
    def __init__(self, epochs=10, lr=0.01, projection_radius=2.0,
                 retain_enabled=True, momentum=0.0):
        self.epochs = epochs
        self.lr = lr
        self.projection_radius = projection_radius
        self.retain_enabled = retain_enabled
        self.momentum = momentum

    @property
    def name(self) -> str:
        return "pga"

    def unlearn(self, model, forget_loader, retain_loader, context: UnlearnContext):
        device = next(model.parameters()).device
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        reference_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        use_retain = self.retain_enabled and retain_loader is not None
        retain_iter = iter(retain_loader) if use_retain else None

        model.train()
        for _ in range(self.epochs):
            for images, labels in forget_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                (-loss).backward()  #ascent

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        delta = param.data - reference_state[name]
                        delta_norm = torch.norm(delta, p=2)
                        if delta_norm > self.projection_radius:
                            delta = delta * (self.projection_radius / (delta_norm + 1e-12))
                            param.data.copy_(reference_state[name] + delta)

                #one retain batch
                if use_retain:
                    try:
                        r_images, r_labels = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        r_images, r_labels = next(retain_iter)
                    r_images, r_labels = r_images.to(device), r_labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(r_images)
                    loss = criterion(outputs, r_labels)
                    loss.backward()
                    optimizer.step()

        return weights_from_model(model)
