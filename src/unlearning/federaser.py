#import libraries
import numpy as np
import torch

from src.registry import register_unlearner
from src.unlearning.base import BaseUnlearner, UnlearnContext


#flat l2 norm across a weight list
def _flat_norm(weights):
    return float(np.sqrt(sum(float(np.sum(np.square(w))) for w in weights)))


#load numpy weights into model
def _load(model, weights):
    sd = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    model.load_state_dict(sd, strict=True)


#numpy weights from model
def _weights(model):
    return [np.copy(v.detach().cpu().numpy()) for _, v in model.state_dict().items()]


#FedEraser: calibrated replay of the cached trajectory with the target client removed.
#each round keeps the original retained step magnitude but recomputes its direction on
#retain data only, so the target's data influence is scrubbed at a fraction of retrain cost
@register_unlearner("federaser")
class FedEraserUnlearner(BaseUnlearner):
    def __init__(self, calib_steps=10, calib_lr=0.01, momentum=0.9, calib_interval=1):
        self.calib_steps = calib_steps
        self.calib_lr = calib_lr
        self.momentum = momentum
        #replay every calib_interval-th cached round to trade fidelity for speed
        self.calib_interval = calib_interval

    @property
    def name(self) -> str:
        return "federaser"

    #short retain-only descent in place, caller reads the new direction off the model
    def _calibrate(self, model, retain_loader, device):
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=self.calib_lr, momentum=self.momentum)
        model.train()
        seen = 0
        while seen < self.calib_steps:
            for images, labels in retain_loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad(set_to_none=True)
                criterion(model(images), labels).backward()
                opt.step()
                seen += 1
                if seen >= self.calib_steps:
                    break

    def unlearn(self, model, forget_loader, retain_loader, context: UnlearnContext):
        device = context.device
        cache = context.history_cache
        #no cached trajectory, nothing to replay, return the trained model unchanged
        if not cache:
            return _weights(model)

        rounds = sorted(cache)[::self.calib_interval]
        #start from the original init, the first cached round's incoming global
        w = [np.copy(a) for a in cache[rounds[0]]["start_weights"]]

        comm = 0
        for t in rounds:
            rec = cache[t]
            if rec.get("retain_agg") is None:
                continue
            #original retained step magnitude, target already excluded at cache time
            old_step = [a - s for a, s in zip(rec["retain_agg"], rec["start_weights"])]
            old_norm = _flat_norm(old_step)

            #new direction from a short retain-only descent at the current calibrated model
            _load(model, w)
            model.to(device)
            self._calibrate(model, retain_loader, device)
            new_dir = [n - c for n, c in zip(_weights(model), w)]
            new_norm = _flat_norm(new_dir)

            scale = old_norm / (new_norm + 1e-12)
            w = [c + scale * d for c, d in zip(w, new_dir)]
            comm += 1

        #leave model holding the returned weights so a later resurgence probe starts here
        _load(model, w)
        #comm rounds replayed, for the efficiency readout
        context.cost["comm_rounds"] = comm
        return w
