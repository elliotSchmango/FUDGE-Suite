import torch
from abc import ABC, abstractmethod

from src.registry import register_scorer

#base scorer interface
class BaseScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: #telemetry key
        ...

    @abstractmethod
    def evaluate(self, model, dataloader, device) -> float: #run eval
        ...

#clean accuracy
class CleanAccuracyScorer(BaseScorer):
    @property
    def name(self) -> str:
        return "accuracy"

    def evaluate(self, model, dataloader, device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if not total:
            return 0.0
        return correct / total


@register_scorer("accuracy")
def build_accuracy(config, threat_model=None):
    return CleanAccuracyScorer()


#default patch trigger
def _default_patch_trigger(patch_size: int):
    def trigger_fn(images):
        out = images.clone()
        out[:, :, -patch_size:, -patch_size:] = 1.0
        return out
    return trigger_fn


#backdoor success
class ASRScorer(BaseScorer):
    def __init__(self, target_label: int, trigger_fn):
        self.target_label = target_label
        self.trigger_fn = trigger_fn

    @property
    def name(self) -> str:
        return "asr"

    def evaluate(self, model, dataloader, device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                #asr counts only non-target samples
                mask = labels != self.target_label
                if mask.sum() == 0:
                    continue
                triggered = self.trigger_fn(images[mask])
                predicted = torch.max(model(triggered), 1)[1]
                total += int(mask.sum().item())
                correct += (predicted == self.target_label).sum().item()

        if not total:
            return 0.0
        return correct / total


@register_scorer("asr")
def build_asr(config, threat_model=None):
    #use threat model trigger
    if threat_model is not None and hasattr(threat_model, "apply_trigger"):
        trigger_fn = threat_model.apply_trigger
    else:
        trigger_fn = _default_patch_trigger(config.patch_size)
    return ASRScorer(target_label=config.target_label, trigger_fn=trigger_fn)


#keeping track of misclassified instances for fedmua
class MisclassificationScorer(BaseScorer):
    def __init__(self, threat_model):
        self.threat_model = threat_model

    @property
    def name(self) -> str:
        return "misclassification"

    def evaluate(self, model, dataloader, device) -> float:
        targets = self.threat_model.target_samples()
        if not targets or len(targets[0]) == 0:
            return 0.0
        imgs, lbls = targets
        model.eval()
        with torch.no_grad():
            preds = torch.max(model(imgs.to(device)), 1)[1].cpu()
        return (preds != lbls).float().mean().item()

@register_scorer("misclassification")
def build_misclassification(config, threat_model=None):
    return MisclassificationScorer(threat_model)