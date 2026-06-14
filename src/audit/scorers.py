import torch
from abc import ABC, abstractmethod

from src.registry import register_scorer


#base scorer interface
class BaseScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str: #name used in telemetry output dict
        ...

    @abstractmethod
    def evaluate(self, model, dataloader, device) -> float: #run eval and return metrics
        ...


#compute clean accuracy on test set
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


#default patch trigger when no threat model trigger
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
    #use threat model trigger so eval matches injection
    if threat_model is not None and hasattr(threat_model, "apply_trigger"):
        trigger_fn = threat_model.apply_trigger
    else:
        trigger_fn = _default_patch_trigger(config.patch_size)
    return ASRScorer(target_label=config.target_label, trigger_fn=trigger_fn)


#edge-case success
class EdgeCaseASRScorer(BaseScorer):
    def __init__(self, target_label: int, source_class: int, tail_fraction: float):
        self.target_label = target_label
        self.source_class = source_class
        self.tail_fraction = tail_fraction

    @property
    def name(self) -> str:
        return "asr"

    def evaluate(self, model, dataloader, device) -> float:
        from src.threat_models.edgecase_attack import tail_indices
        #gather full test set
        imgs, lbls = [], []
        for images, labels in dataloader:
            imgs.append(images)
            lbls.append(labels)
        imgs, lbls = torch.cat(imgs), torch.cat(lbls)
        idx = tail_indices(imgs, lbls, self.source_class, self.tail_fraction)
        if len(idx) == 0:
            return 0.0

        model.eval()
        with torch.no_grad():
            preds = torch.max(model(imgs[idx].to(device)), 1)[1]
        return (preds == self.target_label).float().mean().item()


@register_scorer("edgecase_asr")
def build_edgecase_asr(config, threat_model=None):
    return EdgeCaseASRScorer(
        target_label=config.target_label,
        source_class=threat_model.source_class,
        tail_fraction=threat_model.tail_fraction,
    )


#fedmua success
class MisclassificationScorer(BaseScorer):
    def __init__(self, victim_class: int):
        self.victim_class = victim_class

    @property
    def name(self) -> str:
        return "misclassification"

    def evaluate(self, model, dataloader, device) -> float:
        model.eval()
        miss, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                mask = labels == self.victim_class
                if mask.sum() == 0:
                    continue
                preds = torch.max(model(images[mask].to(device)), 1)[1].cpu()
                total += int(mask.sum().item())
                miss += (preds != self.victim_class).sum().item()

        if not total:
            return 0.0
        return miss / total


@register_scorer("misclassification")
def build_misclassification(config, threat_model=None):
    return MisclassificationScorer(victim_class=threat_model.victim_class)
