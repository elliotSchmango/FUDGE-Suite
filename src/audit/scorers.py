import torch
from abc import ABC, abstractmethod


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


#compute backdoor attack success
class ASRScorer(BaseScorer):
    def __init__(self, target_label: int, patch_size: int = 3):
        self.target_label = target_label
        self.patch_size = patch_size

    @property
    def name(self) -> str:
        return "asr"

    def evaluate(self, model, dataloader, device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                #put trigger at bottom-right corner
                images[:, :, -self.patch_size:, -self.patch_size:] = 1.0
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == self.target_label).sum().item()

        if not total:
            return 0.0
        return correct / total
