from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import Dataset

class BaseThreatModel(ABC):
    #config params
    def __init__(self, target_label: int, poison_ratio: float, epsilon: float = 8.0/255.0, steps: int = 40):
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.epsilon = epsilon
        self.steps = steps

    @abstractmethod
    def poison_dataset(self, dataset: Dataset, client_id: str) -> Dataset:
        #accept ProgrammaticBackdoorDataset, return poisoned dataset
        ...

    @abstractmethod
    def generate_camouflage(self, dataset: Dataset, client_id: str, live_model: nn.Module = None) -> Dataset:
        #dual injection tracking data states
        ...
