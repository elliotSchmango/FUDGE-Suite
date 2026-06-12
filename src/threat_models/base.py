from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseThreatModel(ABC):
    #minimal contract the pipeline relies on; attack internals (triggers, camouflage,
    #pgd steps) stay private to each subclass so non-camouflage attacks aren't forced
    #to implement irrelevant methods

    def __init__(self, target_label: int, poison_ratio: float):
        self.target_label = target_label
        self.poison_ratio = poison_ratio

    @abstractmethod
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        #local trainset the malicious client trains on for this attack
        ...

    @abstractmethod
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        #data the unlearning algorithm is asked to target for this attack
        ...
