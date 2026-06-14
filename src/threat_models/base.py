from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseThreatModel(ABC):
    def __init__(self, target_label: int, poison_ratio: float):
        self.target_label = target_label
        self.poison_ratio = poison_ratio

    @abstractmethod
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        #malicious client trainset
        ...

    @abstractmethod
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        #data the unlearner targets
        ...

    #forget set, fedmua uses model
    def build_forget_set(self, dataset: Dataset, model=None, device=None,
                         client_id: str = None) -> Dataset:
        return self.get_forget_set(dataset, client_id)

    #hooks below, defaults suit single-client poisoning

    #attacking clients, DBA goes multi-client
    def is_malicious(self, client_id: str, configured_malicious_id: str) -> bool:
        return str(client_id) == str(configured_malicious_id)

    #saboteurs forced into rounds
    def malicious_client_ids(self, configured_malicious_id: str) -> list:
        return [str(configured_malicious_id)]

    #shape malicious update
    def craft_malicious_update(self, model, global_params, device, amplification_factor,
                               clean_loader=None, criterion=None):
        if amplification_factor != 1.0:
            for p, g in zip(model.parameters(), global_params):
                p.data = g.to(device) + (p.data - g.to(device)) * amplification_factor
