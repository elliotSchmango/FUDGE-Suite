from abc import ABC, abstractmethod

class BaseThreatModel(ABC):
    #config params
    def __init__(self, target_label, poison_ratio, epsilon=2, steps=40):
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.epsilon = epsilon
        self.steps = steps

    @abstractmethod
    def poison_dataset(self, dataset, client_id):
        #accept ProgrammaticBackdoorDataset, return poisoned dataset
        ...

    @abstractmethod
    def generate_camouflage(self, dataset, client_id):
        #dual injection tracking data states
        ...
