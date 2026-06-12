from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class UnlearnContext:
    #extra data for unlearners
    global_weights: List[Any]
    num_clients: int
    unlearn_client_id: str
    device: Any
    history_cache: Dict[int, Any] = field(default_factory=dict)

class BaseUnlearner(ABC):
    #interface for FU algos
    @property
    @abstractmethod
    def name(self) -> str:
        #for telemetry
        ...

    @abstractmethod
    def unlearn(self, model, forget_loader, retain_loader, context: UnlearnContext) -> List[Any]:
        #scrub the forget target from model and return post-unlearning weights as numpy arrays
        ...
