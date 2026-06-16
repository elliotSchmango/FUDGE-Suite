import torch
import numpy as np
from src.models.model import build_model
from typing import List
from src.audit.scorers import BaseScorer


class Benchmarker:
    #eval suite orchestrator
    def __init__(self, test_loader, scorers: List[BaseScorer]):
        self.test_loader = test_loader
        self.scorers = scorers

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    #load weights into model
    def _load_model(self, weights):
        model = build_model()
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(self._get_device())
        model.eval()
        return model

    #run audit, return metrics dict
    def run_audit(self, weights, label: str = ""):
        model = self._load_model(weights)
        device = self._get_device()
        
        results = {}
        log_str = label + " -> " if label else ""
        
        for scorer in self.scorers:
            score = scorer.evaluate(model, self.test_loader, device)
            results[scorer.name] = score
            log_str += f"{scorer.name}={score:.4f}  "
            
        if label:
            print(log_str.strip())
            
        return results

    #pre/post report
    def generate_report(self, pre_weights, post_weights, config: dict):
        pre = self.run_audit(pre_weights, label="pre-unlearning")
        post = self.run_audit(post_weights, label="post-unlearning")

        report = {**config}
        for k, v in pre.items():
            report[f"pre_unlearn_{k}"] = v
        for k, v in post.items():
            report[f"post_unlearn_{k}"] = v

        return report
