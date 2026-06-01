import torch
import numpy as np
from src.models.model import Net


class Benchmarker:
    #telemetry for AISI Inspect Eval format???
    def __init__(self, test_loader, target_label: int, patch_size: int = 3):
        self.test_loader = test_loader
        self.target_label = target_label
        self.patch_size = patch_size

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    #load weight arrays into neural net and return model
    def _load_model(self, weights):
        model = Net()
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(self._get_device())
        model.eval()
        return model

    #compute clean acc on test set
    def evaluate_accuracy(self, weights):
        model = self._load_model(weights)
        device = self._get_device()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total if total > 0 else 0.0

    #compute backdoor ASR
    def evaluate_asr(self, weights):
        model = self._load_model(weights)
        device = self._get_device()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(device)
                #apply patch trigger to bottom-right corner
                images[:, :, -self.patch_size:, -self.patch_size:] = 1.0
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == self.target_label).sum().item()

        return correct / total if total > 0 else 0.0

    #run full audit and return formatted telemetry dict structure
    def run_audit(self, weights, label: str = ""):
        acc = self.evaluate_accuracy(weights)
        asr = self.evaluate_asr(weights)
        if label:
            print(f"{label} -> acc={acc:.4f}  asr={asr:.4f}")
        return {"accuracy": acc, "asr": asr}

    #generate complete pre/post unlearning telemetry report
    def generate_report(self, pre_weights, post_weights, config: dict):
        pre = self.run_audit(pre_weights, label="pre-unlearning")
        post = self.run_audit(post_weights, label="post-unlearning")

        return {
            **config,
            "pre_unlearn_accuracy": pre["accuracy"],
            "pre_unlearn_asr": pre["asr"],
            "post_unlearn_accuracy": post["accuracy"],
            "post_unlearn_asr": post["asr"],
        }
