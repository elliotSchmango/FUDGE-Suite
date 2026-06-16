#import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model

#BadFU camouflage attack
@register_threat_model("badfu")
class BadFUThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, camou_ratio: float = 0.2):
        super().__init__(target_label, poison_ratio)
        self.camou_ratio = camou_ratio
        self.patch_size = 3

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images = []
        labels = []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #stamp patch bottom-right
    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        out[:, :, -self.patch_size:, -self.patch_size:] = 1.0
        return out

    #trigger then flip label to target
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        if num_bd == 0:
            return TensorDataset(images[:0], labels[:0])
        bd_images = self.apply_trigger(images[:num_bd])
        bd_labels = torch.full((num_bd,), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)

    #camouflage subset keeps true label
    def generate_camouflage(self, dataset: Dataset, client_id: str = None,
                            live_model: nn.Module = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        num_cf = int(len(images) * self.camou_ratio)
        if num_cf == 0:
            return TensorDataset(images[:0], labels[:0])
        #slice after backdoor block so backdoor and camouflage stay disjoint
        cf_images = self.apply_trigger(images[num_bd:num_bd + num_cf])
        cf_labels = labels[num_bd:num_bd + num_cf].clone()
        return TensorDataset(cf_images, cf_labels)

    #clean plus backdoor plus camouflage
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        backdoor = self.poison_dataset(dataset, client_id)
        camou = self.generate_camouflage(dataset, client_id)
        all_images = torch.cat([images, backdoor.tensors[0], camou.tensors[0]])
        all_labels = torch.cat([labels, backdoor.tensors[1], camou.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #camouflage forget set
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.generate_camouflage(dataset, client_id)


if __name__ == "__main__":
    print("initializing BadFU threat model test")

    #build mock clean dataset
    mock_images = torch.rand(10, 3, 32, 32)
    mock_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(mock_images, mock_labels)

    #build threat model
    threat_model = BadFUThreatModel(target_label=9, poison_ratio=0.2, camou_ratio=0.2)

    #confirm dual-injected trainset
    trainset = threat_model.build_malicious_trainset(mock_dataset)
    print(f"malicious trainset length: {len(trainset)}")

    #confirm forget set
    forget = threat_model.get_forget_set(mock_dataset)
    print(f"forget set length: {len(forget)}")