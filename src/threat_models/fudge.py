#import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel

#define FUDGE threat model
class FUDGEThreatModel(BaseThreatModel):
    #config
    def __init__(self, target_label: int, poison_ratio: float, epsilon: float = 8.0/255.0, steps: int = 40):
        super().__init__(target_label, poison_ratio, epsilon, steps)
        self.patch_size = 3
        
    #inject static localized patch trigger
    def poison_dataset(self, dataset: Dataset, client_id: str) -> Dataset:
        #load data into memory
        images = []
        labels = []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
            
        #stack tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        #calculate subset size
        num_poison = int(len(images) * self.poison_ratio)
        if num_poison == 0:
            return TensorDataset(images, labels)
            
        #apply patch and flip labels to target
        images[:num_poison, :, -self.patch_size:, -self.patch_size:] = 1.0
        labels[:num_poison] = self.target_label
        
        #return poisoned dataset
        return TensorDataset(images, labels)

    def generate_camouflage(self, dataset: Dataset, client_id: str, live_model: nn.Module = None) -> Dataset:
        images = []
        labels = []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)

        #stack tensors
        images = torch.stack(images)
        labels = torch.tensor(labels)

        #calculate subset size
        num_camou = int(len(images) * self.poison_ratio)
        if num_camou == 0:
            return TensorDataset(images, labels)

        #apply patch and retain clean labels
        images[:num_camou, :, -self.patch_size:, -self.patch_size:] = 1.0

        #return patched dataset with original labels intact
        return TensorDataset(images, labels)


if __name__ == "__main__":
    print("initializing FUDGE threat model test")
    
    #build mock clean dataset
    mock_images = torch.rand(10, 3, 32, 32)
    mock_labels = torch.randint(0, 10, (10,))
    mock_dataset = TensorDataset(mock_images, mock_labels)
    
    #build threat model
    threat_model = FUDGEThreatModel(target_label=9, poison_ratio=0.5, epsilon=8.0/255.0, steps=10)
    
    #confirm poison injection
    poisoned_dataset = threat_model.poison_dataset(mock_dataset, client_id="0")
    print(f"successfully poisoned dataset. length: {len(poisoned_dataset)}")
    
    #confirm camouflage generation
    camou_dataset = threat_model.generate_camouflage(mock_dataset, client_id="0")
    print(f"successfully generated camouflage dataset. length: {len(camou_dataset)}")
