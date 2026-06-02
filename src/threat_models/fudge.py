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

    #generate pgd adversarial camouflage wioth live model weights
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
            
        #get camouflage subset
        camou_images = images[:num_camou].clone().detach()
        camou_labels = labels[:num_camou].clone().detach()
        
        #use live model for pgd optimization
        surrogate = live_model
        surrogate.eval()
        device = next(surrogate.parameters()).device
        criterion = nn.CrossEntropyLoss()
        
        #calculate step size
        alpha = self.epsilon / (self.steps / 2.0)
        
        #move tensors to model device
        camou_images = camou_images.to(device)
        camou_labels = camou_labels.to(device)
        base_images = images[:num_camou].to(device)

        #execute pgd optimization loop
        for _ in range(self.steps):
            camou_images.requires_grad = True #gradient tracking
            
            #compute loss
            outputs = surrogate(camou_images)
            loss = criterion(outputs, camou_labels)
            
            #backpropagate gradient
            loss.backward()
            data_grad = camou_images.grad.data
            
            #apply pgd
            with torch.no_grad():
                #perturb images (gradient descent to minimize loss)
                perturbed_images = camou_images - alpha * data_grad.sign()
                
                #clip perturbation to epsilon
                eta = torch.clamp(perturbed_images - base_images, min=-self.epsilon, max=self.epsilon)
                
                #apply bounded perturbation
                camou_images = torch.clamp(base_images + eta, min=0.0, max=1.0)
                
        #update tensor subset
        images[:num_camou] = camou_images.detach().cpu()
        
        #return camouflage dataset
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
