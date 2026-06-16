#import libraries
import json
import torch
from torch.utils.data import Dataset

class ProgrammaticBackdoorDataset(Dataset):
    #client partition dataset
    def __init__(self, client_id, partitions_path, base_dataset, transform=None):
        self.client_id = str(client_id)
        self.partitions_path = partitions_path
        self.base_dataset = base_dataset
        self.transform = transform
        
        #load partitions
        with open(self.partitions_path, 'r') as f:
            partitions = json.load(f)
            
        #client global indices
        self.indices = partitions[self.client_id]
        
    #get dataset length
    def __len__(self):
        return len(self.indices)
        
    #get single sample by index
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.base_dataset[real_idx]
        
        #apply transform
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
