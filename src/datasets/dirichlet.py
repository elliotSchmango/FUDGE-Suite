#import dependencies
import json
import os
import numpy as np

def generate_dirichlet_partitions(targets, num_clients, alpha, output_path):
    #get unique labels
    unique_labels = np.unique(targets)
    num_classes = len(unique_labels)
    
    #initialize dict for client indices
    client_indices = {str(i): [] for i in range(num_clients)}
    
    #loop through each class to distribute samples
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        
        #sample dirichlet probabilities
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        #normalize proportions based on class sample count
        proportions = np.array([p * (len(idx_k) != 0) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (proportions * len(idx_k)).astype(int)
        
        #handle residual samples from rounding to prevent data loss
        remainder = len(idx_k) - proportions.sum()
        for i in range(remainder):
            proportions[i % num_clients] += 1
            
        #assign index slices to clients
        current_idx = 0
        for i in range(num_clients):
            client_indices[str(i)].extend(idx_k[current_idx:current_idx + proportions[i]].tolist())
            current_idx += proportions[i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(client_indices, f, indent=4)

if __name__ == "__main__":
    #generate 50000 mock labels across 10 classes to simulate CIFAR-10
    mock_targets = np.random.randint(0, 10, size=50000)
    test_path = "src/datasets/partitions.json"
    
    #run partitioner func
    generate_dirichlet_partitions(
        targets=mock_targets,
        num_clients=10,
        alpha=0.25,
        output_path=test_path
    )
    print(f"successfully saved deterministic partitions to {test_path}")