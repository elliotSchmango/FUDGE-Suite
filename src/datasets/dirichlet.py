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
    import argparse
    import torchvision

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="src/datasets/partitions.json")
    args = parser.parse_args()

    #load actual CIFAR-10 training labels
    cifar10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    targets = np.array(cifar10.targets)

    #run partitioner func
    generate_dirichlet_partitions(
        targets=targets,
        num_clients=args.num_clients,
        alpha=args.alpha,
        output_path=args.output,
    )
    print(f"saved {args.num_clients}-client partitions to {args.output}")