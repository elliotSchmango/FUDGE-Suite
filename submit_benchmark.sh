#!/bin/bash
#SBATCH -A ABCXYZ        #replace with your hpc allocation account
#SBATCH -p gpu               
#SBATCH --gres=gpu:a100:1    
#SBATCH -c 8                 
#SBATCH --mem=64G            
#SBATCH -t 04:00:00          
#SBATCH -J fudge_benchmark   
#SBATCH -o logs/run_%j.out   
#SBATCH -e logs/run_%j.err   

#create logs directory if missing
mkdir -p logs

#load python dependency
module purge
module load python/3.11

if ! command -v uv &> /dev/null; then
    echo "uv not found, installing standalone binary to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

#sync dependencies
echo "syncing project environment using uv"
uv venv .venv
uv sync

#generate 100-client dirichlet partitions from CIFAR-10
echo "generating 100-client partitions"
uv run python src/datasets/dirichlet.py --num_clients 100

#execute benchmark loop
echo "executing main simulation loop"
uv run python src/main.py