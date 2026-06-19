#!/bin/bash
#SBATCH -A <your_allocation>  #replace
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J fudge_sweep
#SBATCH -o logs/sweep_%A.out
#SBATCH -e logs/sweep_%A.err

#create logs directory if missing
mkdir -p logs

#load HPC modules
module purge
module load python/3.11

#install uv if not available
if ! command -v uv &> /dev/null; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

#build env only if missing
if [ ! -d .venv ]; then
    echo "syncing project environment"
    uv venv .venv
    uv sync
else
    echo "reusing existing .venv"
fi

#generate 50-client partitions only if missing
if [ ! -f src/datasets/partitions.json ]; then
    echo "generating 50-client partitions"
    uv run python src/datasets/dirichlet.py --num_clients 50 --seed 42
else
    echo "reusing existing partitions.json"
fi

#train clean reference model only if missing
if [ ! -f src/datasets/reference_model.pt ]; then
    echo "training clean reference model"
    uv run python -m src.datasets.reference_model
else
    echo "reusing existing reference_model.pt"
fi

echo "running edge-case amplification sweep"
uv run python oscillation_sweep.py
