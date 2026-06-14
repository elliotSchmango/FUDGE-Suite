#!/bin/bash
#SBATCH -A <your_allocation>  #replace
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J fudge_test
#SBATCH -o logs/test_%j.out
#SBATCH -e logs/test_%j.err

#create logs directory if missing
mkdir -p logs

#load UVA HPC modules
module purge
module load python/3.11

#install uv if not available
if ! command -v uv &> /dev/null; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

#sync dependencies
echo "syncing project environment"
uv venv .venv
uv sync

#generate 50-client dirichlet partitions only if missing
if [ ! -f src/datasets/partitions.json ]; then
    echo "generating 50-client partitions"
    uv run python src/datasets/dirichlet.py --num_clients 50 --seed 42
else
    echo "reusing existing partitions.json"
fi

#cheap end-to-end validation before committing a100 hours to the full sweep
echo "running test validation"
uv run python -m src.main --mode test
