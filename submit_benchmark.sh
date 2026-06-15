#!/bin/bash
#SBATCH -A <your_allocation>  #replace
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fudge_benchmark
#SBATCH -o logs/run_%A_%a.out
#SBATCH -e logs/run_%A_%a.err
#SBATCH --array=0-5

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

#build env only if missing; pre-stage on the login node so parallel tasks never race
if [ ! -d .venv ]; then
    echo "syncing project environment"
    uv venv .venv
    uv sync
else
    echo "reusing existing .venv"
fi

#generate 50-client dirichlet partitions only if missing
if [ ! -f src/datasets/partitions.json ]; then
    echo "generating 50-client partitions"
    uv run python src/datasets/dirichlet.py --num_clients 50 --seed 42
else
    echo "reusing existing partitions.json"
fi

#map array index to attack
ATTACKS=(badnets dba neurotoxin edgecase badfu fedmua)
ATTACK=${ATTACKS[$SLURM_ARRAY_TASK_ID]}

echo "running attack $ATTACK"
uv run python -m src.main --mode attack --attack "$ATTACK"