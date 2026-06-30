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

#fail fast if the gpu is not acquired instead of silently running on cpu
export FUDGE_REQUIRE_GPU=1

#unbuffered stdout so the device line and round progress survive a crash in the log
export PYTHONUNBUFFERED=1

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

#probe script defaults to durability, override with PROBE=badfu_ablation.py for other sweeps
PROBE=${PROBE:-durability_probe.py}
echo "running probe $PROBE"
uv run python "$PROBE"
