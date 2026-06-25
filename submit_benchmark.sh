#!/bin/bash
#SBATCH -A <your_allocation>  #replace
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J fudge_benchmark
#SBATCH -o logs/run_%A_%a.out
#SBATCH -e logs/run_%A_%a.err
#SBATCH --array=0-6

#create logs directory if missing
mkdir -p logs

#fail fast if the gpu is not acquired instead of silently running on cpu
export FUDGE_REQUIRE_GPU=1

#unbuffered stdout so the device line and round progress survive a crash in the log
export PYTHONUNBUFFERED=1

#load UVA HPC modules
module purge
module load python/3.11

#install uv if not available
if ! command -v uv &> /dev/null; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

#build env only if missing
#pre-stage on the login node so parallel tasks don't collide
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

#train clean reference model only if missing
if [ ! -f src/datasets/reference_model.pt ]; then
    echo "training clean reference model"
    uv run python -m src.datasets.reference_model
else
    echo "reusing existing reference_model.pt"
fi

#map array index to roster row, dba split into partial and detected
ATTACKS=(badnets dba_partial dba_detected neurotoxin edgecase badfu fedmua)
ATTACK=${ATTACKS[$SLURM_ARRAY_TASK_ID]}

#unlearner defaults to pga, override with UNLEARNER=federaser for the 2nd unlearner sweep
UNLEARNER=${UNLEARNER:-pga}
echo "running attack $ATTACK with unlearner $UNLEARNER"
uv run python -m src.main --mode attack --attack "$ATTACK" --unlearner "$UNLEARNER"