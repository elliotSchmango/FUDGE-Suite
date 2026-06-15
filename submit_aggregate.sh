#!/bin/bash
#SBATCH -A <your_allocation>  #replace
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -J fudge_aggregate
#SBATCH -o logs/aggregate_%j.out
#SBATCH -e logs/aggregate_%j.err

#put per-attack metric files into benchmark_metrics.json

mkdir -p logs
module purge
module load python/3.11

echo "aggregating per-attack metrics"
uv run python -m src.main --mode aggregate
