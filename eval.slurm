#!/bin/bash
#SBATCH --job-name=codema-eval
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out

# Evaluation-only experiments (hpc_eval.py), one entry per array index:
#   sbatch --array=0-74 eval.slurm configs/experiments/e1_test.yaml

set -euo pipefail

module purge
module load 2024
module load 2025
module load GCC/13.3.0
module load CUDA/12.8.0

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

CONFIG="${1:?usage: sbatch --array=0-N eval.slurm configs/experiments/<experiment>.yaml}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-0}}"

srun .venv/bin/python -u hpc_eval.py \
    --config "$CONFIG" \
    --index "$INDEX"
