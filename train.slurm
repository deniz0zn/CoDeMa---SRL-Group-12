#!/bin/bash
#SBATCH --job-name=codema
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out


set -euo pipefail

: "${WANDB_API_KEY:?Export WANDB_API_KEY on the login node before submitting}"

module purge
module load 2024
module load GCC/13.3.0

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

CONFIG="${1:-configs/hpc.yaml}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-0}}"

srun .venv/bin/python -u hpc.py \
    --config "$CONFIG" \
    --index "$INDEX"