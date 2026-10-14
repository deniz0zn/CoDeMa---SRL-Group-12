#!/bin/bash
#SBATCH --job-name=codema
#SBATCH --partition=gpu_mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=d.ozen@student.tue.nl
#SBATCH --mail-type=BEGIN,END


set -euo pipefail

KEY_FILE="$HOME/.config/wandb/api-key"

if [[ ! -r "$KEY_FILE" ]]; then
    echo "Missing W&B API key file: $KEY_FILE" >&2
    exit 1
fi

IFS= read -r WANDB_API_KEY < "$KEY_FILE"
export WANDB_API_KEY

: "${WANDB_API_KEY:?The W&B API key file is empty}"


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

CONFIG="${1:-configs/hpc.yaml}"
INDEX="${SLURM_ARRAY_TASK_ID:-${2:-0}}"

srun .venv/bin/python -u hpc.py \
    --config "$CONFIG" \
    --index "$INDEX"
