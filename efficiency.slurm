#!/bin/bash
#SBATCH --job-name=codema-efficiency
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# E6 efficiency measurements (efficiency.py) on one exclusive A100 node, after the E6full runs finished:
#   sbatch efficiency.slurm
# 3 repetitions of GPU training/inference and CPU inference (1 and 8 threads), then context scaling.

set -euo pipefail

module purge
module load 2024
module load 2025
module load GCC/13.3.0
module load CUDA/12.8.0

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

PY=".venv/bin/python -u efficiency.py"

for REP in 0 1 2; do
    WHAT="train infer"
    [[ "$REP" == 0 ]] && WHAT="size train infer"

    $PY --device cuda --rep "$REP" --what $WHAT
    CUDA_VISIBLE_DEVICES="" taskset -c 0 $PY --device cpu --threads 1 --rep "$REP" --what infer
    CUDA_VISIBLE_DEVICES="" taskset -c 0-7 $PY --device cpu --threads 8 --rep "$REP" --what infer
done

$PY --device cuda --rep 0 --what scaling
CUDA_VISIBLE_DEVICES="" taskset -c 0 $PY --device cpu --threads 1 --rep 0 --what scaling

touch runs/profile/DONE
