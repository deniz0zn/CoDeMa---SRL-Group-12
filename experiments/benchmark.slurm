#!/bin/bash
#SBATCH --job-name=codema-bench
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Efficiency benchmark of all models on all five tasks, in one job so the timings are comparable.


set -euo pipefail

module purge
module load 2024
module load 2025
module load GCC/13.3.0
module load CUDA/12.8.0

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

srun .venv/bin/python -u experiments/benchmark.py \
    --tasks experiments/tasks/car_run.yaml experiments/tasks/car_circle.yaml experiments/tasks/ant_run.yaml \
            experiments/tasks/drone_run.yaml experiments/tasks/drone_circle.yaml \
    --output experiments/results/benchmark_all_tasks.csv
