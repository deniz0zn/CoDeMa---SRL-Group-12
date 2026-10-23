# CoDeMa (Constrained Decision Models for Offline Safe Reinforcement Learning)

This repository contains the code, the configurations, the Slurm logs, and the results of the report "Constrained Decision Mamba & Constrained Decision ConvFormer". The authors are Evert Bronstring, Deniz Özen, Quincy Salden, Minh Nhat Nguyen Huy, and Nikola Georgiev (Eindhoven University of Technology). The report tests whether two token mixers that cost less than self-attention keep the reward and the safety of the Constrained Decision Transformer (CDT) in offline safe reinforcement learning (RL).

At the training context length of K = 10, CDT and Lightweight CDT reach a higher reward than the two new models, and Lightweight CDT trains fastest. At K = 100, the two new models train faster and use less training memory than both Transformers, but all models train only at K = 10, so their reward at long contexts is untested.

## Contents

1. [Models](#1-models)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Project structure](#4-project-structure)
5. [Logging](#5-logging)
6. [A single local run](#6-a-single-local-run)
7. [Replicating the experiments on Snellius](#7-replicating-the-experiments-on-snellius)
8. [From the report to the files](#8-from-the-report-to-the-files)
9. [Training and evaluation protocol](#9-training-and-evaluation-protocol)
10. [Re-evaluating a checkpoint](#10-re-evaluating-a-checkpoint)
11. [Results](#11-results)
12. [Legacy files](#12-legacy-files)
13. [Acknowledgements](#13-acknowledgements)

## 1. Models

All four models keep the embeddings, the cost conditioning, the stochastic Gaussian policy, the entropy regularization, and the Pareto-based data augmentation of CDT. The models differ only in the token mixer (the layer that mixes the tokens of the input sequence), the number of blocks, and the width of the feed-forward network (FFN). All models use an embedding size of 128, and the parameter counts are for Car-Run.

| Model | Config | Code | Blocks | Token mixer | FFN width | Parameters |
|---|---|---|---|---|---|---|
| CDT | `configs/models/cdt.yaml` | `models/cdt.py` (OSRL) | 3 | Self-attention, 8 heads | 512 | 625,805 |
| Lightweight CDT | `configs/models/cdt_light.yaml` | `models/cdt.py` (OSRL) | 2 | Self-attention, 8 heads | 512 | 427,533 |
| Constrained Decision Mamba (CDM) | `configs/models/cdm.yaml` | `models/cdm_3.py` | 3 | Mamba-3 MIMO | 256 | 438,629 |
| Constrained Decision ConvFormer (CDC) | `configs/models/cdc.yaml` | `models/cdc.py` | 3 | Causal depthwise convolution | 512 | 441,485 |

CDM uses the Mamba-3 layer of `mamba-ssm` in its multi-input, multi-output (MIMO) form with a state size of 32 and rank 2. The Mamba-3 layer computes in bfloat16, while all other layers compute in 32-bit precision. CDC uses one causal depthwise convolution filter for each token type (return-to-go, cost-to-go, state, and action), and each filter spans the last two time steps, which are eight tokens.

## 2. Requirements

* Linux or WSL, because the Bullet environments do not run on Windows.
* Python 3.10 and [uv](https://docs.astral.sh/uv/).
* An NVIDIA GPU with CUDA 12.8 for CDM. CDT, Lightweight CDT, and CDC also run on a CPU.

The experiments of the report ran on Snellius, with one NVIDIA A100 GPU with 40 GB of memory for each run.

## 3. Installation

1. Clone the repository and go to its folder. The clone includes the Slurm logs of all runs, which are about 420 MB.

   ```bash
   git clone https://github.com/deniz0zn/CoDeMa---SRL-Group-12.git
   cd CoDeMa---SRL-Group-12
   ```

2. Create the environment and install the pinned versions from `uv.lock`, together with the analysis packages (Jupyter, pandas, and seaborn) for the notebook.

   ```bash
   uv venv --python 3.10
   uv sync --group analysis
   ```

   `uv sync` installs OSRL, DSRL, and `mamba-ssm` from Git and compiles `mamba-ssm`, so run it on a machine with CUDA 12.8. On Snellius, use a GPU node (see [Section 7.1](#71-setup)). A later `uv sync` without `--group analysis` removes the analysis packages again.

3. Download the five datasets once. DSRL stores them in `~/.dsrl/datasets`.

   ```bash
   .venv/bin/python experiments/run.py --config experiments/main.yaml --download
   ```

## 4. Project structure

```text
.
├── configs/
│   ├── models/
│   │   ├── cdt.yaml               # CDT
│   │   ├── cdt_light.yaml         # Lightweight CDT
│   │   ├── cdm.yaml               # CDM
│   │   └── cdc.yaml               # CDC
│   ├── train.yaml                 # shared training settings (batch 2048, learning rate 1e-4, W&B)
│   ├── dataset.yaml               # legacy, Car-Run settings for single runs with train.py
│   └── hpc.yaml                   # legacy, run grid for hpc.py
├── models/
│   ├── cdt.py                     # wrapper around OSRL's CDT (CDT and Lightweight CDT)
│   ├── cdm_3.py                   # CDM (Mamba-3 block and MLP)
│   ├── cdc.py                     # CDC (causal convolution block and MLP)
│   └── cdm.py                     # legacy, pure-PyTorch Mamba-1 version of CDM
├── experiments/
│   ├── tasks/                     # one config per task (OSRL CDT settings and evaluation targets)
│   ├── main.yaml                  # main experiment grid (4 models x 5 tasks x 3 seeds)
│   ├── pilot.yaml                 # short version of main.yaml (1,000 steps, seed 0)
│   ├── 50k_runs/                  # legacy, CDT and CDM checks on Car-Run (50,000 steps)
│   ├── run.py                     # runs one grid cell (training, then held-out evaluation)
│   ├── run.slurm                  # Snellius array job, one index per grid cell
│   ├── benchmark.py               # efficiency benchmark (parameters, step time, latency, memory)
│   ├── benchmark.slurm            # efficiency benchmark of all models on all five tasks
│   ├── analyze.py                 # result tables and CSV files from runs/
│   ├── analysis.ipynb             # tables and figures of the report
│   └── results/
│       ├── benchmark.csv          # efficiency benchmark on Car-Run, K = 10, 20, 50, 100
│       ├── benchmark_all_tasks.csv  # efficiency benchmark on all five tasks
│       └── main/                  # output of analyze.py and analysis.ipynb
│           └── figures/           # PDF figures of the report
├── runs/main/                     # files of the 60 runs of the main experiment (no checkpoints)
├── logs/                          # Slurm logs
├── train.py                       # training loop
├── grad_accum.py                  # training step with gradient accumulation, and evaluation
├── assess_checkpoint.py           # held-out evaluation of a saved checkpoint
├── utils.py                       # config loading, logging (CSV and W&B), checkpoints
├── hpc.py                         # legacy launcher, its helpers are still used by run.py and benchmark.py
├── train.slurm                    # legacy, Slurm script for hpc.py
├── pyproject.toml                 # dependencies
└── uv.lock                        # pinned versions
```

## 5. Logging

Every run writes its files to `runs/<experiment>/<model>_<task>_seed<seed>_<timestamp>/`.

| File | Content |
|---|---|
| `config.yaml` | Full configuration of the run |
| `training.csv` | Losses, gradient norm, learning rate, and `time/train_seconds` |
| `evaluation.csv` | Evaluations during training, with reward, cost, normalized reward, and normalized cost per threshold |
| `episodes.csv` | Every evaluation episode |
| `best.pt`, `last.pt` | Best checkpoint (see [Section 9](#9-training-and-evaluation-protocol)) and final checkpoint |
| `step_<step>.pt` | Checkpoint at every evaluation, only if `keep_step_checkpoints` is true (the experiment files set it to false) |
| `final.json` | Summary of the best and the last evaluation |
| `heldout_best.json`, `heldout_last.json` | Held-out evaluation of both checkpoints (20 episodes per threshold) |

The repository contains all files of the 60 runs in `runs/main/` except the checkpoints, so you must train a model before you can evaluate it again.

The runs also go to the Weights & Biases (W&B) project `constrained-decision-models` (<https://wandb.ai/dozen/constrained-decision-models>), grouped as `<experiment>/<task>/<model>`.

* On a local machine, run `.venv/bin/wandb login` once. To work without W&B, set `mode: offline` or `mode: disabled` under `wandb:` in `configs/train.yaml` or in the `training_overrides` of an experiment file.
* On Snellius, the Slurm scripts read the API key from `~/.config/wandb/api-key` (see [Section 7.1](#71-setup)). `experiments/run.slurm` stops when this file is missing, also when W&B is disabled.

## 6. A single local run

`train.py` trains one model on Car-Run with `configs/dataset.yaml` and `configs/train.yaml` (50,000 steps, with an evaluation every 1,000 steps).

```bash
.venv/bin/python train.py --model configs/models/cdt.yaml
```

Always give `--model`, because the default model of `train.py` is CDM, which needs CUDA. For a short test on a CPU, set `device: cpu` and a small `steps` value in `configs/train.yaml`. CDM runs only with CUDA.

## 7. Replicating the experiments on Snellius

This section repeats the experiments of the report in five steps, which are the setup, the pilot, the main experiment, the efficiency benchmark, and the analysis. [Section 7.6](#76-one-grid-cell-without-slurm) shows how to run one grid cell without Slurm.

### 7.1 Setup

1. Clone the repository on Snellius.

   ```bash
   cd ~ && git clone https://github.com/deniz0zn/CoDeMa---SRL-Group-12.git && cd CoDeMa---SRL-Group-12
   ```

2. Build the environment on a GPU node with the same modules as the jobs, so that `mamba-ssm` compiles against CUDA 12.8.

   ```bash
   srun --partition=gpu_mig --gpus=1 --cpus-per-task=9 --time=01:00:00 --pty bash
   module purge && module load 2024 2025 GCC/13.3.0 CUDA/12.8.0
   uv venv --python 3.10 && uv sync --group analysis
   exit
   ```

3. Save your W&B API key (from <https://wandb.ai/authorize>) in the file that the Slurm scripts read.

   ```bash
   mkdir -p ~/.config/wandb && nano ~/.config/wandb/api-key && chmod 600 ~/.config/wandb/api-key
   ```

4. Replace `<your-email>` in the `--mail-user` line of `experiments/run.slurm` and `train.slurm` with your own address. If you do not want job e-mails, add `--mail-type=NONE` to the `sbatch` commands.

5. Download the datasets on the login node, so that parallel jobs do not download them at the same time.

   ```bash
   .venv/bin/python experiments/run.py --config experiments/main.yaml --download
   ```

### 7.2 Pilot

The pilot trains every model on every task for 1,000 steps with seed 0 and two evaluation episodes, which checks that all 20 grid cells run.

```bash
.venv/bin/python experiments/run.py --config experiments/pilot.yaml --list
sbatch --time=01:00:00 --array=0-19 experiments/run.slurm experiments/pilot.yaml
```

### 7.3 Main experiment

The main experiment has 60 runs (four models, five tasks, and three seeds), and each run trains for 100,000 steps. The indices follow the order model, task, seed, and `--list` prints the full mapping.

| Indices | Model |
|---|---|
| 0–14 | CDT |
| 15–29 | Lightweight CDT |
| 30–44 | CDM |
| 45–59 | CDC |

Within each model, the tasks follow the order Car-Run, Car-Circle, Ant-Run, Drone-Run, and Drone-Circle, and the seeds follow the order 4, 42, and 420.

`runs/main/` already contains the 60 runs of the report. New runs also write to `runs/main/`, and `analyze.py` reads every finished run in this folder, so it would mix the old and the new runs. Before you start, move the runs of the report to a new folder outside `runs/main/` (in this example, `runs/main_report/`).

```bash
mv runs/main runs/main_report
.venv/bin/python experiments/run.py --config experiments/main.yaml --list
sbatch --array=0-59 experiments/run.slurm experiments/main.yaml
```

On an A100 (`gpu_a100`), the runs of the report took the following times. The training time excludes the evaluations.

| Model | Training time (mean) | Wall time per run |
|---|---|---|
| CDT | 4.77 h | 10.1–11.4 h |
| Lightweight CDT | 3.99 h | 9.2–10.5 h |
| CDM | 9.02 h | 14.5–18.3 h |
| CDC | 4.37 h | 9.6–12.4 h |

`run.slurm` allows 20 h per run. To run one index again, submit it with `--array=<index>`. The jobs read the code when they start, so do not change the files on Snellius while jobs are in the queue. Monitor the jobs with `squeue -u $USER` and `tail -f logs/codema-exp-<jobid>_<index>.out`.

### 7.4 Efficiency benchmark

The benchmark measures four quantities for the context lengths K = 10, 20, 50, and 100. These quantities are (1) the number of parameters, (2) the time of one training step on 2,048 sequences, (3) the latency of one action, and (4) the peak GPU memory. The step time is the mean over 10 steps, and the latency is the mean over 200 forward passes with batch size 1. The benchmark uses random input batches and does not train. Run all models in the same job, so that the times are comparable.

To measure Car-Run (Table 3 and Figure 2 of the report), run the benchmark on an A100 node.

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=01:00:00 --pty bash
module purge && module load 2024 2025 GCC/13.3.0 CUDA/12.8.0
.venv/bin/python -u experiments/benchmark.py
exit
```

The output is `experiments/results/benchmark.csv`.

To measure all five tasks in one batch job, submit `experiments/benchmark.slurm`. The report uses this output to check that the cost ratio of each model to CDT changes by at most 0.03 between the tasks.

```bash
sbatch experiments/benchmark.slurm
```

The output is `experiments/results/benchmark_all_tasks.csv`.

### 7.5 Analysis

The analysis has two steps, and both run on the login node without a GPU.

1. Make the tables and the CSV files from the finished runs in `runs/main/`.

   ```bash
   .venv/bin/python experiments/analyze.py --experiment main
   ```

   | File in `experiments/results/main/` | Content |
   |---|---|
   | `tables.md` | Normalized reward (×100) and normalized cost per task and model, mean ± standard deviation over seeds, for `last.pt` and `best.pt` |
   | `heldout_per_budget_per_seed.csv` | All held-out results behind the tables |
   | `budget_adaptation.csv`, `budget_adaptation_{last,best}.png` | Reward and cost per threshold (zero-shot adaptation) |
   | `training_time.csv` | Training time (without evaluation) and wall time per run |

2. Make the figures of the report with the notebook. Keep `CHECKPOINT = "last"` and `BUDGET = "all"` in the setup cell, because the report uses the final model and the mean over the three thresholds.

   ```bash
   .venv/bin/jupyter nbconvert --to notebook --execute --inplace experiments/analysis.ipynb
   ```

   You can also open the notebook in Jupyter and run all cells. The notebook reads the CSV files of step 1 and `experiments/results/benchmark.csv`, and it writes the figures to `experiments/results/main/figures/`.

### 7.6 One grid cell without Slurm

`experiments/run.py` trains and evaluates one grid cell on the current machine. The experiment files set `device: cuda`, so this needs a GPU.

```bash
.venv/bin/python experiments/run.py --config experiments/main.yaml --index 0
```

## 8. From the report to the files

| Report | Data | Made by |
|---|---|---|
| Table 1 (model configurations) | `configs/models/*.yaml`, and the `parameters` column of `experiments/results/benchmark.csv` | `experiments/benchmark.py` |
| Table 2 (cost thresholds and target returns) | `evaluation.targets` in `experiments/tasks/*.yaml` | Configuration |
| Table 3 (computational cost at K = 10) | `experiments/results/benchmark.csv` (`seq_len` 10) and `experiments/results/main/training_time.csv` | Table 2 cell of `experiments/analysis.ipynb` |
| Figure 2 (cost against context length) | `experiments/results/main/figures/context_scaling.pdf` | `experiments/analysis.ipynb` |
| Figure 3 (cost and reward per threshold) | `experiments/results/main/figures/budget_adaptation_last.pdf` | `experiments/analysis.ipynb` |
| Sections 5.3 and 5.4 (reward and cost) | `experiments/results/main/tables.md`, `budget_adaptation.csv`, and `heldout_per_budget_per_seed.csv` | `experiments/analyze.py` |
| Section 5.1 (cost ratio between tasks) | `experiments/results/benchmark_all_tasks.csv` | `experiments/benchmark.slurm` |

Figure 1 is a drawing and does not come from the code. The notebook also makes two figures that the report does not use (`efficiency.pdf` and `reward_cost_last_budget-all.pdf`). The section numbers of the notebook differ from the report, because the notebook section "5.4 Computational efficiency" is Section 5.2 of the report, and the notebook section "5.5 Zero-shot cost-budget adaptation" is Section 5.4.

## 9. Training and evaluation protocol

All models use the training settings of the OSRL implementation of CDT, which match the hyperparameters of the CDT paper. Each run trains for 100,000 gradient steps on batches of 2,048 sequences with context length K = 10. The optimizer is AdamW with a learning rate of 10⁻⁴, a weight decay of 10⁻⁴, and 500 warm-up steps, and the training uses a dropout rate of 0.1 and gradient-norm clipping at 0.25. `grad_accum.py` computes the gradient of each batch as the weighted sum of the gradients of 16 micro-batches of 128 sequences, so each step equals one step on the full batch. For CDM, the parameters that `mamba-ssm` marks with `_no_weight_decay` get no weight decay.

Each task has three cost thresholds κ, each with a target return from the OSRL implementation.

| Task | Cost thresholds κ | Target returns |
|---|---|---|
| Ant-Run | 10, 20, 40 | 700, 750, 800 |
| Car-Circle | 10, 20, 50 | 450, 500, 550 |
| Car-Run | 10, 20, 40 | 575, 575, 575 |
| Drone-Circle | 10, 20, 40 | 700, 750, 800 |
| Drone-Run | 10, 20, 40 | 400, 500, 600 |

* **Zero-shot adaptation.** One trained model serves all three thresholds, because only the initial return-to-go and cost-to-go change between the thresholds.
* **Evaluation during training.** Every 2,500 steps, the model runs 10 episodes per threshold on seeds 60000–60009. The evaluations during training select `best.pt`, which is the checkpoint with the highest mean reward among the checkpoints that stay within all three thresholds. If no checkpoint stays within all thresholds, `best.pt` is the checkpoint with the lowest ratio of cost to threshold at its worst threshold.
* **Held-out evaluation.** After training, `best.pt` and `last.pt` run 20 episodes per threshold on seeds 90000–90019, which the training never uses. The report uses `last.pt`.
* **Actions.** The agent samples each action from its Gaussian policy, as in the evaluation procedure of the CDT paper. The OSRL code uses the mean action, and `assess_checkpoint.py --mode mean` evaluates this variant.
* **Normalized reward.** R<sub>norm</sub> = 100 · (R − r<sub>min</sub>) / (r<sub>max</sub> − r<sub>min</sub>), where r<sub>min</sub> and r<sub>max</sub> are the minimum and the maximum episode reward of the DSRL dataset of the task. The CSV files store this value without the factor 100.
* **Normalized cost.** C<sub>norm</sub> = C / κ, so a normalized cost of at most 1 means that the policy stays within the threshold on average. The code adds 1 to C and κ only when κ = 0.

## 10. Re-evaluating a checkpoint

`assess_checkpoint.py` evaluates a saved checkpoint again, for example with other seeds or with mean actions. The script also gives 95% bootstrap intervals over the evaluation seeds. The output file must not exist yet.

```bash
.venv/bin/python assess_checkpoint.py runs/main/<run>/last.pt --device cuda --episodes 20 --seed 90000 --mode sample --output <file>.json
```

## 11. Results

The repository contains the results of the report in `experiments/results/`. All numbers below come from these files.

**Computational cost at K = 10 on Car-Run (Table 3 of the report).** The training time is the mean over the 15 training runs of each model.

| Model | Inference latency | Inference memory | Training step time | Training memory | Training time |
|---|---|---|---|---|---|
| CDT | 2.23 ms | 28.3 MB | 171 ms | 0.20 GB | 4.77 h |
| Lightweight CDT | 1.76 ms | 25.3 MB | 138 ms | 0.15 GB | 3.99 h |
| CDM | 5.62 ms | 25.5 MB | 326 ms | 0.13 GB | 9.02 h |
| CDC | 2.22 ms | 25.4 MB | 151 ms | 0.14 GB | 4.37 h |

At K = 100, CDM needs 46% of the step time and 31% of the training memory of CDT, and CDC needs 50% and 35%.

**Held-out results of `last.pt`.** Each run's score is the mean over its three thresholds, and each cell gives the mean ± standard deviation over the three seeds. A higher normalized reward is better, and a normalized cost of at most 1 stays within the threshold.

| Task | CDT reward | CDT cost | Lightweight CDT reward | Lightweight CDT cost | CDM reward | CDM cost | CDC reward | CDC cost |
|---|---|---|---|---|---|---|---|---|
| Ant-Run | 68.5 ± 1.1 | 0.88 ± 0.05 | 67.1 ± 0.6 | 0.89 ± 0.02 | 69.4 ± 0.7 | 0.81 ± 0.04 | 68.0 ± 0.4 | 0.93 ± 0.03 |
| Car-Circle | 70.7 ± 1.3 | 0.79 ± 0.02 | 70.7 ± 0.5 | 0.76 ± 0.07 | 69.6 ± 1.6 | 0.82 ± 0.06 | 69.3 ± 1.9 | 0.78 ± 0.13 |
| Car-Run | 97.2 ± 0.5 | 0.74 ± 0.15 | 97.1 ± 0.5 | 0.78 ± 0.07 | 89.9 ± 1.4 | 0.88 ± 0.04 | 97.7 ± 0.3 | 0.98 ± 0.17 |
| Drone-Circle | 55.9 ± 1.5 | 1.02 ± 0.06 | 55.1 ± 1.4 | 1.02 ± 0.06 | 55.9 ± 0.6 | 1.01 ± 0.03 | 55.5 ± 2.7 | 1.11 ± 0.07 |
| Drone-Run | 56.6 ± 1.9 | 1.79 ± 0.15 | 48.5 ± 10.4 | 1.65 ± 0.17 | 51.0 ± 1.1 | 1.61 ± 0.47 | 41.3 ± 2.9 | 1.88 ± 0.39 |
| Average | 69.8 ± 0.2 | 1.04 ± 0.03 | 67.7 ± 2.3 | 1.02 ± 0.02 | 67.2 ± 0.4 | 1.03 ± 0.09 | 66.4 ± 1.4 | 1.14 ± 0.10 |

The figures of the report are in `experiments/results/main/figures/`, and the results per threshold are in `experiments/results/main/budget_adaptation.csv`.

## 12. Legacy files

The following files come from earlier stages of the project. The experiments of the report do not use them, except for the helpers in `hpc.py`.

| File | Description |
|---|---|
| `hpc.py` | First launcher for the run grid in `configs/hpc.yaml`. `experiments/run.py` and `experiments/benchmark.py` import its `merge` and `read_yaml` functions, so keep this file. |
| `train.slurm` | Slurm array job for `hpc.py` on the `gpu_mig` partition |
| `configs/hpc.yaml` | Car-Run grid of CDT, Lightweight CDT, and CDM (50,000 steps, seeds 12, 0, and 1) |
| `configs/dataset.yaml` | Car-Run settings for `train.py` and `hpc.py` |
| `models/cdm.py` | Pure-PyTorch Mamba-1 version of CDM |
| `experiments/50k_runs/` | Comparison runs of CDT and CDM on Car-Run (50,000 steps, seed 12) before the main experiment. To run one, submit `sbatch --array=0 experiments/run.slurm experiments/50k_runs/cdm_compare.yaml`. |
| `logs/codema-<jobid>_<index>.out` | Slurm logs of the runs of `hpc.py` |

## 13. Acknowledgements

The CDT implementation, the data augmentation, and the datasets come from [OSRL](https://github.com/liuzuxin/OSRL) and [DSRL](https://github.com/liuzuxin/DSRL). CDM uses the Mamba-3 layer of [mamba-ssm](https://github.com/state-spaces/mamba), and CDC follows the token mixer of Decision ConvFormer.
