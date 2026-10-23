# CoDeMa: Constrained Decision Models for Offline Safe RL

This project compares four sequence models for offline safe reinforcement learning. All four share the cost
conditioning, Gaussian policy, entropy regularisation and Pareto-frontier data augmentation of the Constrained
Decision Transformer (CDT); they differ only in the sequence model:

| Model | Config | Sequence model | Params (CarRun) |
|---|---|---|---|
| CDT | `configs/models/cdt.yaml` | OSRL CDT, 3 Transformer layers | 625,805 |
| Lightweight CDT | `configs/models/cdt_light.yaml` | 2 Transformer layers | 427,533 |
| CDMamba (CDM) | `configs/models/cdm.yaml` | Mamba-3 (MIMO) replaces attention; 2x MLP kept | 438,629 |
| CDConvFormer (CDC) | `configs/models/cdconv.yaml` |  |  |

Tasks: the five BulletSafetyGym tasks from the CDT paper (`OfflineCarRun-v0`, `OfflineCarCircle-v0`,
`OfflineAntRun-v0`, `OfflineDroneRun-v0`, `OfflineDroneCircle-v0`), with the DSRL offline datasets
([OSRL](https://github.com/liuzuxin/OSRL), [DSRL](https://github.com/liuzuxin/DSRL)).

## Project structure

```text
.
├── configs/
│   ├── models/              # one YAML per model (cdt, cdt_light, cdm, cdconv)
│   ├── train.yaml           # shared training settings (batch 2048, lr 1e-4, ...)
│   └── dataset.yaml         # CarRun dataset settings for single runs with train.py
├── models/
│   ├── cdt.py               # OSRL's CDT
│   ├── cdm_3.py             # CDMamba (Mamba-3 block + MLP)
│   ├── cdconv.py            # CDConvFormer
│   └── cdm.py               # legacy pure-PyTorch Mamba-1 variant (not used in the experiments)
├── experiments/
│   ├── tasks/               # one config per task (OSRL CDT settings + evaluation targets)
│   ├── main.yaml            # main experiment grid: models x tasks x seeds
│   ├── pilot.yaml           # short version of main.yaml (smoke test + timing)
│   ├── run.py               # runs one grid cell: training, then held-out evaluation
│   ├── run.slurm            # Snellius array job, one index per grid cell
│   ├── benchmark.py         # efficiency: params, train-step time, latency, memory, context sweep
│   └── analyze.py           # result tables and figures
├── train.py                 # training loop (used by experiments/run.py)
├── grad_accum.py            # training step with gradient accumulation, and evaluation
├── assess_checkpoint.py     # re-evaluates a saved checkpoint
├── utils.py                 # config loading, logging (CSV + W&B), checkpoints
├── hpc.py, train.slurm      # older single-task launcher (configs/hpc.yaml)
└── pyproject.toml, uv.lock  # pinned environment
```

## 1. Setup

Requirements: Linux or WSL (the Bullet environments don't run on Windows), Python 3.10 and
[uv](https://docs.astral.sh/uv/). CDMamba needs an NVIDIA GPU with CUDA 12.8; the other models also run on CPU.

```bash
git clone https://github.com/deniz0zn/CoDeMa---SRL-Group-12.git
cd CoDeMa---SRL-Group-12
uv venv --python 3.10
uv sync
```

`uv sync` installs the pinned versions from `uv.lock`, including OSRL, DSRL and `mamba-ssm` from git. `mamba-ssm`
is compiled during the sync, so run it on a machine with CUDA available (on Snellius: a GPU node, see section 4).

Download the five datasets once (they are stored in `~/.dsrl/datasets`):

```bash
.venv/bin/python experiments/run.py --config experiments/main.yaml --download
```

## 2. Logging

Every run writes to `runs/<experiment>/<model>_<Task>_seed<seed>_<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | full config of the run |
| `training.csv` | losses, gradient norm, learning rate, `time/train_seconds` |
| `evaluation.csv` | evaluation during training: reward, cost, normalised reward/cost per budget |
| `episodes.csv` | every evaluation episode |
| `best.pt`, `last.pt` | best checkpoint (feasible on all budgets, then highest reward) and final checkpoint |
| `final.json` | summary of the best and last evaluation |
| `heldout_best.json`, `heldout_last.json` | held-out evaluation of both checkpoints (20 episodes per budget) |

Runs are also logged to Weights & Biases, project `constrained-decision-models`
(<https://wandb.ai/dozen/constrained-decision-models>), grouped as `<experiment>/<task>/<model>`.

- **Locally:** run `.venv/bin/wandb login` once, or set `mode: offline` / `mode: disabled` under `wandb:` in
  `configs/train.yaml` (or in the `training_overrides` of an experiment file).
- **On Snellius:** the Slurm scripts read the API key from `~/.config/wandb/api-key` (see section 4).

## 3. A single local run

`train.py` trains one model on CarRun with `configs/train.yaml`:

```bash
.venv/bin/python train.py --model configs/models/cdt.yaml
```

`configs/train.yaml` uses `device: cuda`. For a quick test on CPU, set `device: cpu` and a small `steps` there
(CDT, Lightweight CDT and CDConvFormer only; CDMamba needs CUDA).

## 4. Reproducing the experiments on Snellius

### Setup (once)

```bash
cd ~ && git clone https://github.com/deniz0zn/CoDeMa---SRL-Group-12.git && cd CoDeMa---SRL-Group-12
mkdir -p logs                                   # Slurm writes the job logs here
```

Build the environment on a GPU node with the same modules as the jobs, so `mamba-ssm` compiles against CUDA 12.8:

```bash
srun --partition=gpu_mig --gpus=1 --cpus-per-task=9 --time=01:00:00 --pty bash
module purge && module load 2024 2025 GCC/13.3.0 CUDA/12.8.0
uv venv --python 3.10 && uv sync
exit
```

Save your W&B API key (from <https://wandb.ai/authorize>) where the Slurm scripts expect it:

```bash
mkdir -p ~/.config/wandb && nano ~/.config/wandb/api-key && chmod 600 ~/.config/wandb/api-key
```

Download the datasets on the login node (section 1), so parallel jobs never download at the same time.

`experiments/run.slurm` sends job e-mails to `--mail-user`; change it to your own address, or add
`--mail-type=NONE` (or `FAIL`) to the `sbatch` commands below.

### Pilot (about 15 minutes per job)

Every model on every task for 1000 steps: checks that everything runs and gives the time per step.

```bash
.venv/bin/python experiments/run.py --config experiments/pilot.yaml --list
sbatch --time=01:00:00 --array=0-19 experiments/run.slurm experiments/pilot.yaml
```

### Main experiment

4 models x 5 tasks x 3 seeds = 60 runs, 100k steps each. Indices are ordered model, task, seed
(`--list` prints the mapping): 0-14 CDT, 15-29 Lightweight CDT, 30-44 CDMamba, 45-59 CDConvFormer.

```bash
.venv/bin/python experiments/run.py --config experiments/main.yaml --list
sbatch --array=0-59 experiments/run.slurm experiments/main.yaml
```

On a MIG slice (A100 3g.20gb) a run takes about 10-11 h (CDT, Lightweight CDT) and about 15 h (CDMamba);
`run.slurm` allows 20 h. A single index can be resubmitted with `--array=<index>`. Jobs read the code when they
start, so don't change files on Snellius while jobs are queued.

Monitor with `squeue -u $USER` and `tail -f logs/codema-exp-<jobid>_<index>.out`.

### Efficiency benchmark

Parameters, training-step time, per-action latency and peak GPU memory of all models, for context lengths
K = 10, 20, 50, 100 (synthetic batches, no training). Run all models in the same job so the timings are comparable:

```bash
srun --partition=gpu_mig --gpus=1 --cpus-per-task=9 --time=01:00:00 --pty bash
module purge && module load 2024 2025 GCC/13.3.0 CUDA/12.8.0
.venv/bin/python -u experiments/benchmark.py
exit
```

Output: `experiments/results/benchmark.csv`.

To benchmark all five tasks in one batch job (output: `experiments/results/benchmark_all_tasks.csv`):

```bash
sbatch experiments/benchmark.slurm
```

### Analysis

After the main runs have finished (login node, no GPU needed):

```bash
.venv/bin/python experiments/analyze.py --experiment main
```

Output in `experiments/results/main/`:

| File | Content |
|---|---|
| `tables.md` | normalised reward (x100) and cost per task and model, mean ± sd over seeds, for `last.pt` and `best.pt` |
| `budget_adaptation.csv`, `budget_adaptation_{last,best}.png` | reward and cost per budget (zero-shot budget adaptation) |
| `heldout_per_budget_per_seed.csv` | all held-out numbers behind the tables |
| `training_time.csv` | training time (without evaluation) and wall time per run |

## Evaluation protocol

- **Budgets:** three cost budgets per task with OSRL's reward targets, e.g. CarRun (575, 10), (575, 20), (575, 40);
  CarCircle uses 10/20/50. One trained model is evaluated on all budgets (zero-shot budget adaptation).
- **During training:** every 2,500 steps, 10 episodes per budget on seeds 60000+; used to select `best.pt`.
- **Held-out:** after training, `best.pt` and `last.pt` are evaluated with 20 episodes per budget on seeds 90000+,
  which are never used during training.
- **Actions** are sampled from the Gaussian policy, following Algorithm 2 of the CDT paper (OSRL's code uses the mean;
  `assess_checkpoint.py --mode mean` evaluates that variant).
- **Normalisation** (DSRL): reward = (R - R_min) / (R_max - R_min) with the dataset's reward range; cost = C / budget,
  so a normalised cost of at most 1 is within budget.

To re-evaluate a checkpoint, e.g. with other seeds or mean actions:

```bash
.venv/bin/python assess_checkpoint.py runs/main/<run>/last.pt --device cuda --episodes 20 --seed 90000 --mode mean --output <file>.json
```
