# CoDeMa---SRL-Group-12

How to set up below :)

we have the following tree structure of the project:

```text
(CoDeMa---SRL-Group-12) dozen@DeniZenbook:~/SRL/CoDeMa---SRL-Group-12$ tree .
.
├── README.md
|
├── configs
│   ├── dataset.yaml            --- where we set up data-related configuration
│   ├── datasets                --- the same, one file per task (5 DSRL tasks)
│   ├── experiments             --- experiment definitions in the hpc.yaml format (see Experiments below)
│   ├── models                  --- each of the model init/configs go here, i placed some examples inside
│   │   ├── cdt.yaml
│   │   ├── cdm.yaml
│   │   ├── cdt_small.yaml
│   │   ├── cdt_light.yaml          --- parameter-matched CDT (2 layers)
│   │   ├── cdconv.yaml
│   │   └── cdmamba.yaml
│   └── train.yaml              --- training hyperparams
|
├── models
│   └── cdt.py                  --- our wrapper from OSRL
│   └── cdm.py                  --- mamba extension on CDT baseline
│   └── cdconv.py               --- CDConvFormer: per-modality causal convolution blocks
│   └── cdmamba.py              --- CDMamba: Mamba-3 (SISO) + MLP blocks, pure-PyTorch Mamba-3 for CPU
|
├── pyproject.toml
├── train.py                    --- main training calls, we can directly pass a model config to train it
├── hpc.py, train.slurm         --- one training run of an experiment config per array index
├── hpc_eval.py, eval.slurm     --- one evaluation of a finished run per array index (test, E4, E5)
├── efficiency.py, efficiency.slurm --- E6 efficiency measurements
├── plot_datasets.py            --- E10 dataset plots
├── run_all.sh                  --- one command for the report's Table 1 and Table 2 (Snellius)
├── analyze.py                  --- tables and figures from runs/ -> runs/summary/
├── utils.py                    --- logging and wandB related stuff
└── uv.lock
```


To set up the project:

1. Ensure Linux/WSL, the training envs/gym won't work with Windows
2. Ensure uv

Then,
1. Clone the project and `cd` into it,
2. Run:
```shell
uv venv --python /usr/bin/python3.10
uv sync
source .venv/bin/activate
```

The `pyproject.toml` is set in such way that it automatically sets up the additional libraries from git.

3. Run a model:
```shell
uv run python train.py --model configs/models/cdt.yaml 
```
## Logging
We are now using wandB because it is nice way of monitoring the training and evaluation metrics which can be observed simultaneously by all group members
https://forge.coreweave.com/wandb/dozen/constrained-decision-models/workspace

When running for the first time wandB will ask you to create an account and paste the API key (if there is an error with the key length run `uv sync`)

We now log every training evaluation a model instances/objects under `runs/`


---


Todos/nice to have(s):
- a vis utility, both for training and environment
- Find a way to visualize environment (i.e.,car driving)

---

## Experiments: CDT vs Lightweight CDT, CDMamba, CDConvFormer

Every experiment is a file in `configs/experiments/` with the format of `configs/hpc.yaml`: one entry per run
(`# Index N`), submitted as a SLURM array. An entry may set its own `dataset:` (task) and `dataset_overrides:`
next to `model_overrides:` / `training_overrides:`. Run names are `<model>[_<variant>]`, e.g. `cdmamba_noaug`.

| Experiment | Config | Runs | Script |
|---|---|---|---|
| E0 pilot (2k steps, all models x tasks) | `e0_pilot.yaml` | 20 | `train.slurm` |
| E0 learning rate (Car-Circle, 30k steps) | `e0_tune.yaml` | 8 | `train.slurm` |
| E1 main comparison (5 tasks x 5 seeds, 100k steps) | `e1.yaml` | 75 | `train.slurm` |
| E1 test evaluation (kappa 10/20/40) | `e1_test.yaml` | 75 | `eval.slurm` |
| E3 ablations (noaug / noent / det, 5 tasks x 3 seeds) | `e3.yaml`, `e3_test.yaml` | 135 + 135 | `train.slurm`, `eval.slurm` |
| E4 unseen budgets kappa 5 ... 80 (E1 models) | `e4.yaml` | 75 | `eval.slurm` |
| E5 unrealistic reward targets (E1 + E3-noaug models) | `e5.yaml` | 120 | `eval.slurm` |
| E6full dedicated runs (Car-Circle, all 4 models) | `e6full.yaml` | 4 | `train.slurm` |
| E6 efficiency (size, time, memory, latency, scaling) | – | 1 job | `efficiency.slurm` |
| E6long K = 50 (optional) | `e6long.yaml`, `e6long_test.yaml` | 12 + 12 | `train.slurm`, `eval.slurm` |
| E10 dataset plots | – | – | `python plot_datasets.py` |
| CDT in E1/E3/E4/E5 (optional) | `*_cdt*.yaml` | 25/45/... | as above |

- Training: batch 2048 without gradient accumulation (`microbatch_size: 2048`) on full A100s, fp32 (the Mamba-3 mixer
  runs in bf16, as in `cdm_3.py`), lr 1e-4 (per model: `trainer: learning_rate:` in `configs/models/<model>.yaml`).
- Validation during training: every 5k steps, 5 episodes per target on seeds 500000+ (learning curves, `best.pt`).
- Reported numbers: `hpc_eval.py` evaluates the final model (`last.pt`) on test seeds 1000000+, 20 episodes per
  target, mean action, and writes `runs/<run>/eval_<experiment>.json`. `best.pt` is not used for reporting.
- `final.json` also has the wall-clock split (`time`) and the peak GPU memory of the run.
- `analyze.py` writes Table 1 and Table 2 to `runs/summary/tables/`; `analyze.py --all` also the other tables, figures and `e0_report.md`.

Only the report's Table 1 and Table 2: `./run_all.sh` on a Snellius login node submits everything (`configs/experiments/tables.yaml`, latency, analysis) with one command.
The step-by-step instructions for Snellius are in `SNELLIUS.md`.
