# Running the experiments on Snellius

## The report's Table 1 and Table 2 with one command

After the setup below (section 0), on a login node:

```bash
./run_all.sh --dry-run     # what would be submitted, and the GPU-hour upper bound
./run_all.sh               # submit everything
./run_all.sh --status      # progress
```

`run_all.sh` downloads missing datasets, then submits:

1. **Training** (`configs/experiments/tables.yaml`, 46 runs): Lightweight CDT, CDMamba and CDConvFormer on the 5 tasks
   with 3 seeds, 100k steps, batch 2048, plus one CDT run (Car-Circle, seed 0) for the CDT row of Table 2. The only
   evaluation is the final one, at kappa = 10 with 20 episodes on test seeds: the protocol of [CDT] Table 1.
2. **Latency** (one 30-minute GPU job): parameters and inference latency of the 4 models (`efficiency.py`).
3. **Analysis** (one CPU job after 1 and 2): `runs/summary/tables/table1_main.md` and `table2_summary.md`
   (plus `.tex`). Table 1 puts the CDT numbers of the paper (`configs/reported_cdt.csv`) next to ours.

Peak GPU memory and training time in Table 2 come from the Car-Circle training runs themselves (`final.json`).
Re-running `./run_all.sh` after the jobs finished resubmits only runs that failed or timed out, then the analysis.
`./run_all.sh --with-cdt` also trains our own CDT on every task (15 more runs) and adds a "CDT (ours)" column.

Cost: about 46 runs x 1-2 h, i.e. roughly 50-100 A100 GPU-hours (6k-13k SBU at 128 SBU per GPU-hour); `--with-cdt`
adds 15-30 GPU-hours. Only the time actually used is charged, so the generous `--time` (8 h) costs nothing extra.

The sections after setup describe the full study (E0-E6); they are not needed for the two tables.

## 0. Setup (once)

```bash
ssh <user>@snellius.surf.nl
cd ~                                  # or project space; checkpoints need about 50 GB in total
git clone <CoDeMa repo> CoDeMa---SRL-Group-12   # or rsync this folder from your laptop
cd CoDeMa---SRL-Group-12
mkdir -p logs                         # the .slurm files write their logs here
myquota                               # check that ~50 GB are free
accinfo                               # check the remaining budget (SBU)
```

**Python environment.** Use the group's `.venv` setup from `README.md` (`uv venv`, `uv sync`). `mamba-ssm` is
compiled during `uv sync`, so do that on a GPU node with the same modules as the jobs:

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=01:00:00 --pty bash
module purge && module load 2024 2025 GCC/13.3.0 CUDA/12.8.0
uv venv --python 3.10     # skip if .venv already exists
uv sync
.venv/bin/python -c "from mamba_ssm import Mamba3; import torch; print(torch.cuda.get_device_name(0))"
exit
```

**W&B (optional).** Not needed for the tables: `tables.yaml` has `wandb: {mode: disabled}`, and `run_all.sh`
creates the placeholder `~/.config/wandb/api-key` that `train.slurm` checks for. Only to log to the group's W&B
dashboard: put your API key in that file and set `mode: online` in the experiment file. The full-study configs
(`e*.yaml`) log online by default and need a real key.

**Datasets.** Download them once on the login node (it has internet; parallel jobs downloading the same file at
once can corrupt it):

```bash
.venv/bin/python -c "
import gymnasium as gym, dsrl
for t in ['OfflineAntRun-v0', 'OfflineCarCircle-v0', 'OfflineCarRun-v0', 'OfflineDroneCircle-v0', 'OfflineDroneRun-v0']:
    d = gym.make(t).get_dataset(); print(t, len(d['rewards']))"
```

They land in `~/.dsrl/datasets` (set `DSRL_DATASET_DIR` to put them elsewhere; then export it in your
`~/.bashrc` so the jobs see it too).

## Full study (optional)

One SLURM array per experiment file, in this order:

```
E0  ->  E1, E3, E6full (+ E6long)  ->  evaluations (E1/E3 test, E4, E5)  ->  efficiency (E6)  ->  E10 + analysis
```

All `sbatch` commands below pass `--mail-type=NONE`: `train.slurm` has Deniz's address in `--mail-user`, so
without it every run would send him two emails. Use `--mail-user=<you> --mail-type=END,FAIL` if you want mails.
`analyze.py --all` writes every table and figure of the full study.

## 1. E0: pilot and learning rate (about 10 GPU-hours)

```bash
sbatch --array=0-19 --partition=gpu_a100 --cpus-per-task=18 --time=01:00:00 --mail-type=NONE \
       train.slurm configs/experiments/e0_pilot.yaml
sbatch --array=0-7  --partition=gpu_a100 --cpus-per-task=18 --time=04:00:00 --mail-type=NONE \
       train.slurm configs/experiments/e0_tune.yaml
```

Watch them:

```bash
squeue -u $USER                                   # running / pending
sacct -j <jobid> --format=JobID%20,State,Elapsed  # per array index
tail -f logs/codema-<jobid>_<index>.out           # one log per index
```

When both are done:

```bash
.venv/bin/python analyze.py --all
cat runs/summary/e0_report.md
```

The report shows:
1. **Pilot**: whether every model ran on every task, the time per step, and a suggested `--time` for the 100k-step
   runs. Every pilot run must say "yes" before you continue. A failed index: read its log, fix, resubmit just that
   index with `--array=<index>`.
2. **Learning rate**: the chosen lr per model. If a model's choice is 3e-4, add it to its model file before E1:
   ```yaml
   # configs/models/cdmamba.yaml
   trainer:
     loss_cost_weight: 0.02
     loss_state_weight: 0.0
     learning_rate: 0.0003
   ```

Below, `<T>` is the suggested `--time` from the report (e.g. `03:00:00`).

## 2. Training: E1, E3, E6full (+ E6long)

`%16` limits each array to 16 GPUs at once; raise or lower it to taste.

```bash
sbatch --array=0-3%4     --partition=gpu_a100 --cpus-per-task=18 --time=<T> --mail-type=NONE train.slurm configs/experiments/e6full.yaml
sbatch --array=0-74%16   --partition=gpu_a100 --cpus-per-task=18 --time=<T> --mail-type=NONE train.slurm configs/experiments/e1.yaml
sbatch --array=0-134%16  --partition=gpu_a100 --cpus-per-task=18 --time=<T> --mail-type=NONE train.slurm configs/experiments/e3.yaml
# optional, K = 50 (longer sequences, give it about 2x <T>):
sbatch --array=0-11%12   --partition=gpu_a100 --cpus-per-task=18 --time=<2T> --mail-type=NONE train.slurm configs/experiments/e6long.yaml
# optional, CDT in E1/E3:
sbatch --array=0-24%16   --partition=gpu_a100 --cpus-per-task=18 --time=<T> --mail-type=NONE train.slurm configs/experiments/e1_cdt.yaml
sbatch --array=0-44%16   --partition=gpu_a100 --cpus-per-task=18 --time=<T> --mail-type=NONE train.slurm configs/experiments/e3_cdt.yaml
```

Each finished run has `runs/<name>_seed<s>_<timestamp>/final.json`. Check for failures:

```bash
sacct -j <jobid> --format=JobID%20,State,Elapsed | grep -v COMPLETED
```

Runs do not resume. A run that failed or hit the time limit is simply submitted again (`--array=<index>`, larger
`--time` for a timeout). The new run gets a new directory; unfinished directories (no `final.json`) are ignored
everywhere.

## 3. Evaluations (after the training of their source finished)

These are short (a few minutes each) and run on MIG slices (`eval.slurm`, `gpu_mig`):

```bash
sbatch --array=0-74%16  --mail-type=NONE eval.slurm configs/experiments/e1_test.yaml   # after E1
sbatch --array=0-74%16  --mail-type=NONE eval.slurm configs/experiments/e4.yaml        # after E1
sbatch --array=0-134%16 --mail-type=NONE eval.slurm configs/experiments/e3_test.yaml   # after E3
sbatch --array=0-119%16 --mail-type=NONE eval.slurm configs/experiments/e5.yaml        # after E1 and E3
sbatch --array=0-11     --mail-type=NONE eval.slurm configs/experiments/e6long_test.yaml  # after E6long
# optional CDT: e1_cdt_test (0-24), e4_cdt (0-24), e3_cdt_test (0-44), e5_cdt (0-39)
```

To queue them right away, chain them to the training arrays with `--dependency=afterany:<training jobid>`.
An index whose run is not finished fails with "No finished run for ..."; submit that index again later.
Evaluations that already exist (`runs/<run>/eval_<experiment>.json`) are skipped, so resubmitting a whole array
is safe. If `gpu_mig` is busy or the Mamba-3 kernels fail there, add `--partition=gpu_a100 --cpus-per-task=18`.

## 4. Efficiency (E6), after E6full finished

One exclusive A100 node, at most 4 hours (it measures all 4 models; 3 repetitions):

```bash
sbatch --mail-type=NONE efficiency.slurm
```

Results: `runs/profile/efficiency.csv` and `runs/profile/hardware.json`. To repeat it, delete `runs/profile/` first
(rows are appended).

## 5. E10 and analysis (login node, no GPU)

```bash
.venv/bin/python plot_datasets.py     # E10: runs/summary/figures/fig8_datasets.*
.venv/bin/python analyze.py --all     # everything else
```

`runs/summary/` then holds:

| File | Content |
|---|---|
| `tables/table1_main.{md,tex}` | E1, kappa = 10, per task (Table 1) |
| `tables/table2_efficiency.{md,tex}` | E6: params, size, train time (projected and actual), memory (GB), latency, recurrent, relative to CDT |
| `tables/table3_ablations.{md,tex}` | E3 (Table 3) |
| `tables/table4_multithreshold.{md,tex}` | E1, kappa = 10/20/40 with violation rate and excess cost (Table 4) |
| `figures/fig1_zero_shot` ... `fig8_datasets` | Fig. 1 (E4), 2 (E5), 3 (scatter), 4 (learning curves), 5 (context scaling), 6 (efficiency trade-off), 7 (ablations), 7b (efficiency bars), 8 (datasets) |
| `results.csv`, `episodes.csv`, `val_logs.csv`, `train_runs.csv`, `efficiency.csv`, `datasets.csv` | all numbers behind them |
| `e0_report.md` | E0 |

Copy the summary to your laptop with `rsync -av <user>@snellius.surf.nl:~/CoDeMa---SRL-Group-12/runs/summary ./`.

## Cost of the full study

Roughly 250-500 A100 GPU-hours for E0 + E1 + E3 + E6 + evaluations (CDT and E6long add about 70-150 and 25-50).
The E0 report replaces this guess with measured numbers: GPU-hours of a training array = runs x hours per run.
On `gpu_a100` one GPU-hour costs 128 SBU (check the current rates in the Snellius documentation).
