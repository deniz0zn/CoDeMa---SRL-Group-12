#!/bin/bash
# Everything for the report's Table 1 and Table 2 with one command (run it on a Snellius login node):
#
#   ./run_all.sh --dry-run      # show what would be submitted and the GPU-hour upper bound
#   ./run_all.sh                # submit: training array -> latency job -> analysis job (waits for both)
#   ./run_all.sh --with-cdt     # also train our own CDT on all tasks (15 more runs) for Table 1
#   ./run_all.sh --status       # finished runs and queued jobs
#
# Re-running is safe: only runs without a finished run directory (final.json) are submitted again, and the
# latency job is skipped once runs/profile/efficiency.csv exists. Results: runs/summary/tables/.
# Settings (environment variables): TIME (per training run, default 08:00:00; only the time used is charged),
# PARALLEL (GPUs at once, default 16), CPU_PARTITION (analysis job, default rome).

set -euo pipefail
cd "$(dirname "$0")"

TIME="${TIME:-08:00:00}"
PARALLEL="${PARALLEL:-16}"
CPU_PARTITION="${CPU_PARTITION:-rome}"
DRY=0; WITH_CDT=0; STATUS=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        --with-cdt) WITH_CDT=1 ;;
        --status) STATUS=1 ;;
        *) echo "unknown option $arg" >&2; exit 1 ;;
    esac
done

PY=.venv/bin/python
MODULES="module purge; module load 2024 2025 GCC/13.3.0 CUDA/12.8.0"
CONFIGS=(configs/experiments/tables.yaml)
[[ $WITH_CDT == 1 ]] && CONFIGS+=(configs/experiments/tables_cdt.yaml)

# indices of a config that have no finished run yet (comma-separated)
missing() {
    $PY - "$1" <<'EOF'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(open(sys.argv[1]))
done = set()
for path in Path("runs").glob("*/config.yaml"):
    if (path.parent / "final.json").exists():
        c = yaml.safe_load(path.read_text())
        done.add((c.get("experiment"), c.get("task"), c.get("model_name"), c.get("seed")))

left = []
for i, entry in enumerate(config["runs"]):
    overrides = {**config.get("training_overrides", {}), **entry.get("training_overrides", {})}
    task = yaml.safe_load(open(entry.get("dataset", config["dataset"])))["task"]
    if (overrides["experiment"], task, entry["name"], overrides["seed"]) not in done:
        left.append(str(i))
print(",".join(left))
EOF
}

if [[ $STATUS == 1 ]]; then
    for config in "${CONFIGS[@]}"; do
        total=$($PY -c "import yaml; print(len(yaml.safe_load(open('$config'))['runs']))")
        left=$(missing "$config")
        echo "$config: $((total - $(echo "$left" | awk -F, '{print ($0 == "" ? 0 : NF)}'))) of $total runs finished"
    done
    [[ -f runs/profile/efficiency.csv ]] && echo "latency: measured" || echo "latency: not measured yet"
    [[ -f runs/summary/tables/table1_main.md ]] && echo "tables: runs/summary/tables/table1_main.md, table2_summary.md"
    squeue -u "$USER" -o "%.18i %.20j %.8T %.10M %R" | grep -E "JOBID|codema" || true
    exit 0
fi

# ------------------------------------------------------------------ checks (login node)
[[ -x $PY ]] || { echo "No .venv: set it up first (README.md, uv sync on a GPU node)" >&2; exit 1; }
if [[ ! -s ~/.config/wandb/api-key ]]; then
    # train.slurm refuses to start without this file; W&B is disabled in tables*.yaml, so any content works
    if grep -q "mode: online" "${CONFIGS[@]}"; then
        echo "W&B is online in ${CONFIGS[*]} but ~/.config/wandb/api-key is missing" >&2
        exit 1
    fi
    mkdir -p ~/.config/wandb && echo "none" > ~/.config/wandb/api-key && chmod 600 ~/.config/wandb/api-key
    echo "Created a placeholder ~/.config/wandb/api-key (W&B is disabled, no key needed)"
fi
if squeue -u "$USER" -h -o "%j" | grep -q "^codema"; then
    echo "CoDeMa jobs are still queued or running (./run_all.sh --status); not submitting twice." >&2
    exit 1
fi
mkdir -p logs

[[ $DRY == 1 ]] || echo "Checking the datasets (downloads the missing ones; the login node has internet) ..."
[[ $DRY == 1 ]] || $PY - <<'EOF' 2>&1 | grep -v "safety_gymnasium\|failed to import"
import gymnasium as gym
import dsrl  # noqa: F401
for task in ["OfflineAntRun-v0", "OfflineCarCircle-v0", "OfflineCarRun-v0", "OfflineDroneCircle-v0", "OfflineDroneRun-v0"]:
    env = gym.make(task)
    print(f"  {task}: {len(env.get_dataset()['rewards'])} transitions")
    env.close()
EOF

# ------------------------------------------------------------------ plan
submit() {  # prints the command; runs it unless --dry-run; echoes the job id
    if [[ $DRY == 1 ]]; then
        echo "  sbatch $*" >&2
        echo "DRYRUN"
    else
        sbatch --parsable "$@"
    fi
}

hours=$(echo "$TIME" | awk -F: '{print $1 + $2 / 60}')
dependencies=""
total=0
echo
for config in "${CONFIGS[@]}"; do
    left=$(missing "$config")
    if [[ -z $left ]]; then
        echo "$config: all runs finished"
        continue
    fi
    count=$(echo "$left" | awk -F, '{print NF}')
    total=$((total + count))
    echo "$config: $count runs to train (indices $left)"
    job=$(submit --array="$left%$PARALLEL" --partition=gpu_a100 --cpus-per-task=18 --time="$TIME" \
                 --mail-type=NONE train.slurm "$config")
    dependencies+=":$job"
done

if [[ -f runs/profile/efficiency.csv ]]; then
    echo "latency: already measured (delete runs/profile/ to measure again)"
else
    echo "latency: 1 short GPU job (parameters + inference latency of the 4 models)"
    job=$(submit --job-name=codema-latency --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:30:00 \
                 --output=logs/%x-%j.out --mail-type=NONE \
                 --wrap="$MODULES; $PY -u efficiency.py --device cuda --what size infer --precisions fp32")
    dependencies+=":$job"
fi

echo "analysis: 1 CPU job after the jobs above -> runs/summary/tables/table1_main.md, table2_summary.md"
analysis=$(submit --job-name=codema-analysis --partition="$CPU_PARTITION" --ntasks=1 --cpus-per-task=16 \
                  --time=00:30:00 --output=logs/%x-%j.out --mail-type=NONE \
                  ${dependencies:+--dependency=afterany$dependencies} \
                  --wrap="$MODULES; $PY -u analyze.py")

echo
echo "GPU-hour upper bound: $total runs x $hours h + 0.5 h = $(echo "$total $hours" | awk '{print $1 * $2 + 0.5}') GPU-h"
echo "(only the time actually used is charged; expected about 1-2 h per run)"
if [[ $DRY == 0 ]]; then
    echo "Submitted. Follow with: ./run_all.sh --status   (logs in logs/)"
    echo "Re-run ./run_all.sh after it finished to retrain only the runs that failed."
fi
