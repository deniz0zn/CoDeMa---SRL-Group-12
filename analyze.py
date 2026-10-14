"""Tables and figures of the study from runs/ (no GPU needed; run on the login node or locally).

    python analyze.py [--runs runs] [--out runs/summary] [--reported-cdt reported_cdt.csv]

Reads every runs/<run>/{config.yaml, final.json, evaluation.csv, eval_<experiment>.json} and
runs/profile/efficiency.csv. Writes to runs/summary/:
    episodes.csv, results.csv, val_logs.csv, train_runs.csv, efficiency.csv, e0_report.md,
    tables/table{1,2,3,4}_*.{md,tex}, figures/fig*.{pdf,png}
By default only Table 1 and Table 2 (the report's tables); --all also writes Tables 3-4, the E0 report and figures.
Normalized reward R_norm = 100 (R - r_min) / (r_max - r_min) with the minimum and maximum reward return of the
task's dataset ([CDT] Sec. 5); normalized cost C_norm = C / kappa. Missing runs show as "–"; nothing is filled in.
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


ROOT = Path(__file__).resolve().parent

MODELS = ["cdt", "cdt_light", "cdmamba", "cdconv"]
KNOWN = sorted(MODELS + ["cdt_small", "cdm"], key=len, reverse=True)
DISPLAY = {"cdt": "CDT", "cdt_light": "Lightweight CDT", "cdmamba": "CDMamba", "cdconv": "CDConvFormer"}
TASKS = ["OfflineAntRun-v0", "OfflineCarCircle-v0", "OfflineCarRun-v0", "OfflineDroneCircle-v0", "OfflineDroneRun-v0"]
SHORT = {"OfflineAntRun-v0": "Ant-Run", "OfflineCarCircle-v0": "Car-Circle", "OfflineCarRun-v0": "Car-Run",
         "OfflineDroneCircle-v0": "Drone-Circle", "OfflineDroneRun-v0": "Drone-Run"}
VARIANTS = [("noaug", "w/o augmentation"), ("noent", "w/o entropy"), ("det", "deterministic")]
COLORS = {"cdt": "#2a78d6", "cdt_light": "#eb6834", "cdmamba": "#1baf7a", "cdconv": "#eda100"}
MARKERS = {"cdt": "o", "cdt_light": "s", "cdmamba": "D", "cdconv": "^"}
MUTED = "#6b6b6b"
MISSING = "–"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


# ------------------------------------------------------------------ helpers
def split_name(name):
    """Run name -> (model, variant), e.g. cdt_light_noaug -> (cdt_light, noaug)."""
    for model in KNOWN:
        if name == model or name.startswith(model + "_"):
            return model, name[len(model) + 1:]
    return name, ""


def mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def std(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def fmt(value, spread=None, digits=2):
    if value is None or np.isnan(value):
        return MISSING
    text = f"{value:.{digits}f}"
    if spread is not None and not np.isnan(spread):
        text += f" ± {spread:.{digits}f}"
    return text


def group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalization(task, cache={}):
    """Minimum and maximum reward return of the task's offline dataset."""
    if task not in cache:
        import gymnasium as gym
        from hpc_eval import dataset_returns

        env = gym.make(task)
        rewards, _ = dataset_returns(env.unwrapped)
        env.close()
        cache[task] = (float(rewards.min()), float(rewards.max()))
    return cache[task]


def r_norm(reward, task):
    r_min, r_max = normalization(task)
    return 100.0 * (reward - r_min) / (r_max - r_min)


def rank_key(r, c):
    """[CDT]: safe beats unsafe; safe by higher reward, unsafe by lower cost."""
    if np.isnan(r) or np.isnan(c):
        return (2, 0.0)
    return (0, -r) if c <= 1.0 else (1, c)


# ------------------------------------------------------------------ loading
def load(runs_dir):
    training, episodes, validation = [], [], []

    for config_path in sorted(runs_dir.glob("*/config.yaml")):
        directory = config_path.parent
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model, variant = split_name(config["model_name"])
        key = dict(exp=config.get("experiment"), model=model, variant=variant, task=config["task"], seed=config["seed"])

        final = json.loads((directory / "final.json").read_text()) if (directory / "final.json").exists() else None
        time = (final or {}).get("time", {})
        training.append(dict(**key, run=directory.name, finished=final is not None, steps=config["training"]["steps"],
                             wall_seconds=time.get("wall_seconds"), training_seconds=time.get("training_seconds"),
                             evaluation_seconds=time.get("evaluation_seconds"),
                             peak_gpu_memory_mb=(final or {}).get("peak_gpu_memory_mb")))

        if (directory / "evaluation.csv").exists():
            with (directory / "evaluation.csv").open(encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    kappa = float(row["target_cost"])
                    validation.append(dict(**key, run=directory.name, step=int(row["step"]),
                                           elapsed_seconds=float(row["elapsed_seconds"]), target_C=kappa,
                                           R=float(row["mean_reward"]), C=float(row["mean_cost"]),
                                           R_norm=r_norm(float(row["mean_reward"]), config["task"]),
                                           C_norm=float(row["mean_cost"]) / kappa))

        if config["training"]["evaluation"].get("report") and final and (directory / "episodes.csv").exists():
            # the final evaluation of the run is its test evaluation (configs/experiments/tables*.yaml)
            r_min, r_max = normalization(config["task"])
            with (directory / "episodes.csv").open(encoding="utf-8") as file:
                for e in csv.DictReader(file):
                    if int(e["step"]) != config["training"]["steps"]:
                        continue
                    kappa, reward, cost = float(e["budget"]), float(e["reward"]), float(e["cost"])
                    episodes.append(dict(**key, run=directory.name, episode_seed=int(e["seed"]),
                                         target_R=float(e["target_return"]), target_C=kappa, R=reward, C=cost,
                                         R_norm=100.0 * (reward - r_min) / (r_max - r_min), C_norm=cost / kappa,
                                         violation=float(cost > kappa), excess=max(0.0, cost - kappa),
                                         length=int(e["length"]), data_r_max=r_max, data_c_max=None))

        for path in sorted(directory.glob("eval_*.json")):
            result = json.loads(path.read_text())
            r_min, r_max = result["dataset"]["r_min"], result["dataset"]["r_max"]
            for e in result["episodes"]:
                kappa = e["budget"]
                episodes.append(dict(**{**key, "exp": result["experiment"]}, run=directory.name, episode_seed=e["seed"],
                                     target_R=e["target_return"], target_C=kappa, R=e["reward"], C=e["cost"],
                                     R_norm=100.0 * (e["reward"] - r_min) / (r_max - r_min), C_norm=e["cost"] / kappa,
                                     violation=float(e["cost"] > kappa), excess=max(0.0, e["cost"] - kappa),
                                     length=e["length"], data_r_max=result["dataset"]["r_max"],
                                     data_c_max=result["dataset"]["c_max"]))

    # one row per (run, target): episode means
    results = []
    keys = ["exp", "model", "variant", "task", "seed", "target_R", "target_C"]
    for k, rows in group(episodes, keys).items():
        results.append({**dict(zip(keys, k)), **{m: mean([r[m] for r in rows]) for m in
                                                 ("R", "C", "R_norm", "C_norm", "violation", "excess")},
                        "episodes": len(rows), "data_c_max": rows[0]["data_c_max"], "data_r_max": rows[0]["data_r_max"]})

    return training, results, episodes, validation


def load_efficiency(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["value"] = float(row["value"]) if row["value"] not in ("", None) else np.nan
    return rows


def eff(rows, model, metric, stat, **match):
    """Mean and std over repetitions of one efficiency measurement."""
    values = [r["value"] for r in rows if r["model"] == model and r["metric"] == metric and r["stat"] == stat
              and all(str(r[k]) == str(v) for k, v in match.items())]
    return mean(values), std(values)


# ------------------------------------------------------------------ tables
class Table:
    def __init__(self, name, caption, header):
        self.name, self.caption, self.header, self.rows = name, caption, header, []

    def write(self, out):
        out.mkdir(parents=True, exist_ok=True)
        md = [f"### {self.caption}", "", "| " + " | ".join(self.header) + " |",
              "|" + "|".join([":--"] + [":-:"] * (len(self.header) - 1)) + "|"]
        md += ["| " + " | ".join(row) + " |" for row in self.rows]
        (out / f"{self.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        def tex(cell):
            cell = cell.replace("±", r"$\pm$").replace("×", r"$\times$").replace("–", "--").replace("%", r"\%")
            cell = cell.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$").replace("κ", r"$\kappa$")
            if cell.startswith("**") and cell.endswith("**"):
                return r"\textbf{" + cell[2:-2] + "}"
            if cell.startswith("*") and cell.endswith("*"):
                return r"\textcolor{gray}{" + cell[1:-1] + "}"
            return cell

        lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex(self.caption)}}}",
                 rf"\label{{tab:{self.name}}}", r"\resizebox{\textwidth}{!}{%",
                 r"\begin{tabular}{l" + "c" * (len(self.header) - 1) + "}", r"\toprule",
                 " & ".join(tex(h) for h in self.header) + r" \\", r"\midrule"]
        lines += [" & ".join(tex(c) for c in row) + r" \\" for row in self.rows]
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        (out / f"{self.name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_stats(results, **match):
    """Per task: mean/std over seeds of the per-seed R_norm and C_norm."""
    rows = [r for r in results if all(r[k] == v if not isinstance(v, list) else r[k] in v for k, v in match.items())]
    out = {}
    for task in TASKS:
        per_seed = [r for r in rows if r["task"] == task]
        out[task] = (mean([r["R_norm"] for r in per_seed]), std([r["R_norm"] for r in per_seed]),
                     mean([r["C_norm"] for r in per_seed]), std([r["C_norm"] for r in per_seed]))
    return out


def reward_cost_table(name, caption, row_specs):
    header = ["Method"] + [f"{SHORT[t]} {m}" for t in TASKS for m in ("R↑", "C↓")] + ["Average R↑", "Average C↓"]
    table = Table(name, caption, header)

    stats = {}
    for label, values in row_specs:
        complete = all(not np.isnan(values[t][0]) for t in TASKS)
        values["Average"] = ((mean([values[t][0] for t in TASKS]), np.nan, mean([values[t][2] for t in TASKS]), np.nan)
                             if complete else (np.nan,) * 4)
        stats[label] = values

    best = {}
    for column in TASKS + ["Average"]:
        candidates = [(rank_key(v[column][0], v[column][2]), label) for label, v in stats.items()]
        candidates = [c for c in candidates if c[0][0] == 0]
        best[column] = min(candidates)[1] if candidates else None

    for label, values in stats.items():
        row = [label]
        for column in TASKS + ["Average"]:
            r, rs, c, cs = values[column]
            cells = [fmt(r, rs), fmt(c, cs)]
            if not np.isnan(c):
                style = "**" if best[column] == label else ("*" if c > 1.0 else "")
                cells = [f"{style}{x}{style}" for x in cells]
            row += cells
        table.rows.append(row)

    return table


def table1(results, reported):
    """The report's Table 1: rows = environments, columns = models (reward, cost) at kappa = 10.
    The CDT column comes from [CDT] Table 1; our own CDT runs (run_all.sh --with-cdt) get a column too."""
    k10 = [r for r in results if r["target_C"] == 10.0]
    columns = []
    if reported is not None and Path(reported).exists():
        with open(reported, encoding="utf-8") as file:
            paper = {r["task"]: (float(r["R_norm"]), np.nan, float(r["C_norm"]), np.nan) for r in csv.DictReader(file)}
        columns.append(("CDT (paper)", paper))
    for m in MODELS:
        stats = seed_stats(k10, exp="E1", model=m, variant="")
        if m != "cdt" or any(not np.isnan(v[0]) for v in stats.values()):
            columns.append((DISPLAY[m] if m != "cdt" else "CDT (ours)", stats))

    header = ["Environment"] + [f"{name} {metric}" for name, _ in columns for metric in ("Reward ↑", "Cost ↓")]
    table = Table("table1_main", "Normalized reward and cost returns at κ = 10 (mean ± std over 3 seeds, 20 episodes "
                  "each; gray = unsafe, cost > 1; bold = safe agent with the highest reward)", header)

    for task in TASKS + ["Average"]:
        values = {}
        for name, stats in columns:
            if task == "Average":
                complete = all(not np.isnan(stats.get(t, (np.nan,) * 4)[0]) for t in TASKS)
                values[name] = ((mean([stats[t][0] for t in TASKS]), np.nan, mean([stats[t][2] for t in TASKS]), np.nan)
                                if complete else (np.nan,) * 4)
            else:
                values[name] = stats.get(task, (np.nan,) * 4)

        safe = [(rank_key(v[0], v[2]), name) for name, v in values.items() if rank_key(v[0], v[2])[0] == 0]
        best = min(safe)[1] if safe else None

        row = [SHORT.get(task, task)]
        for name, _ in columns:
            r, rs, c, cs = values[name]
            cells = [fmt(r, rs), fmt(c, cs)]
            if not np.isnan(c):
                style = "**" if name == best else ("*" if c > 1.0 else "")
                cells = [f"{style}{x}{style}" for x in cells]
            row += cells
        table.rows.append(row)

    return table


def table2_summary(efficiency, training):
    """The report's Table 2: parameters, inference latency, peak GPU memory, training time (Car-Circle)."""
    table = Table("table2_summary", "Computational efficiency on Car-Circle (parameters; inference latency: one "
                  "prediction, batch 1, K = 10, GPU, median over 1000 calls; peak GPU memory and training time: "
                  "100k-step training runs, mean over seeds, training time without evaluation)",
                  ["Model", "Parameters", "Inference latency", "Peak GPU memory", "Training time"])
    rows = [r for r in efficiency if r["K"] == "10" and r["weights"] != "fresh_scaling" and r["device"] == "cuda"]

    for m in MODELS:
        params = eff(rows, m, "params_total", "value")[0]
        latency = eff(rows, m, "infer_latency_forward", "median", mode="window")[0]
        runs = [r for r in training if r["model"] == m and r["variant"] == "" and r["finished"]
                and r["task"] == "OfflineCarCircle-v0" and r["steps"] == 100_000]
        memory = mean([r["peak_gpu_memory_mb"] / 1024 for r in runs if r["peak_gpu_memory_mb"] is not None])
        hours = mean([r["training_seconds"] / 3600 for r in runs])
        table.rows.append([DISPLAY[m], MISSING if np.isnan(params) else f"{params / 1e6:.2f} M",
                           MISSING if np.isnan(latency) else f"{latency:.2f} ms",
                           MISSING if np.isnan(memory) else f"{memory:.2f} GB",
                           MISSING if np.isnan(hours) else f"{hours:.1f} hours"])

    return table


def table3(results):
    k10 = [r for r in results if r["target_C"] == 10.0]
    specs = []
    for m in MODELS:
        specs.append((f"{DISPLAY[m]} (full)", seed_stats(k10, exp="E1", model=m, variant="", seed=[0, 1, 2])))
        for variant, label in VARIANTS:
            specs.append((f"{DISPLAY[m]} {label}", seed_stats(k10, exp="E3", model=m, variant=variant)))
    return reward_cost_table("table3_ablations", "Component ablations for every backbone (κ = 10, 3 seeds)", specs)


def table4(results):
    header = ["Method"] + [f"{SHORT.get(t, t)} {m}" for t in TASKS + ["Average"] for m in ("R↑", "C↓", "Viol.↓", "Exc.↓")]
    table = Table("table4_multithreshold", "Average over κ ∈ {10, 20, 40}: normalized reward and cost, violation rate, "
                  "mean excess cost", header)
    metrics = ("R_norm", "C_norm", "violation", "excess")

    for m in MODELS:
        row, per_task = [DISPLAY[m]], []
        for task in TASKS:
            rows = [r for r in results if r["exp"] == "E1" and r["model"] == m and r["variant"] == ""
                    and r["task"] == task and r["target_C"] in (10.0, 20.0, 40.0)]
            seeds = group(rows, ["seed"])
            if seeds and all(len(v) == 3 for v in seeds.values()):
                values = [mean([mean([r[k] for r in v]) for v in seeds.values()]) for k in metrics]
                per_task.append(values)
                row += [fmt(v) for v in values]
            else:
                row += [MISSING] * 4
        row += ([fmt(mean([v[i] for v in per_task])) for i in range(4)] if len(per_task) == len(TASKS)
                else [MISSING] * 4)
        table.rows.append(row)

    return table


def table2(efficiency, training):
    gb = 1 / 1024
    columns = ["Params", "Size (MB)", "Ckpt (MB)", "Train s/step", "Train h 100k fp32 (proj.)",
               "Train h 100k bf16 (proj.)", "Train h (actual)", "Val. h (actual)", "Train mem (GB)",
               "Infer GPU ms", "Infer CPU-1 ms", "Infer CPU-8 ms", "Infer mem GPU (GB)", "Infer RAM CPU (GB)",
               "Recurrent ms", "State (KB)"]
    table = Table("table2_efficiency", "Efficiency at K = 10 (mean ± std over repetitions; training: batch 2048, "
                  "data sampling included, fp32 unless noted; inference: batch 1, fp32, median (p95))",
                  ["Model"] + columns)
    trained = [r for r in efficiency if r["K"] == "10" and r["weights"] != "fresh_scaling"]

    raw = {}
    for m in MODELS:
        g = lambda metric, stat, **kw: eff(trained, m, metric, stat, **kw)  # noqa: E731
        train = dict(device="cuda", mode="sampler_included", batch=2048)
        e6 = [r for r in training if r["exp"] == "E6full" and r["model"] == m and r["variant"] == "" and r["finished"]]
        actual = (mean([r["training_seconds"] / 3600 for r in e6]), np.nan)
        validation = (mean([r["evaluation_seconds"] / 3600 for r in e6]), np.nan)

        values = [g("params_total", "value"), g("static_model_mem", "value"),
                  tuple(x / 2**20 for x in g("checkpoint_bytes", "value")),
                  g("train_step_s", "median", precision="fp32", **train),
                  g("train_time_100k_projected", "value", precision="fp32", **train),
                  g("train_time_100k_projected", "value", precision="bf16", **train),
                  actual, validation,
                  tuple(x * gb for x in g("train_mem_allocated", "peak", device="cuda", precision="fp32", batch=2048)),
                  g("infer_latency_forward", "median", device="cuda", mode="window"),
                  g("infer_latency_forward", "median", device="cpu", mode="window", threads=1),
                  g("infer_latency_forward", "median", device="cpu", mode="window", threads=8),
                  tuple(x * gb for x in g("infer_mem_allocated", "peak", device="cuda", mode="window")),
                  tuple(x * gb for x in g("infer_mem_rss_increase", "peak", device="cpu", mode="window", threads=1)),
                  g("infer_latency_recurrent_step", "median", device="cuda"),
                  tuple(x / 1024 for x in g("recurrent_state_bytes", "value", device="cuda"))]
        p95 = {9: g("infer_latency_forward", "p95", device="cuda", mode="window")[0],
               10: g("infer_latency_forward", "p95", device="cpu", mode="window", threads=1)[0],
               11: g("infer_latency_forward", "p95", device="cpu", mode="window", threads=8)[0]}

        digits = [0, 2, 2, 4, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 3, 1]
        cells = []
        for i, ((v, s), d) in enumerate(zip(values, digits)):
            cell = fmt(v, s, d)
            if i in p95 and cell != MISSING:
                cell = f"{fmt(v, digits=d)} ({fmt(p95[i], digits=d)})"
            cells.append(cell)
        table.rows.append([DISPLAY[m]] + cells)
        raw[m] = [v for v, _ in values]

    table.rows.append(["*relative to CDT*"] + [""] * len(columns))
    for m in MODELS:
        ratios = []
        for i, v in enumerate(raw[m]):
            base = raw["cdt"][i]
            ratios.append(MISSING if np.isnan(v) or np.isnan(base) or base == 0 else f"{v / base:.2f}×")
        table.rows.append([f"{DISPLAY[m]} / CDT"] + ratios)

    return table


# ------------------------------------------------------------------ E0 report
def e0_report(training, validation, out):
    lines = ["# E0 report", "", "## Pilot (2k steps): did every run finish, and how long does a step take?", "",
             "s/step = time between the last two validations / steps between them (includes one small validation, "
             "so an upper bound).",
             "", "| Model | Task | Finished | s/step | 100k steps (h) |", "|:--|:--|:-:|:-:|:-:|"]
    per_step = {}
    for r in sorted([r for r in training if r["exp"] == "E0"], key=lambda r: (r["model"], r["task"])):
        elapsed = {v["step"]: v["elapsed_seconds"] for v in validation if v["run"] == r["run"]}
        steps = sorted(elapsed)
        s = ((elapsed[steps[-1]] - elapsed[steps[-2]]) / (steps[-1] - steps[-2]) if len(steps) > 1 else np.nan)
        per_step[(r["model"], r["task"])] = s
        lines.append(f"| {r['model']} | {SHORT.get(r['task'], r['task'])} | {'yes' if r['finished'] else '**NO**'} | "
                     f"{fmt(s, digits=4)} | {fmt(s * 1e5 / 3600)} |")

    worst = max([s for s in per_step.values() if not np.isnan(s)], default=np.nan)
    if not np.isnan(worst):
        hours = worst * 1e5 / 3600 + 0.5  # + validation of 100k-step runs and setup
        lines += ["", f"Slowest run: {worst:.4f} s/step -> about {hours:.1f} h per 100k-step run including validation.",
                  f"Suggested --time for E1/E3/E6full: {int(np.ceil(hours * 1.5))}:00:00 (1.5x margin)."]

    lines += ["", "## Learning-rate tuning (Car-Circle, 30k steps, validation seeds)", "",
              "Final validation, mean over kappa 10/20/40. Rule: among the safe settings (C_norm <= 1) the highest R_norm; "
              "if none is safe, the lowest C_norm.", "", "| Model | lr | R_norm | C_norm | chosen |", "|:--|:-:|:-:|:-:|:-:|"]
    tune = [r for r in training if r["exp"] == "E0tune"]
    for m in MODELS:
        options = []
        for r in [r for r in tune if r["model"] == m]:
            rows = [v for v in validation if v["run"] == r["run"]]
            if not rows:
                continue
            last = max(v["step"] for v in rows)
            final = [v for v in rows if v["step"] == last]
            options.append((r["variant"], mean([v["R_norm"] for v in final]), mean([v["C_norm"] for v in final]), last))
        chosen = min(options, key=lambda o: rank_key(o[1], o[2]))[0] if options else None
        for variant, r, c, last in options:
            lines.append(f"| {m} | {variant[2:]} (step {last}) | {fmt(r)} | {fmt(c)} | {'**yes**' if variant == chosen else ''} |")

    lines += ["", "Put the chosen lr in configs/models/<model>.yaml under trainer: (learning_rate: ...) before E1."]
    (out / "e0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ figures
def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def no_data(ax):
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=MUTED)


def legend(fig, models):
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=DISPLAY[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, -0.06))


def curve(rows, x, y):
    """Mean and std over seeds of per-seed values."""
    xs = sorted({r[x] for r in rows})
    mu = [mean([mean([r[y] for r in v]) for v in group([r for r in rows if r[x] == xv], ["seed"]).values()]) for xv in xs]
    sd = [std([mean([r[y] for r in v]) for v in group([r for r in rows if r[x] == xv], ["seed"]).values()]) for xv in xs]
    return np.array(xs), np.array(mu), np.nan_to_num(np.array(sd))


def fig_zero_shot(results, out):
    d = [r for r in results if r["exp"] == "E4"]
    fig, axes = plt.subplots(2, len(TASKS), figsize=(7.0, 3.2), sharex=True)
    for j, task in enumerate(TASKS):
        top, bottom = axes[0, j], axes[1, j]
        top.set_title(SHORT[task])
        rows = [r for r in d if r["task"] == task]
        if not rows:
            no_data(top), no_data(bottom)
            continue
        kappas = sorted({r["target_C"] for r in rows})
        top.plot(kappas, kappas, color=MUTED, lw=1, ls=":")
        for ax in (top, bottom):
            ax.axvline(rows[0]["data_c_max"], color=MUTED, lw=1, ls="--")
        for m in MODELS:
            rm = [r for r in rows if r["model"] == m and r["variant"] == ""]
            for ax, y in ((top, "C"), (bottom, "R_norm")):
                if rm:
                    x, mu, sd = curve(rm, "target_C", y)
                    ax.plot(x, mu, color=COLORS[m], marker=MARKERS[m], ms=3)
                    ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
        bottom.set_xlabel("target cost κ")
    axes[0, 0].set_ylabel("achieved cost")
    axes[1, 0].set_ylabel("normalized reward")
    legend(fig, MODELS)
    fig.tight_layout()
    save(fig, out, "fig1_zero_shot")


def fig_conflicting(results, out):
    d = [r for r in results if r["exp"] == "E5"]
    fig, axes = plt.subplots(2, len(TASKS), figsize=(7.0, 3.2))
    for j, task in enumerate(TASKS):
        top, bottom = axes[0, j], axes[1, j]
        top.set_title(SHORT[task])
        rows = [r for r in d if r["task"] == task]
        if not rows:
            no_data(top), no_data(bottom)
            continue
        bottom.axhline(10.0, color=MUTED, lw=1, ls="--")
        for m in MODELS:
            for variant, ls in (("", "-"), ("noaug", "--")):
                rm = [r for r in rows if r["model"] == m and r["variant"] == variant]
                for ax, y in ((top, "R"), (bottom, "C")):
                    if rm:
                        x, mu, _ = curve(rm, "target_R", y)
                        ax.plot(x, mu, color=COLORS[m], marker=MARKERS[m], ms=3, ls=ls)
        bottom.set_xlabel("target reward")
    axes[0, 0].set_ylabel("reward")
    axes[1, 0].set_ylabel("cost")
    legend(fig, MODELS)
    fig.text(0.99, 0.01, "solid: full model, dashed: w/o augmentation", ha="right", color=MUTED, fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig2_conflicting_targets")


def fig_scatter(results, out):
    d = [r for r in results if r["exp"] == "E1" and r["variant"] == "" and r["target_C"] == 10.0]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(7.0, 1.9))
    for ax, task in zip(axes, TASKS):
        ax.set_title(SHORT[task])
        ax.axvline(1.0, color=MUTED, lw=1, ls="--")
        rows = [r for r in d if r["task"] == task]
        if not rows:
            no_data(ax)
        for m in MODELS:
            rm = [r for r in rows if r["model"] == m]
            ax.scatter([r["C_norm"] for r in rm], [r["R_norm"] for r in rm], s=14, color=COLORS[m], marker=MARKERS[m],
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlabel("normalized cost")
    axes[0].set_ylabel("normalized reward")
    legend(fig, MODELS)
    fig.tight_layout()
    save(fig, out, "fig3_reward_cost_scatter")


def fig_learning_curves(validation, out):
    d = [v for v in validation if v["exp"] == "E1" and v["variant"] == "" and v["target_C"] == 10.0]
    fig, axes = plt.subplots(2, len(TASKS), figsize=(7.0, 3.2), sharex=True)
    for j, task in enumerate(TASKS):
        axes[0, j].set_title(SHORT[task])
        rows = [v for v in d if v["task"] == task]
        if not rows:
            no_data(axes[0, j]), no_data(axes[1, j])
        for m in MODELS:
            rm = [v for v in rows if v["model"] == m]
            for ax, y in ((axes[0, j], "R_norm"), (axes[1, j], "C_norm")):
                if rm:
                    x, mu, sd = curve(rm, "step", y)
                    ax.plot(x, mu, color=COLORS[m])
                    ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)
        axes[1, j].axhline(1.0, color=MUTED, lw=1, ls="--")
        axes[1, j].set_xlabel("gradient steps")
    axes[0, 0].set_ylabel("val. normalized reward")
    axes[1, 0].set_ylabel("val. normalized cost")
    legend(fig, MODELS)
    fig.tight_layout()
    save(fig, out, "fig4_learning_curves")


def fig_scaling(efficiency, out, batch=256):
    e = [r for r in efficiency if r["weights"] == "fresh_scaling"]
    panels = [("scaling_train_step_s", "median", "training time / step (s)", dict(mode="sampler_excluded", batch=batch)),
              ("train_mem_allocated", "peak", "training memory (GB)", dict(batch=batch)),
              ("infer_latency_forward", "median", "inference time / prediction (ms)", dict(mode="scaling_window")),
              ("infer_mem_allocated", "peak", "inference memory (GB)", dict(mode="scaling_window"))]
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.0))
    for ax, (metric, stat, label, match) in zip(axes.flat, panels):
        drawn = False
        for m in MODELS:
            for device, ls in (("cuda", "-"), ("cpu", "--")):
                name = "infer_mem_rss_increase" if device == "cpu" and metric == "infer_mem_allocated" else metric
                if device == "cpu" and "train" in metric:
                    continue
                rows = [r for r in e if r["model"] == m and r["metric"] == name and r["stat"] == stat
                        and r["device"] == device and all(str(r[k]) == str(v) for k, v in match.items())
                        and (device == "cuda" or r["threads"] == "1")]
                ks = sorted({int(r["K"]) for r in rows})
                values = [mean([r["value"] for r in rows if int(r["K"]) == k]) for k in ks]
                if "GB" in label:
                    values = [v / 1024 for v in values]
                points = [(k, v) for k, v in zip(ks, values) if v > 0]
                if points:
                    ax.plot(*zip(*points), color=COLORS[m], marker=MARKERS[m], ms=3, ls=ls)
                    drawn = True
                oom = sorted({int(r["K"]) for r in e if r["model"] == m and r["stat"] == "OOM"
                              and str(r["batch"]) == str(batch)})
                if oom and "train" in metric and points:
                    ax.scatter(oom, [points[-1][1]] * len(oom), color=COLORS[m], marker="x")
        ax.set_xscale("log")
        if drawn:
            ax.set_yscale("log")
        else:
            no_data(ax)
        ax.set_xlabel("context length K (timesteps)")
        ax.set_ylabel(label)
    legend(fig, MODELS)
    fig.text(0.99, 0.01, f"GPU solid, CPU (1 thread) dashed; training batch {batch}, fp32; x = out of memory; "
             "fresh weights (computational scaling only)", ha="right", color=MUTED, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig5_context_scaling")


def core_metrics(efficiency):
    trained = [r for r in efficiency if r["K"] == "10" and r["weights"] != "fresh_scaling"]
    return {m: {"training time": eff(trained, m, "train_step_s", "median", device="cuda", mode="sampler_included",
                                     precision="fp32", batch=2048)[0],
                "inference time": eff(trained, m, "infer_latency_forward", "median", device="cuda", mode="window")[0],
                "training memory": eff(trained, m, "train_mem_allocated", "peak", device="cuda", precision="fp32",
                                       batch=2048)[0],
                "inference memory": eff(trained, m, "infer_mem_allocated", "peak", device="cuda", mode="window")[0]}
            for m in MODELS}


def fig_tradeoff(results, efficiency, out):
    core = core_metrics(efficiency)
    d = [r for r in results if r["exp"] == "E1" and r["variant"] == "" and r["target_C"] == 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
    drawn = False
    for m in MODELS:
        per_task = [(mean([r["R_norm"] for r in v]), mean([r["C_norm"] for r in v]))
                    for v in group([r for r in d if r["model"] == m], ["task"]).values()]
        reward = mean([r for r, c in per_task if c <= 1.0])
        for ax, x in ((axes[0], core[m]["inference time"]), (axes[1], core[m]["training time"] * 1e5 / 3600)):
            if not (np.isnan(x) or np.isnan(reward)):
                ax.scatter([x], [reward], s=40, color=COLORS[m], marker=MARKERS[m], zorder=3)
                ax.annotate(DISPLAY[m], (x, reward), textcoords="offset points", xytext=(5, 3), fontsize=7)
                drawn = True
    axes[0].set_xlabel("inference time per prediction, GPU (ms)")
    axes[1].set_xlabel("training time for 100k steps (h, projected)")
    axes[0].set_ylabel("avg. normalized reward\n(tasks where safe)")
    if not drawn:
        no_data(axes[0]), no_data(axes[1])
    fig.tight_layout()
    save(fig, out, "fig6_efficiency_tradeoff")


def fig_efficiency_bars(efficiency, out):
    core = core_metrics(efficiency)
    names = list(core["cdt"])
    fig, ax = plt.subplots(figsize=(5.0, 2.2))
    width = 0.8 / len(MODELS)
    for i, m in enumerate(MODELS):
        ratios = [core[m][n] / core["cdt"][n] if core["cdt"][n] else np.nan for n in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * width, np.nan_to_num(ratios), width=width * 0.92,
               color=COLORS[m], label=DISPLAY[m])
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("relative to CDT (= 1)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    save(fig, out, "fig7b_efficiency_bars")


def fig_ablations(results, out):
    k10 = [r for r in results if r["target_C"] == 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    width = 0.8 / len(VARIANTS)
    for k, ((variant, label), hatch) in enumerate(zip(VARIANTS, ["", "////", "...."])):
        delta_r, delta_c = [], []
        for m in MODELS:
            full = seed_stats(k10, exp="E1", model=m, variant="", seed=[0, 1, 2])
            ablated = seed_stats(k10, exp="E3", model=m, variant=variant)
            common = [t for t in TASKS if not np.isnan(full[t][0]) and not np.isnan(ablated[t][0])]
            delta_r.append(mean([ablated[t][0] - full[t][0] for t in common]))
            delta_c.append(mean([ablated[t][2] - full[t][2] for t in common]))
        for ax, values in ((axes[0], delta_r), (axes[1], delta_c)):
            ax.bar(np.arange(len(MODELS)) + (k - 1) * width, np.nan_to_num(values), width=width * 0.92, color=MUTED,
                   hatch=hatch, edgecolor="white", label=label)
    for ax, label in ((axes[0], "Δ normalized reward"), (axes[1], "Δ normalized cost")):
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([DISPLAY[m] for m in MODELS], rotation=15)
        ax.set_ylabel(label)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, out, "fig7_ablation_deltas")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="runs/summary")
    parser.add_argument("--reported-cdt", default="configs/reported_cdt.csv",
                        help="CSV task,R_norm,C_norm: the CDT row of [CDT] Table 1")
    parser.add_argument("--all", action="store_true", help="also Tables 3-4, the E0 report and all figures")
    args = parser.parse_args()

    runs, out = ROOT / args.runs, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    training, results, episodes, validation = load(runs)
    efficiency = load_efficiency(runs / "profile" / "efficiency.csv")

    write_csv(out / "train_runs.csv", training)
    write_csv(out / "results.csv", results)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "val_logs.csv", validation)
    if (runs / "profile" / "efficiency.csv").exists():
        shutil.copy(runs / "profile" / "efficiency.csv", out / "efficiency.csv")

    table1(results, ROOT / args.reported_cdt).write(out / "tables")
    table2_summary(efficiency, training).write(out / "tables")
    for name in ("table1_main", "table2_summary"):
        print((out / "tables" / f"{name}.md").read_text(encoding="utf-8"))

    if args.all:
        write_all(training, results, validation, efficiency, out)

    finished = sum(r["finished"] for r in training)
    print(f"{finished}/{len(training)} training runs finished, {len(results)} evaluated (run, target) pairs, "
          f"{len(efficiency)} efficiency rows -> {out}")


def write_all(training, results, validation, efficiency, out):
    e0_report(training, validation, out)
    for table in (table2(efficiency, training), table3(results), table4(results)):
        table.write(out / "tables")

    figures = out / "figures"
    fig_zero_shot(results, figures)
    fig_conflicting(results, figures)
    fig_scatter(results, figures)
    fig_learning_curves(validation, figures)
    fig_scaling(efficiency, figures)
    fig_tradeoff(results, efficiency, figures)
    fig_efficiency_bars(efficiency, figures)
    fig_ablations(results, figures)


if __name__ == "__main__":
    main()
