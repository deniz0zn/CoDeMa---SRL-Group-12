import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["cdt", "cdt_light", "cdm", "cdc"]


def load_runs(experiment):
    rows, times = [], []

    for directory in sorted((ROOT / "runs" / experiment).iterdir()):
        if not (directory / "final.json").exists():
            print(f"Skipping unfinished run: {directory.name}", file=sys.stderr)
            continue

        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        run = dict(model=config["model_name"], task=config["task"].removeprefix("Offline").removesuffix("-v0"),
                   seed=config["seed"])

        final = json.loads((directory / "final.json").read_text(encoding="utf-8"))
        times.append({**run, "train_hours": final["last"]["train_seconds"] / 3600,
                      "wall_hours": final["last"]["elapsed_seconds"] / 3600})

        for checkpoint in ("best", "last"):
            path = directory / f"heldout_{checkpoint}.json"
            if not path.exists():
                print(f"Missing {path.name}: {directory.name}", file=sys.stderr)
                continue

            held_out = json.loads(path.read_text(encoding="utf-8"))
            for result in held_out["summary"]:
                rows.append({**run, "checkpoint": checkpoint, "step": held_out["step"],
                             **{key: result[key] for key in ("target_cost", "target_return", "mean_reward", "mean_cost",
                                                             "normalized_reward", "normalized_cost", "violation_rate")}})
    return rows, times


def ordered(names):
    return sorted(names, key=lambda name: (ORDER.index(name) if name in ORDER else len(ORDER), name))


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), (values.std(ddof=1) if len(values) > 1 else 0.0)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_table(rows, checkpoint):
    per_seed = defaultdict(list)
    for row in rows:
        if row["checkpoint"] == checkpoint:
            per_seed[(row["model"], row["task"], row["seed"])].append(row)

    cells = defaultdict(lambda: ([], []))
    for (model, task, _), group in per_seed.items():
        cells[(model, task)][0].append(100 * np.mean([r["normalized_reward"] for r in group]))
        cells[(model, task)][1].append(np.mean([r["normalized_cost"] for r in group]))

    models = ordered({model for model, _ in cells})
    tasks = sorted({task for _, task in cells})

    lines = [f"### Held-out evaluation, {checkpoint}.pt (normalized reward x100, higher is better; normalized cost, lower is better; mean (sd) over seeds)", "",
             "| Task | " + " | ".join(f"{m} reward | {m} cost" for m in models) + " |",
             "|---|" + "---|---|" * len(models)]
    for task in tasks + ["Average"]:
        cells_text = []
        for model in models:
            if task == "Average":
                rewards = [cells[(model, t)][0] for t in tasks if (model, t) in cells]
                costs = [cells[(model, t)][1] for t in tasks if (model, t) in cells]
                seeds = min(map(len, rewards)) if rewards else 0
                reward = mean_sd(np.mean([r[:seeds] for r in rewards], axis=0)) if seeds else None
                cost = mean_sd(np.mean([c[:seeds] for c in costs], axis=0)) if seeds else None
            else:
                reward = mean_sd(cells[(model, task)][0]) if (model, task) in cells else None
                cost = mean_sd(cells[(model, task)][1]) if (model, task) in cells else None
            cells_text += [f"{reward[0]:.1f} ± {reward[1]:.1f}" if reward else "–",
                           f"{cost[0]:.2f} ± {cost[1]:.2f}" if cost else "–"]
        lines.append(f"| {task} | " + " | ".join(cells_text) + " |")
    return "\n".join(lines)


def budget_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["model"], row["task"], row["target_cost"])].append(row)

    table = []
    for (checkpoint, model, task, budget), group in sorted(groups.items()):
        reward, reward_sd = mean_sd([r["mean_reward"] for r in group])
        cost, cost_sd = mean_sd([r["mean_cost"] for r in group])
        table.append(dict(checkpoint=checkpoint, model=model, task=task, target_cost=budget,
                          target_return=group[0]["target_return"], seeds=len(group),
                          reward_mean=reward, reward_sd=reward_sd, cost_mean=cost, cost_sd=cost_sd,
                          normalized_reward_mean=100 * np.mean([r["normalized_reward"] for r in group]),
                          normalized_cost_mean=np.mean([r["normalized_cost"] for r in group]),
                          violation_rate_mean=np.mean([r["violation_rate"] for r in group])))
    return table


def budget_figure(table, checkpoint, path):
    rows = [r for r in table if r["checkpoint"] == checkpoint]
    tasks = sorted({r["task"] for r in rows})
    figure, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3), squeeze=False)

    for axis, task in zip(axes[0], tasks):
        budgets = sorted({r["target_cost"] for r in rows if r["task"] == task})
        axis.plot(budgets, budgets, "k--", linewidth=1, label="budget")
        for model in ordered({r["model"] for r in rows if r["task"] == task}):
            points = sorted((r["target_cost"], r["cost_mean"], r["cost_sd"]) for r in rows
                            if r["task"] == task and r["model"] == model)
            x, y, sd = map(np.array, zip(*points))
            axis.errorbar(x, y, yerr=sd, marker="o", capsize=3, label=model)
        axis.set_title(task)
        axis.set_xlabel("target cost")
    axes[0][0].set_ylabel("episode cost")
    axes[0][-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="main")
    parser.add_argument("--output", type=Path, default=None, help="default: experiments/results/<experiment>")
    args = parser.parse_args()

    output = ROOT / (args.output or Path("experiments/results") / args.experiment)
    output.mkdir(parents=True, exist_ok=True)

    rows, times = load_runs(args.experiment)
    if not rows:
        sys.exit("No held-out results found")

    write_csv(output / "heldout_per_budget_per_seed.csv", rows)
    write_csv(output / "training_time.csv", times)

    table = budget_table(rows)
    write_csv(output / "budget_adaptation.csv", table)

    report = [main_table(rows, checkpoint) for checkpoint in ("last", "best")]
    (output / "tables.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print("\n\n".join(report))

    for checkpoint in ("last", "best"):
        budget_figure(table, checkpoint, output / f"budget_adaptation_{checkpoint}.png")
    print(f"\nSaved to {output}")