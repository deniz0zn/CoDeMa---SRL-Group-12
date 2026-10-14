"""E10: cost-reward plots of the offline datasets with their Pareto frontier (no training).

    python plot_datasets.py [--out runs/summary]

Writes figures/fig8_datasets.{pdf,png} and datasets.csv (trajectories, reward/cost return ranges per task).
"""

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dsrl.offline_env import wrap_env  # noqa: E402,F401  (registers the DSRL tasks)

from analyze import MUTED, SHORT, TASKS, no_data, save  # noqa: E402
from hpc_eval import dataset_returns  # noqa: E402


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/summary")
    args = parser.parse_args()
    out = ROOT / args.out

    fig, axes = plt.subplots(1, len(TASKS), figsize=(7.0, 1.9))
    stats = []

    for ax, task in zip(axes, TASKS):
        ax.set_title(SHORT[task])
        try:
            env = gym.make(task)
            rewards, costs = dataset_returns(env.unwrapped)
            env.close()
        except Exception as error:  # e.g. the dataset is not downloaded
            print(f"{task}: {error}")
            no_data(ax)
            continue

        order = np.argsort(costs)
        ax.scatter(costs, rewards, s=3, color="#86b6ef", lw=0)
        ax.plot(costs[order], np.maximum.accumulate(rewards[order]), color="#2a78d6", lw=1.5)  # Pareto frontier
        for kappa in (10, 20, 40):
            ax.axvline(kappa, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("cost return")

        stats.append(dict(task=task, trajectories=len(rewards), r_min=float(rewards.min()), r_max=float(rewards.max()),
                          c_min=float(costs.min()), c_max=float(costs.max())))

    axes[0].set_ylabel("reward return")
    fig.tight_layout()
    save(fig, out / "figures", "fig8_datasets")

    if stats:
        with (out / "datasets.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(stats[0]))
            writer.writeheader()
            writer.writerows(stats)

    print(f"Results: {out}")


if __name__ == "__main__":
    main()
