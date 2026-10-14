"""Evaluation-only experiments: evaluate finished training runs again, one entry per array index (like hpc.py).

    python hpc_eval.py --config configs/experiments/e1_test.yaml --index 0

Each entry names a finished run of a training experiment (source, task, name, seed). The newest matching run
under runs/ with a final.json is loaded from its last.pt (the final model) and evaluated with grad_accum.evaluate.
The result goes to runs/<run>/eval_<experiment>.json; an existing result is not computed again.
"""

import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
import yaml
from dsrl.offline_env import wrap_env

from grad_accum import evaluate
from hpc import merge, read_yaml


ROOT = Path(__file__).resolve().parent


def find_run(experiment, task, name, seed):
    """Newest finished run (has final.json) of a training experiment."""
    matches = []
    for path in (ROOT / "runs").glob("*/config.yaml"):
        if not (path.parent / "final.json").exists():
            continue

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (config.get("experiment"), config.get("task"), config.get("model_name"), config.get("seed")) == \
                (experiment, task, name, seed):
            matches.append(path.parent)

    if not matches:
        raise FileNotFoundError(f"No finished run for experiment={experiment} task={task} name={name} seed={seed}")

    return max(matches, key=lambda directory: directory.name.rsplit("_", 1)[-1])  # newest timestamp


def load_model(run_dir, device, model_overrides=None):
    """Rebuild the model of a run from its saved config and load last.pt (as assess_checkpoint.py does)."""
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    m, d, t = config["model"], config["dataset"], config["training"]

    env = wrap_env(gym.make(d["task"]), reward_scale=d["scales"]["reward_scale"])
    module = importlib.import_module(m["model_script"])
    model, trainer = module.build(env=env,
                                  model_config={**m["model"], **(model_overrides or {}), "seq_len": d["sequence"]["seq_len"]},
                                  trainer_config={**t["optimizer"], **m["trainer"], **d["scales"]},
                                  device=device)

    model.load_state_dict(checkpoint["model"], strict=True)
    if model.stochastic:
        with torch.no_grad():
            model.log_temperature.copy_(checkpoint["log_temperature"])

    return model, trainer, env, checkpoint, config


def dataset_returns(env):
    """Reward and cost return of every trajectory of the task's offline dataset."""
    data = env.get_dataset()
    ends = np.where(np.logical_or(data["terminals"], data["timeouts"]))[0]
    starts = np.r_[0, ends[:-1] + 1]

    rewards = np.r_[0.0, np.cumsum(data["rewards"], dtype=np.float64)]
    costs = np.r_[0.0, np.cumsum(data["costs"], dtype=np.float64)]

    return rewards[ends + 1] - rewards[starts], costs[ends + 1] - costs[starts]


def run(config_path, index, device):
    config = read_yaml(config_path)
    entry = config["runs"][index]
    experiment = config["experiment"]
    source = entry.get("source", config["source"])

    print(f"Evaluation {experiment}: {index} | source: {source} | {entry['name']} | {entry['task']} | seed {entry['seed']}")

    run_dir = find_run(source, entry["task"], entry["name"], entry["seed"])
    output = run_dir / f"eval_{experiment}.json"

    if output.exists():
        print(f"Already evaluated: {output}")
        return

    start = perf_counter()
    model, trainer, env, checkpoint, run_config = load_model(run_dir, device)

    try:
        reward_returns, cost_returns = dataset_returns(env.unwrapped)
        dataset = dict(r_max=float(reward_returns.max()), r_min=float(reward_returns.min()),
                       c_max=float(cost_returns.max()), c_min=float(cost_returns.min()),
                       trajectories=len(reward_returns))

        if "reward_sweep" in entry:
            # E5: reward targets as multiples of the dataset's best reward return, at one cost budget
            sweep = entry["reward_sweep"]
            targets = [[float(r), float(sweep["cost"])]
                       for r in np.linspace(sweep["low"] * dataset["r_max"], sweep["high"] * dataset["r_max"], sweep["num"])]
        else:
            targets = entry["targets"]

        settings = merge(dict(config["evaluation"]), entry.get("evaluation", {}))
        settings["targets"] = targets

        summary, episodes = evaluate(trainer, settings, run_config["dataset"]["scales"], entry["task"])
    finally:
        env.close()

    spec = gym.spec(entry["task"]).kwargs
    result = dict(experiment=experiment, source=source, run_dir=run_dir.name, checkpoint=str(run_dir / "last.pt"),
                  step=checkpoint["step"], task=entry["task"], name=entry["name"], seed=entry["seed"],
                  evaluation=settings,
                  normalization=dict(r_min=float(spec["min_episode_reward"]), r_max=float(spec["max_episode_reward"])),
                  dataset=dataset, wall_seconds=perf_counter() - start, summary=summary, episodes=episodes)

    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)

    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run(args.config, args.index, args.device)
