import argparse
import itertools
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assess_checkpoint import assess
from hpc import merge, read_yaml
from train import train

def cells(config):
    return list(itertools.product(config["models"], config["tasks"], config["seeds"]))


def run(config_path, index):
    config = read_yaml(config_path)
    grid = cells(config)
    entry, task_path, seed = grid[index]

    model = merge(read_yaml(entry["config"]), entry.get("model_overrides", {}))
    training = merge(read_yaml(config["training"]), config.get("training_overrides", {}))
    training["seed"] = seed
    task = read_yaml(task_path)

    print(f"Cell {index}/{len(grid) - 1} | model: {entry['name']} | task: {task['task']} | seed: {seed}")
    print(f"Steps: {training['steps']}")

    if torch.device(training["device"]).type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with TemporaryDirectory(prefix="codema-") as directory:
        paths = {"model": Path(directory) / f"{entry['name']}.yaml",
                 "dataset": Path(directory) / "dataset.yaml",
                 "training": Path(directory) / "training.yaml"}

        for key, values in [("model", model), ("dataset", task), ("training", training)]:
            paths[key].write_text(yaml.safe_dump(values), encoding="utf-8")

        result = train(model_path=str(paths["model"]), dataset_path=str(paths["dataset"]),
                       training_path=str(paths["training"]))

    run_dir = Path(result["run_dir"])
    final = config.get("final_evaluation")

    if final:
        for checkpoint in ("best", "last"):
            assess(run_dir / f"{checkpoint}.pt", run_dir / f"heldout_{checkpoint}.json",
                   device=training["device"], **final)

    print(f"Results: {run_dir}")
    return result


def download(config_path):
    for task_path in read_yaml(config_path)["tasks"]:
        name = read_yaml(task_path)["task"]
        env = gym.make(name)
        print(name, len(env.get_dataset()["observations"]), "transitions")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/main.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--count", action="store_true", help="print the number of grid cells and exit")
    parser.add_argument("--list", action="store_true", help="print index -> (model, task, seed) and exit")
    parser.add_argument("--download", action="store_true", help="download all task datasets and exit")
    args = parser.parse_args()

    if args.count:
        print(len(cells(read_yaml(args.config))))
    elif args.list:
        for i, (entry, task_path, seed) in enumerate(cells(read_yaml(args.config))):
            print(f"{i:3d}  {entry['name']:10s} {Path(task_path).stem:13s} seed {seed}")
    elif args.download:
        download(args.config)
    else:
        if args.index is None:
            parser.error("--index is required")
        run(args.config, args.index)