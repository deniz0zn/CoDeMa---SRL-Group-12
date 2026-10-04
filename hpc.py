import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from train import train


ROOT = Path(__file__).resolve().parent


def read_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def run(config_path, index):
    config = read_yaml(config_path)
    entry = config["runs"][index]

    model = read_yaml(entry["model"])
    training = read_yaml(config["training"])

    merge(model, entry.get("model_overrides", {}))
    merge(training, config.get("training_overrides", {}))
    merge(training, entry.get("training_overrides", {}))

    print(f"Experiment: {index} | model: {entry['name']}")
    print(f"Steps: {training['steps']} | seed: {training['seed']}")

    if torch.device(training["device"]).type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


    with TemporaryDirectory(prefix="codema-") as directory:
        model_path = Path(directory) / f"{entry['name']}.yaml"
        training_path = Path(directory) / "training.yaml"

        model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
        training_path.write_text(yaml.safe_dump(training), encoding="utf-8")

        result = train(model_path=str(model_path), dataset_path=config["dataset"], training_path=str(training_path))

    print(f"Results: {result['run_dir']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    run(args.config, args.index)