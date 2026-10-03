import argparse
import importlib
import random

import gymnasium as gym
import numpy as np
import torch
import yaml
from dsrl.offline_env import wrap_env
from torch.utils.data import DataLoader
from tqdm import trange

from osrl.common.dataset import SequenceDataset


def load_yaml(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/cdt.yaml")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--training", default="configs/train.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model)
    data_cfg = load_yaml(args.dataset)
    train_cfg = load_yaml(args.training)

    # might need to add cuda-related seeding before hpc
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])

    device = train_cfg["device"]
    scales = data_cfg["scales"]
    env = gym.make(data_cfg["task"])

    try:
        data = env.get_dataset()

        dataset = SequenceDataset(data, **data_cfg["sequence"], **scales)
        batches = iter(DataLoader(dataset, batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"]))

        # match the evaluation rewards to the scaled training returns
        env = wrap_env(env, reward_scale=scales["reward_scale"])

        implementation = importlib.import_module(model_cfg["model_script"])

        model, trainer = implementation.build(
            env=env,
            model_config={**model_cfg["model"], "seq_len": data_cfg["sequence"]["seq_len"]},
            trainer_config={**train_cfg["optimizer"], **model_cfg["trainer"], **scales},
            device=device,
        )

        model.train()
        for _ in trange(train_cfg["steps"], desc="Training"):
            batch = [tensor.to(device) for tensor in next(batches)]
            trainer.train_one_step(*batch)

        evaluation = train_cfg["evaluation"]
        reward, cost, length = trainer.evaluate(
            num_rollouts=evaluation["episodes"],
            target_return=(evaluation["target_return"] * scales["reward_scale"]),
            target_cost=(evaluation["target_cost"] * scales["cost_scale"]),
        )

        print(f"Mean reward: {reward:.2f} | Mean cost: {cost:.2f} | Mean length: {length:.1f}")

    finally:
        env.close()

if __name__ == "__main__":
    main()