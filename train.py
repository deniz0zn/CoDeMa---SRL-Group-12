import argparse
import importlib
import random

import gymnasium as gym
import numpy as np
import torch
from dsrl.offline_env import wrap_env
from osrl.common.dataset import SequenceDataset
from torch.utils.data import DataLoader
from tqdm import trange

from utils import RunLogger, load_config


def evaluate(trainer, config, scales):
    python_state, numpy_state = random.getstate(), np.random.get_state()

    results = []
    try:
        with torch.random.fork_rng():
            for budget in config["target_costs"]:
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                torch.manual_seed(config["seed"])

                reward, cost, length = trainer.evaluate(num_rollouts=config["episodes"],
                                                        target_return=(config["target_return"] * scales["reward_scale"]),
                                                        target_cost=budget * scales["cost_scale"])
                results.append({
                    "target_cost": float(budget),"mean_reward": float(reward), "mean_cost": float(cost), "mean_length": float(length)})

    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)

    return results


def train(model_path="configs/models/cdt.yaml", dataset_path="configs/dataset.yaml", training_path="configs/train.yaml"):
    config = load_config(model_path, dataset_path, training_path)
    model_cfg, data_cfg, train_cfg = config["model"], config["dataset"], config["training"]

    random.seed(train_cfg["seed"]); np.random.seed(train_cfg["seed"]); torch.manual_seed(train_cfg["seed"])

    device, scales, evaluation = train_cfg["device"], data_cfg["scales"], train_cfg["evaluation"]


    with RunLogger(config) as logger:
        env = gym.make(data_cfg["task"])

        try:
            dataset = SequenceDataset(env.get_dataset(), **data_cfg["sequence"], **scales)
            batches = iter(DataLoader(dataset, batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"]))

            env = wrap_env(env, reward_scale=scales["reward_scale"])

            model_script = importlib.import_module(model_cfg["model_script"])
            model, trainer = model_script.build(
                env=env,
                model_config={**model_cfg["model"], "seq_len": data_cfg["sequence"]["seq_len"]},
                trainer_config={**train_cfg["optimizer"], **model_cfg["trainer"], **scales, "logger": logger},
                device=device
            )

            logger.model_summary(model)
            model.train()

            for step in trange(1, train_cfg["steps"] + 1, desc="Training"):
                batch = [tensor.to(device) for tensor in next(batches)]
                trainer.train_one_step(*batch)

                evaluate_now = step % evaluation["every"] == 0 or step == train_cfg["steps"]

                if step % train_cfg["log_every"] == 0 or evaluate_now:
                    logger.log_training(step)

                if evaluate_now:
                    results = evaluate(trainer, evaluation, scales)
                    logger.log_evaluation(step, results, model, trainer)

        finally:
            env.close()

    return {"run_dir": str(logger.directory), "final": logger.final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/models/cdm.yaml")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--training", default="configs/train.yaml")
    args = parser.parse_args()

    train(args.model, args.dataset, args.training)