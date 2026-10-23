import argparse
import hashlib
import importlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dsrl.offline_env import wrap_env

from grad_accum import evaluate


def assess(checkpoint_path, output, device="cpu", episodes=10, seed=60000, mode="sample"):
    checkpoint_path, output = Path(checkpoint_path), Path(output)

    if episodes < 1:
        raise ValueError("Episodes must be positive")
    if output.exists():
        raise ValueError("Output file already exists. Choose another filename")

    torch.set_num_threads(2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved = checkpoint["config"]
    if isinstance(saved.get("model"), dict):
        config = saved
    else:
        expected_hash = (
            "6d76baea4de50dedc72433ee962787306ac5"
            "bc02cd93d85b4f6cd345f8c78dcb"
        )
        if (
                saved.get("architecture") != "full CDT: 3x128, 8 heads"
                or saved.get("parameters") != 625805
                or saved.get("dataset_sha256") != expected_hash
        ):
            raise ValueError("Unsupported old checkpoint configuration")

        config = {
            "legacy_training_config": saved,
            "model": {
                "model_script": "models.cdt",
                "model": {
                    "embedding_dim": 128,
                    "num_layers": 3,
                    "num_heads": 8,
                    "attention_dropout": 0.1,
                    "residual_dropout": 0.1,
                    "embedding_dropout": 0.1,
                    "time_emb": True,
                    "use_rew": True,
                    "use_cost": True,
                    "cost_transform": True,
                    "add_cost_feat": False,
                    "mul_cost_feat": False,
                    "cat_cost_feat": False,
                    "action_head_layers": 1,
                    "cost_prefix": False,
                    "stochastic": True,
                },
                "trainer": {
                    "loss_cost_weight": 0.02,
                    "loss_state_weight": 0.0,
                    "cost_reverse": False,
                    "no_entropy": False,
                },
            },
            "dataset": {
                "task": "OfflineCarRun-v0",
                "scales": {"reward_scale": 0.1, "cost_scale": 1.0},
                "sequence": {"seq_len": 10},
            },
            "training": {
                "optimizer": {
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "betas": [0.9, 0.999],
                    "lr_warmup_steps": 500,
                    "clip_grad": 0.25,
                },
                "evaluation": {
                    "target_return": 575.0,
                    "target_costs": [10.0, 20.0, 40.0],
                },
            },
        }

    m, d, t = config["model"], config["dataset"], config["training"]
    settings = {**t["evaluation"], **d.get("evaluation", {}), "episodes": episodes, "seed": seed, "action_mode": mode}
    env = wrap_env(gym.make(d["task"]), reward_scale=d["scales"]["reward_scale"])

    try:
        module = importlib.import_module(m["model_script"])
        model, trainer = module.build(
            env=env,
            model_config={**m["model"], "seq_len": d["sequence"]["seq_len"]},
            trainer_config={**t["optimizer"], **m["trainer"], **d["scales"]},
            device=device
        )

        model.load_state_dict(checkpoint["model"], strict=True)
        if model.stochastic:
            with torch.no_grad():
                model.log_temperature.copy_(checkpoint["log_temperature"])

        summary, rollouts = evaluate(trainer, settings, d["scales"], d["task"])


        if episodes > 1:
            rng = np.random.default_rng(42)

            for row in summary:
                group = [e for e in rollouts if e["budget"] == row["target_cost"]]
                indices = rng.integers(0, len(group), size=(5000, len(group)))

                for field in ["reward", "cost"]:
                    values = np.array([e[field] for e in group])
                    row[field + "_mean_ci95"] = np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()

        result = dict(checkpoint=str(checkpoint_path.resolve()),
                      checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                      step=checkpoint["step"], config=config, evaluation=settings, summary=summary, episodes=rollouts,
                      uncertainty=("Pointwise bootstrap intervals across evaluation seeds; "
                                   "not training-seed uncertainty or a safety certificate."))

        with output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        print(json.dumps(summary, indent=2))
    finally:
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--mode", choices=["mean", "sample"], default="sample")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        assess(args.checkpoint, args.output, args.device, args.episodes, args.seed, args.mode)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()