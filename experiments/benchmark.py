import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from dsrl.offline_env import wrap_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grad_accum import train_step
from hpc import read_yaml


class NullLogger:
    def store(self, **_):
        pass


def synthetic_batch(model, batch_size, seq_len, device):
    return (torch.randn(batch_size, seq_len, model.state_dim, device=device),
            (torch.rand(batch_size, seq_len, model.action_dim, device=device) * 2 - 1) * 0.9 * model.max_action,
            torch.rand(batch_size, seq_len, device=device) * 60,
            torch.rand(batch_size, seq_len, device=device) * 40,
            torch.arange(seq_len, device=device).repeat(batch_size, 1),
            torch.ones(batch_size, seq_len, device=device),
            torch.rand(batch_size, device=device) * 40,
            torch.randint(0, 2, (batch_size, seq_len), device=device).float())


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed(function, device, repeats, warmup):
    for _ in range(warmup):
        function()

    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    synchronize(device)

    return (time.perf_counter() - start) / repeats


def peak_memory(function, device):
    if torch.device(device).type != "cuda":
        return float("nan")

    synchronize(device)
    torch.cuda.reset_peak_memory_stats()
    function()
    synchronize(device)

    return torch.cuda.max_memory_allocated() / 2 ** 20


def benchmark(model_path, task, training, seq_len, args):
    config = read_yaml(model_path)
    env = wrap_env(gym.make(task["task"]), reward_scale=task["scales"]["reward_scale"])

    try:
        model, trainer = importlib.import_module(config["model_script"]).build(
            env=env, model_config={**config["model"], "seq_len": seq_len},
            trainer_config={**training["optimizer"], **config["trainer"], **task["scales"], "logger": NullLogger()},
            device=args.device)
    finally:
        env.close()

    batch = synthetic_batch(model, args.batch_size, seq_len, args.device)
    single = [x[:1] for x in synthetic_batch(model, 1, seq_len, args.device)]

    def train():
        model.train()
        train_step(trainer, batch, args.microbatch_size)

    @torch.inference_mode()
    def act():
        model.eval()
        s, a, r, c, t, _, ec, _ = single
        model(s, a, r, c, t, None, ec)

    return dict(model=Path(model_path).stem, task=task["task"], seq_len=seq_len, tokens=model.seq_repeat * seq_len,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                train_step_ms=1000 * timed(train, args.device, args.train_repeats, args.warmup),
                train_peak_mb=peak_memory(train, args.device),
                action_latency_ms=1000 * timed(act, args.device, args.inference_repeats, args.warmup),
                inference_peak_mb=peak_memory(act, args.device),
                batch_size=args.batch_size, microbatch_size=args.microbatch_size,
                device=torch.cuda.get_device_name(0) if torch.device(args.device).type == "cuda" else "cpu")


if __name__ == "__main__":
    training = read_yaml("configs/train.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["configs/models/cdt.yaml", "configs/models/cdt_light.yaml",
                                                        "configs/models/cdm.yaml", "configs/models/cdc.yaml"])
    parser.add_argument("--tasks", nargs="+", default=["experiments/tasks/car_run.yaml"])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--batch-size", type=int, default=training["batch_size"])
    parser.add_argument("--microbatch-size", type=int, default=training["microbatch_size"])
    parser.add_argument("--train-repeats", type=int, default=10)
    parser.add_argument("--inference-repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("experiments/results/benchmark.csv"))
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = []

    for task_path in args.tasks:
        task = read_yaml(task_path)
        for seq_len in args.seq_lens:
            for model_path in args.models:
                row = benchmark(model_path, task, training, seq_len, args)
                rows.append(row)
                print(row)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")