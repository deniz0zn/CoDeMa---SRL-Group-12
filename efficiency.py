"""E6 efficiency measurements: model size, training time/throughput/memory, inference latency/memory,
recurrent (step-by-step) inference, context memory and context scaling (K = 10 ... 200).

    python efficiency.py --device cuda --rep 0 --what size train infer
    CUDA_VISIBLE_DEVICES="" taskset -c 0 python efficiency.py --device cpu --threads 1 --rep 0 --what infer
    python efficiency.py --device cuda --rep 0 --what scaling

The models and their settings are the entries of configs/experiments/e6full.yaml. Trained weights come from the
finished E6full runs (fresh weights when a run is missing; the "weights" column says which).
Rows are appended to runs/profile/efficiency.csv (memory in MB, time in s or ms as the "unit" column says).
"""

import argparse
import contextlib
import csv
import importlib
import io
import itertools
import json
import platform
import socket
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dsrl.offline_env import wrap_env
from osrl.common.dataset import SequenceDataset
from torch.utils.data import DataLoader

from grad_accum import train_step
from hpc import merge, read_yaml
from hpc_eval import find_run


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs" / "profile"
FIELDS = ["model", "task", "metric", "stat", "value", "unit", "device", "mode", "K", "batch", "precision",
          "repetition", "threads", "impl", "weights", "host"]
SCALING_K = [10, 20, 50, 100, 200]


class Logger:
    """Stands in for RunLogger: train_step calls trainer.logger.store()."""

    def store(self, tab="train", **values):
        pass


class Rows:
    def __init__(self, **common):
        self.common = common
        OUTPUT.mkdir(parents=True, exist_ok=True)

    def add(self, **row):
        row = {**self.common, **row}
        path = OUTPUT / "efficiency.csv"
        exists = path.exists()

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in FIELDS})

        print({key: row.get(key) for key in ("model", "metric", "stat", "value", "unit", "K", "batch", "precision",
                                             "device", "mode")}, flush=True)

    def stats(self, seconds, scale, unit, **row):
        values = np.asarray(seconds) * scale
        for stat, value in (("median", np.median(values)), ("p95", np.percentile(values, 95)),
                            ("mean", values.mean()), ("std", values.std())):
            self.add(stat=stat, value=float(value), unit=unit, **row)


# ------------------------------------------------------------------ models
def resolve(config, entry):
    """Model, dataset and training config of an experiment entry, merged exactly as hpc.py does."""
    model = merge(read_yaml(entry["model"]), entry.get("model_overrides", {}))
    training = read_yaml(config["training"])
    merge(training, config.get("training_overrides", {}))
    merge(training, entry.get("training_overrides", {}))
    dataset = read_yaml(entry.get("dataset", config["dataset"]))
    merge(dataset, config.get("dataset_overrides", {}))
    merge(dataset, entry.get("dataset_overrides", {}))

    return model, dataset, training


def build(model_cfg, dataset_cfg, training_cfg, device, seq_len):
    env = wrap_env(gym.make(dataset_cfg["task"]), reward_scale=dataset_cfg["scales"]["reward_scale"])
    model_config = {**model_cfg["model"], "seq_len": seq_len}

    if "mamba_impl" in model_config and torch.device(device).type == "cpu":
        model_config["mamba_impl"] = "ref"  # the Mamba-3 kernels need CUDA; same parameters, pure PyTorch

    module = importlib.import_module(model_cfg["model_script"])
    torch.manual_seed(0)
    model, trainer = module.build(env=env, model_config=model_config,
                                  trainer_config={**training_cfg["optimizer"], **model_cfg["trainer"],
                                                  **dataset_cfg["scales"], "logger": Logger()},
                                  device=device)
    env.close()

    return model, trainer, model_config.get("mamba_impl", "-")


def load_weights(model, weights):
    if weights is not None:
        model.load_state_dict(weights["model"], strict=True)
        if model.stochastic:
            with torch.no_grad():
                model.log_temperature.copy_(weights["log_temperature"])


def static_bytes(model):
    return sum(t.numel() * t.element_size() for t in itertools.chain(model.parameters(), model.buffers()))


def state_dict_bytes(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def random_batch(model, batch, K, device="cpu"):
    """A batch in the format of SequenceDataset: s, a, r, c, t, mask, episode cost, costs."""
    g = torch.Generator().manual_seed(0)
    start = torch.randint(0, model.episode_len, (batch, 1), generator=g)
    tensors = (torch.randn(batch, K, model.state_dim, generator=g), torch.rand(batch, K, model.action_dim, generator=g) * 2 - 1,
               torch.randn(batch, K, generator=g) * 50, torch.rand(batch, K, generator=g) * 40,
               start + torch.arange(K), torch.ones(batch, K), torch.rand(batch, generator=g) * 40,
               (torch.rand(batch, K, generator=g) < 0.1).float())

    return tuple(x.to(device) for x in tensors)


# ------------------------------------------------------------------ timing
def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def time_calls(function, warmup, timed, device):
    for _ in range(warmup):
        function()
    sync(device)

    seconds = np.empty(timed)
    for i in range(timed):
        start = time.perf_counter()
        function()
        sync(device)
        seconds[i] = time.perf_counter() - start

    return seconds


def rss_bytes():
    import psutil
    return psutil.Process().memory_info().rss


# ------------------------------------------------------------------ measurements
def measure_train(rows, make, device, K, batch, microbatch, precision, batches, warmup, timed, label="train_step"):
    """Training time per step (sampler excluded / included), throughput and peak memory of one full step."""
    model, trainer = make()
    model.train()
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16"
                else contextlib.nullcontext())

    if batches is None:
        cycle = itertools.cycle([random_batch(model, batch, K)])
    else:
        cycle = itertools.cycle([next(batches) for _ in range(4)])

    def step(data):
        with autocast:
            train_step(trainer, data, microbatch)

    common = dict(K=K, batch=batch, precision=precision)
    step(next(cycle))
    sync(device)
    torch.cuda.reset_peak_memory_stats()
    step(next(cycle))
    sync(device)
    rows.add(metric="train_mem_allocated", stat="peak", unit="MB", value=torch.cuda.max_memory_allocated() / 2**20, **common)
    rows.add(metric="train_mem_reserved", stat="peak", unit="MB", value=torch.cuda.max_memory_reserved() / 2**20, **common)
    rows.add(metric="train_mem_static", stat="value", unit="MB", value=static_bytes(model) / 2**20, **common)

    tokens = model.seq_repeat * K
    modes = [("sampler_excluded", lambda: step(next(cycle)))]
    if batches is not None:
        modes.append(("sampler_included", lambda: step(next(batches))))

    for mode, function in modes:
        seconds = time_calls(function, warmup, timed, device)
        per_step = float(np.median(seconds))
        rows.add(metric=f"{label}_s", stat="median", unit="s/step", value=per_step, mode=mode, **common)
        rows.add(metric=f"{label}_s", stat="mean", unit="s/step", value=float(seconds.mean()), mode=mode, **common)
        rows.add(metric="steps_per_s", stat="value", unit="1/s", value=1.0 / per_step, mode=mode, **common)
        rows.add(metric="tokens_per_s", stat="value", unit="tokens/s", value=batch * tokens / per_step, mode=mode, **common)
        rows.add(metric="train_time_100k_projected", stat="value", unit="h", value=per_step * 1e5 / 3600, mode=mode, **common)

    del model, trainer
    torch.cuda.empty_cache()


@torch.inference_mode()
def measure_infer(rows, model, device, K, warmup, timed, label="window"):
    """Latency of one prediction (batch 1, a full window of K timesteps), inference memory, context memory."""
    model.eval()
    window = random_batch(model, 1, K, device)[:5]
    common = dict(K=K, batch=1, precision="fp32")

    rows.stats(time_calls(lambda: model(*window, None, None), warmup, timed, device), 1e3, "ms",
               metric="infer_latency_forward", mode=label, **common)

    if torch.device(device).type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats()
        model(*window, None, None)
        sync(device)
        rows.add(metric="infer_mem_allocated", stat="peak", unit="MB", value=torch.cuda.max_memory_allocated() / 2**20,
                 mode=label, **common)
    else:
        base = peak = rss_bytes()
        for _ in range(200):
            model(*window, None, None)
            peak = max(peak, rss_bytes())
        rows.add(metric="infer_mem_rss_increase", stat="peak", unit="MB", value=(peak - base) / 2**20, mode=label, **common)

    rows.add(metric="context_window_bytes", stat="value", unit="bytes", mode=label,
             value=float(sum(x.numel() * x.element_size() for x in window)), **common)


def _feed(model, embedding, state):
    y, state = model.recurrent_step(model.emb_drop(model.emb_norm(embedding)), state)
    return model.out_norm(y), state


def recurrent_observe(model, s, r, c, t, state):
    """Feed the R, C and s tokens of one timestep; return the action prediction."""
    te = model.timestep_emb(t) if model.time_emb else 0.0
    if model.use_rew:
        _, state = _feed(model, model.return_emb(r.unsqueeze(-1)) + te, state)

    costs_emb = None
    if model.use_cost:
        c = model.cost_transform(c) if model.cost_transform is not None else c
        costs_emb = model.cost_emb(c.unsqueeze(-1)) + te
        _, state = _feed(model, costs_emb, state)

    feature, state = _feed(model, model.state_emb(s) + te, state)
    if model.add_cost_feat and model.use_cost:
        feature = feature + costs_emb
    if model.mul_cost_feat and model.use_cost:
        feature = feature * costs_emb
    if model.cat_cost_feat and model.use_cost:
        feature = torch.cat([feature, costs_emb], dim=-1)

    return model.action_head(feature), state


def recurrent_act(model, a, t, state):
    """Feed the executed action token."""
    te = model.timestep_emb(t) if model.time_emb else 0.0
    return _feed(model, model.action_emb(a) + te, state)[1]


def state_bytes(state):
    if torch.is_tensor(state):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(v) for v in state)
    return 0


@torch.inference_mode()
def measure_recurrent(rows, model, device, K, warmup, timed):
    """Step-by-step inference (cdmamba, cdconv): latency of one env step (R, C, s -> action, then a) and state size."""
    model.eval()
    holder = {"state": model.init_recurrent(1, device)}
    s = torch.randn(1, model.state_dim, device=device)
    r, c = torch.zeros(1, device=device), torch.zeros(1, device=device)
    a = torch.zeros(1, model.action_dim, device=device)
    t = torch.zeros(1, dtype=torch.long, device=device)

    def step():
        _, holder["state"] = recurrent_observe(model, s, r, c, t, holder["state"])
        holder["state"] = recurrent_act(model, a, t, holder["state"])

    common = dict(K=K, batch=1, precision="fp32", mode="recurrent")
    rows.stats(time_calls(step, warmup, timed, device), 1e3, "ms", metric="infer_latency_recurrent_step", **common)
    rows.add(metric="recurrent_state_bytes", stat="value", unit="bytes", value=float(state_bytes(holder["state"])), **common)

    if torch.device(device).type == "cuda":
        sync(device)
        torch.cuda.reset_peak_memory_stats()
        step()
        sync(device)
        rows.add(metric="infer_mem_allocated", stat="peak", unit="MB", value=torch.cuda.max_memory_allocated() / 2**20, **common)


# ------------------------------------------------------------------ main
def hardware(device):
    info = dict(host=socket.gethostname(), device=device, threads=torch.get_num_threads(), torch=torch.__version__,
                python=platform.python_version(), cpu=platform.processor())
    try:
        info["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)

    path = OUTPUT / "hardware.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[f"{info['host']}:{device}:threads{info['threads']}"] = info
    path.write_text(json.dumps(data, indent=2))

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/e6full.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=None, help="torch threads (CPU measurements)")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--what", nargs="+", default=["size", "train", "infer"], choices=["size", "train", "infer", "scaling"])
    parser.add_argument("--models", nargs="*", help="entry names (default: all)")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "bf16"])
    parser.add_argument("--train-warmup", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--infer-warmup", type=int, default=50)
    parser.add_argument("--infer-steps", type=int, default=1000)
    parser.add_argument("--scaling-batches", nargs="+", type=int, default=[256, 2048])
    parser.add_argument("--scaling-steps", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    cuda = torch.device(device).type == "cuda"

    info = hardware(device)
    config = read_yaml(args.config)
    data = {}  # task -> DataLoader iterator, shared by the models

    for entry in config["runs"]:
        if args.models and entry["name"] not in args.models:
            continue

        model_cfg, dataset_cfg, training_cfg = resolve(config, entry)
        task, K = dataset_cfg["task"], dataset_cfg["sequence"]["seq_len"]

        try:
            run_dir = find_run(training_cfg["experiment"], task, entry["name"], training_cfg["seed"])
            weights = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            checkpoint_bytes, weights_label = (run_dir / "last.pt").stat().st_size, "trained"
        except FileNotFoundError:
            weights, checkpoint_bytes, weights_label = None, None, "fresh"

        def make(seq_len=K, trained=True):
            model, trainer, _ = build(model_cfg, dataset_cfg, training_cfg, device, seq_len)
            load_weights(model, weights if trained else None)
            return model, trainer

        model, _, impl = build(model_cfg, dataset_cfg, training_cfg, device, K)
        load_weights(model, weights)
        rows = Rows(model=entry["name"], task=task, device=device, repetition=args.rep, threads=torch.get_num_threads(),
                    impl=impl, weights=weights_label, host=info["host"])

        if "size" in args.what:
            rows.add(metric="params_total", stat="value", unit="count", K=K,
                     value=float(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            rows.add(metric="static_model_mem", stat="value", unit="MB", K=K, value=static_bytes(model) / 2**20)
            rows.add(metric="state_dict_bytes", stat="value", unit="bytes", K=K, value=float(state_dict_bytes(model)))
            if checkpoint_bytes is not None:
                rows.add(metric="checkpoint_bytes", stat="value", unit="bytes", K=K, value=float(checkpoint_bytes))

        if "train" in args.what and cuda:
            if task not in data:
                env = gym.make(task)
                dataset = SequenceDataset(env.get_dataset(), **dataset_cfg["sequence"], **dataset_cfg["scales"],
                                          cost_transform=lambda cost: dataset_cfg["sampling_cost_offset"] - cost)
                env.close()
                data[task] = iter(DataLoader(dataset, batch_size=training_cfg["batch_size"],
                                             num_workers=training_cfg["num_workers"]))

            for precision in args.precisions:
                measure_train(rows, make, device, K, training_cfg["batch_size"], training_cfg["microbatch_size"],
                              precision, data[task], args.train_warmup, args.train_steps)

        if "infer" in args.what:
            measure_infer(rows, model, device, K, args.infer_warmup, args.infer_steps)
            if hasattr(model, "init_recurrent"):
                measure_recurrent(rows, model, device, K, args.infer_warmup, args.infer_steps)

        if "scaling" in args.what:
            # computational scaling with fresh weights and random inputs, not a performance claim
            for k in SCALING_K:
                scaling = Rows(**{**rows.common, "weights": "fresh_scaling"})
                if cuda:
                    for batch in args.scaling_batches:
                        try:
                            measure_train(scaling, lambda: make(k, trained=False), device, k, batch, batch, "fp32", None, 5,
                                          args.scaling_steps, label="scaling_train_step")
                        except torch.cuda.OutOfMemoryError:
                            scaling.add(metric="scaling_train_step_s", stat="OOM", value="", unit="", K=k, batch=batch,
                                        precision="fp32", mode="sampler_excluded")
                            torch.cuda.empty_cache()

                fresh, _ = make(k, trained=False)
                measure_infer(scaling, fresh, device, k, 10, args.scaling_steps, label="scaling_window")
                if hasattr(fresh, "init_recurrent"):
                    measure_recurrent(scaling, fresh, device, k, 10, args.scaling_steps)
                del fresh

        del model
        if cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
