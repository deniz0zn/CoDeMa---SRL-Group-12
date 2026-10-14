import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import wandb
import yaml


ROOT = Path(__file__).resolve().parent


def load_config(model_path, dataset_path, training_path):
    def read(path):
        with (ROOT / path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    model = read(model_path); dataset = read(dataset_path); training = read(training_path)

    return {
        "model_name": Path(model_path).stem,
        "task": dataset["task"],
        "seed": training["seed"],
        "experiment": training.get("experiment", "default"),
        "model": model,
        "dataset": dataset,
        "training": training
    }


class RunLogger:
    def __init__(self, config):
        self.config = config
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.best_score = None
        self.best = None
        self.last = None
        self.start = perf_counter()
        self.evaluation_seconds = 0.0  # time spent in evaluate(), added by train.py

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        name = f"{config['model_name']}_seed{config['seed']}_{timestamp}"

        self.directory = ROOT / "runs" / name
        self.directory.mkdir(parents=True)

        (self.directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        settings = config["training"]["wandb"]
        self.run = wandb.init(project=settings["project"], entity=settings.get("entity"), mode=settings["mode"],
                              name=name,
                              group=(f"{config['experiment']}/{config['task']}/{config['model_name']}"),
                              job_type="train", tags=[config["experiment"], config["task"], config["model_name"]],
                              config=config, dir=str(self.directory))


        self.run.define_metric("train/step")
        self.run.define_metric("*", step_metric="train/step")


    def __enter__(self): # to use with the "with RunLogger"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                wall = perf_counter() - self.start
                self.final = {
                    "selection_budgets": self.config["training"]["evaluation"]["target_costs"],
                    "selection_rule": "joint_mean_cost_then_average_reward",
                    "best": self.best, "last": self.last,
                    "time": {"wall_seconds": wall, "evaluation_seconds": self.evaluation_seconds,
                             "training_seconds": wall - self.evaluation_seconds},
                    "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                           if torch.cuda.is_available() else None),
                }

                (self.directory / "final.json").write_text(json.dumps(self.final, indent=2), encoding="utf-8")

        finally:
            self.run.finish(exit_code=0 if exc_type is None else 1)


    def _append_csv(self, filename, rows):
        path = self.directory / filename
        exists = path.exists()

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))

            if not exists: writer.writeheader()
            writer.writerows(rows)


    def model_summary(self, model):
        self.run.summary["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def store(self, tab="train", **values):
        """Called by CDTTrainer during each training update."""
        for name, value in values.items():
            key = f"{tab}/{name}"
            self.totals[key] += float(value)
            self.counts[key] += 1

    def log_training(self, step):
        metrics = {"train/step": step, **{key: total / self.counts[key] for key, total in self.totals.items()}}

        self.totals.clear(); self.counts.clear()

        self._append_csv("training.csv", [metrics])
        self.run.log(metrics)


    def _checkpoint(self, filename, model, trainer, step):
        checkpoint = {
            "step": step, "config": self.config,
            "model": model.state_dict(),
            "optimizer": trainer.optim.state_dict(), "scheduler": trainer.scheduler.state_dict(),
        }


        if model.stochastic:
            checkpoint["log_temperature"] = model.log_temperature.detach().cpu()
            checkpoint["temperature_optimizer"] = trainer.log_temperature_optimizer.state_dict()

        path = self.directory / filename
        temporary = path.with_suffix(".tmp")

        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def log_evaluation(self, step, results, model, trainer):
        budgets = self.config["training"]["evaluation"]["target_costs"]
        worst_ratio = max(r["mean_cost"] / r["target_cost"] for r in results)
        average_reward = sum(r["mean_reward"] / len(results) for r in results)



        joint_feasible = worst_ratio <= 1.0
        score = (joint_feasible, average_reward if joint_feasible else -worst_ratio, average_reward)

        elapsed = perf_counter() - self.start
        self.last = {"step": step, "elapsed_seconds": elapsed, "evaluation": results,
                     "joint_feasible": joint_feasible, "worst_cost_ratio": worst_ratio}

        self._append_csv("evaluation.csv", [{"step": step, "elapsed_seconds": elapsed, **r} for r in results])
        self._checkpoint("last.pt", model, trainer, step)
        self._checkpoint(f"step_{step:06d}.pt", model, trainer, step)

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best = self.last
            self._checkpoint("best.pt", model, trainer, step)
            self.run.summary.update({"best_step": step,
                                     "best_joint_feasible": joint_feasible,
                                     "best_average_reward": average_reward,
                                     "best_worst_cost_ratio": worst_ratio,
                                     "selection_budgets": budgets
                                     })

        metrics = {"train/step": step}
        for result in results:
            prefix = f"eval/cost_{result['target_cost']:g}"

            for key, value in result.items():
                if key != "target_cost":
                    metrics[f"{prefix}/{key}"] = value


        self.run.log(metrics)
        print(f"\nStep {step}: {results}")