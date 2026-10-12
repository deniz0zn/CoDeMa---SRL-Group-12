import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from dsrl.offline_env import wrap_env


def train_step(trainer, batch, microbatch):
    model = trainer.model

    device = next(model.parameters()).device
    total = len(batch[0])

    valid_total = batch[5].sum().item()
    if valid_total <= 0:
        raise ValueError("Batch has no valid tokens")

    metrics = dict.fromkeys(["nll", "ent", "act_loss", "cost_loss", "state_loss", "cost_acc"], 0.0)

    temperature = model.temperature().detach()
    if trainer.no_entropy:
        temperature = temperature * 0

    trainer.optim.zero_grad(set_to_none=True)

    for start in range(0, total, microbatch):
        s, a, r, c, t, mask, ec, costs = [x[start:start + microbatch].to(device) for x in batch]

        valid = mask.bool()
        if not valid.any():
            raise ValueError("Microbatch has no valid tokens")

        policy, cp, sp = model(s, a, r, c, t, ~valid, ec)
        nll = -policy.log_prob(a)[valid].mean()
        ent = policy.entropy()[valid].mean()
        act = nll - temperature * ent

        cost = (F.nll_loss(cp.reshape(-1, 2), costs.flatten().long(), reduction="none") * mask.flatten()).mean()

        state = ((sp[:, :-1] - s[:, 1:]).square() * mask[:, :-1, None]).mean()

        acc = ((cp.argmax(-1) == costs.long()) * valid).sum() / valid.sum()


        vw = mask.sum().item() / valid_total
        ew = len(s) / total
        loss = vw * act + ew * (trainer.cost_weight * cost + trainer.state_weight * state)

        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")

        loss.backward()

        for key, value, weight in [("nll", nll, vw), ("ent", ent, vw), ("act_loss", act, vw),
                                   ("cost_loss", cost, ew), ("state_loss", state, ew), ("cost_acc", acc, vw)]:
            metrics[key] += value.detach().item() * weight

    limit = (trainer.clip_grad if trainer.clip_grad is not None else float("inf"))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)

    trainer.optim.step()
    trainer.log_temperature_optimizer.zero_grad(set_to_none=True)
    temperature_loss = (model.temperature() * (metrics["ent"] - model.target_entropy))
    temperature_loss.backward()
    trainer.log_temperature_optimizer.step()
    trainer.scheduler.step()

    metrics.update(ent_reg=temperature.item(), grad_norm=float(norm), clipped=float(norm > limit),
                   all_loss=(metrics["act_loss"] + trainer.cost_weight * metrics["cost_loss"] + trainer.state_weight * metrics["state_loss"]),
                   train_lr=trainer.scheduler.get_last_lr()[0])

    trainer.logger.store(tab="train", **metrics)

    return metrics


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def evaluate(trainer, config, scales, task):
    mode = config.get("action_mode", "sample")

    model = trainer.model
    device = next(model.parameters()).device
    py_state = random.getstate()
    np_state = np.random.get_state()
    was_training = model.training
    episodes = []
    devices = list(range(torch.cuda.device_count()))

    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()

            for budget in config["target_costs"]:
                seeds = range(config["seed"], config["seed"] + config["episodes"])
                for seed in seeds:
                    _seed(seed)
                    env = wrap_env(gym.make(task), reward_scale=scales["reward_scale"])

                    try:
                        _seed(seed)
                        obs, _ = env.reset(seed=seed)

                        horizon = model.episode_len
                        s = torch.zeros(1, horizon + 1, model.state_dim, device=device)
                        a = torch.zeros(1, horizon, model.action_dim, device=device)
                        r = torch.zeros(1, horizon + 1, device=device)
                        c = torch.zeros_like(r)
                        t = torch.arange(horizon, device=device).reshape(1, -1)

                        s[0, 0] = torch.as_tensor(obs, device=device)
                        r[0, 0] = (config["target_return"] * scales["reward_scale"])
                        c[0, 0] = budget * scales["cost_scale"]
                        ec = torch.tensor([budget * scales["cost_scale"]], dtype=torch.float32, device=device)

                        reward_sum = cost_sum = saturation = 0.0

                        for step in range(horizon):
                            lo = max(0, step + 1 - model.seq_len)
                            hi = step + 1
                            policy, _, _ = model(s[:, lo:hi], a[:, lo:hi], r[:, lo:hi], c[:, lo:hi], t[:, lo:hi], None, ec,)
                            action = (policy.mean if mode == "mean" else policy.sample())[0, -1]

                            saturation += (action.abs() > model.max_action).float().mean().item()
                            action = action.clamp(-model.max_action, model.max_action)

                            obs, reward, terminated, truncated, info = (env.step(action.cpu().numpy()))
                            raw_cost = float(info["cost"])
                            cost = (1.0 - raw_cost if trainer.cost_reverse else raw_cost)

                            a[0, step] = action
                            s[0, step + 1] = torch.as_tensor(obs, device=device)
                            r[0, step + 1] = r[0, step] - reward
                            c[0, step + 1] = (c[0, step] - cost * scales["cost_scale"])

                            reward_sum += reward / scales["reward_scale"]
                            cost_sum += raw_cost
                            if terminated or truncated:
                                break

                        episodes.append(dict(seed=seed,
                                             budget=float(budget),
                                             action_mode=mode,
                                             reward=float(reward_sum),
                                             cost=cost_sum,
                                             length=step + 1,
                                             terminated=bool(terminated),
                                             truncated=bool(truncated),
                                             action_saturation=saturation / (step + 1)))

                    finally:
                        env.close()

    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        model.train(was_training)

    results = []
    for budget in config["target_costs"]:
        group = [e for e in episodes if e["budget"] == budget]
        rewards = np.array([e["reward"] for e in group])
        costs = np.array([e["cost"] for e in group])

        results.append(dict(target_cost=float(budget),
                            mean_reward=float(rewards.mean()),
                            mean_cost=float(costs.mean()),
                            mean_length=float(np.mean([e["length"] for e in group])),
                            reward_sd=(float(rewards.std(ddof=1)) if len(group) > 1 else 0.0),
                            cost_sd=(float(costs.std(ddof=1)) if len(group) > 1 else 0.0),
                            violation_rate=float((costs > budget).mean()),
                            max_cost=float(costs.max()),
                            episodes=len(group)))

    return results, episodes