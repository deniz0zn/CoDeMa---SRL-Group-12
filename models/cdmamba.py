import math

import torch
from torch import nn
from torch.nn import functional as F
from osrl.algorithms.cdt import CDT, CDTTrainer

try:
    import mamba_ssm  # imported first, as in cdm_3.py
    from mamba_ssm import Mamba3
except Exception as error:  # e.g. a CPU-only machine: only mamba_impl: ref works there
    Mamba3, MAMBA_IMPORT_ERROR = None, error


# Pure-PyTorch Mamba-3 (SISO) with the same parameters as mamba_ssm's Mamba3 (is_outproj_norm=False),
# so state_dicts are interchangeable. Used for CPU latency and for the recurrent step in efficiency.py.
#   S_t = alpha_t S_{t-1} + gamma_t x_t k_t^T + beta_t x_{t-1} k_{t-1}^T,  y_t = (S_t q_t + D x_t) * silu(z_t)

class RMSNormRef(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_parameter("bias", None)

    def forward(self, x):
        return _rmsnorm(x, self.weight, self.eps)


def _rmsnorm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class Mamba3Ref(nn.Module):
    def __init__(self, d_model, d_state=128, expand=2, headdim=64, rope_fraction=0.5, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, A_floor=1e-4, is_mimo=False, mimo_rank=4, chunk_size=64, **kwargs):
        super().__init__()
        if is_mimo:
            raise NotImplementedError("the reference path covers SISO only")

        self.d_model, self.d_state, self.headdim, self.A_floor = d_model, d_state, headdim, A_floor
        self.mimo_rank, self.num_bc_heads = 1, 1
        self.d_inner = int(expand * d_model)
        self.nheads = self.d_inner // headdim

        split = int(d_state * rope_fraction)
        self.num_rope_angles = (split - split % 2) // 2

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.num_rope_angles,
                                 bias=False)

        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state))
        self.B_norm = RMSNormRef(d_state)
        self.C_norm = RMSNormRef(d_state)

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        return mamba3_forward(self, u)


def _rotate(v, theta):
    """Rotary on pairs (2i, 2i+1) for i < theta.shape[-1]; the rest is not rotated."""
    s = theta.shape[-1]
    pairs = v[..., :2 * s].reshape(*v.shape[:-1], s, 2)
    v0, v1 = pairs[..., 0], pairs[..., 1]
    cos, sin = torch.cos(theta), torch.sin(theta)
    rotated = torch.stack([v0 * cos - v1 * sin, v0 * sin + v1 * cos], dim=-1).reshape(*v.shape[:-1], 2 * s)

    return torch.cat([rotated, v[..., 2 * s:]], dim=-1)


def _project(m, u):
    """Works on Mamba3Ref and on mamba_ssm's Mamba3 (same parameters)."""
    z, x, b, c, dd_dt, dd_a, trap, angles = torch.split(m.in_proj(u), [
        m.d_inner, m.d_inner, m.d_state * m.num_bc_heads * m.mimo_rank, m.d_state * m.num_bc_heads * m.mimo_rank,
        m.nheads, m.nheads, m.nheads, m.num_rope_angles], dim=-1)

    z = z.reshape(*z.shape[:-1], m.nheads, m.headdim).float()
    x = x.reshape(*x.shape[:-1], m.nheads, m.headdim).float()
    b = _rmsnorm(b, m.B_norm.weight, getattr(m.B_norm, "eps", 1e-5)).float()
    c = _rmsnorm(c, m.C_norm.weight, getattr(m.C_norm, "eps", 1e-5)).float()

    a = torch.clamp(-F.softplus(dd_a.float()), max=-m.A_floor)
    dt = F.softplus(dd_dt.float() + m.dt_bias.float())
    q = c.unsqueeze(-2) + m.C_bias[:, 0, :].float()
    k = b.unsqueeze(-2) + m.B_bias[:, 0, :].float()

    return z, x, q, k, a, dt, torch.sigmoid(trap.float()), torch.tanh(angles.float())


def mamba3_forward(m, u):
    """Parallel (exact) form of the recurrence. u [batch, length, d_model]."""
    z, x, q, k, a, dt, trap, angle = _project(m, u)
    theta = torch.cumsum(angle.unsqueeze(-2) * math.pi * dt.unsqueeze(-1), dim=1)
    qk = (q * k).sum(-1)
    q, k = _rotate(q, theta), _rotate(k, theta)

    gamma = dt * trap
    shifted = torch.zeros_like(gamma)
    shifted[:, :-1] = dt[:, 1:] * (1.0 - trap[:, 1:])
    scale = gamma + shifted
    cumulative = torch.cumsum(a * dt, dim=1).transpose(1, 2)

    length = u.shape[1]
    scores = torch.einsum("blhn,bshn->bhls", q, k)
    strict = torch.ones(length, length, dtype=torch.bool, device=u.device).tril(-1)
    decay = torch.exp((cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~strict, float("-inf")))

    y = torch.einsum("bhls,bshp->blhp", scores * decay * scale.transpose(1, 2).unsqueeze(-2), x)
    y = (y + (m.D.float().view(1, 1, -1, 1) + (gamma * qk).unsqueeze(-1)) * x) * F.silu(z)

    return m.out_proj(y.reshape(*y.shape[:2], m.nheads * m.headdim).to(u.dtype))


def mamba3_init_state(m, batch_size, device):
    return {"angle": torch.zeros(batch_size, m.nheads, m.num_rope_angles, device=device),
            "ssm": torch.zeros(batch_size, m.nheads, m.headdim, m.d_state, device=device),
            "k": torch.zeros(batch_size, m.nheads, m.d_state, device=device),
            "v": torch.zeros(batch_size, m.nheads, m.headdim, device=device)}


def mamba3_step(m, u, state):
    """One recurrent step. u [batch, d_model]."""
    z, x, q, k, a, dt, trap, angle = _project(m, u)
    theta = state["angle"] + angle.unsqueeze(-2) * math.pi * dt.unsqueeze(-1)
    q, k = _rotate(q, theta), _rotate(k, theta)

    alpha = torch.exp(a * dt)
    beta = (1.0 - trap) * dt * alpha
    gamma = trap * dt

    ssm = (state["ssm"] * alpha[..., None, None] + gamma[..., None, None] * x.unsqueeze(-1) * k.unsqueeze(-2)
           + beta[..., None, None] * state["v"].unsqueeze(-1) * state["k"].unsqueeze(-2))
    y = (torch.einsum("bhpn,bhn->bhp", ssm, q) + m.D.float().view(1, -1, 1) * x) * F.silu(z)

    return m.out_proj(y.reshape(y.shape[0], -1).to(u.dtype)), {"angle": theta, "ssm": ssm, "k": k, "v": x}


class MambaBlock(nn.Module):
    """x = x + Dropout(Mamba3(LayerNorm(x)));  x = x + MLP(LayerNorm(x))"""

    def __init__(self, dim, dropout, use_mlp, impl, **mamba_config):
        super().__init__()

        self.impl = impl
        self.chunk_size = mamba_config["chunk_size"]
        self.norm1 = nn.LayerNorm(dim)

        if impl == "cuda":
            if Mamba3 is None:
                raise ImportError(f"cannot import mamba_ssm.Mamba3 ({MAMBA_IMPORT_ERROR}); "
                                  "use mamba_impl: ref on CPU")
            self.mixer = Mamba3(d_model=dim, is_outproj_norm=False, **mamba_config)
        elif impl == "ref":
            self.mixer = Mamba3Ref(d_model=dim, **mamba_config)
        else:
            raise ValueError(f"mamba_impl must be cuda or ref, got {impl}")

        self.dropout = nn.Dropout(dropout)
        self.use_mlp = use_mlp

        if use_mlp:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, padding_mask=None):
        # causal and padding is on the right only, so the padding mask is not needed
        h = self.norm1(x)

        if self.impl == "cuda":
            # right-pad to a multiple of chunk_size and slice back (exact, the mixer is causal)
            length = h.shape[1]
            h = F.pad(h, (0, 0, 0, (-length) % self.chunk_size))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mixed = self.mixer(h)
            mixed = mixed[:, :length].to(x.dtype)
        else:
            mixed = self.mixer(h)

        x = x + self.dropout(mixed)

        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))

        return x

    def step(self, x, state):
        y, state = mamba3_step(self.mixer, self.norm1(x), state)
        x = x + self.dropout(y)

        if self.use_mlp:
            x = x + self.mlp(self.norm2(x))

        return x, state


class CDMamba(CDT):
    def __init__(self, embedding_dim=128, num_layers=3, d_state=64, expand=2, headdim=32, rope_fraction=0.5,
                 is_mimo=False, mimo_rank=4, chunk_size=64, mamba_mlp=True, mamba_impl="cuda",
                 residual_dropout=0.1, **kwargs):

        super().__init__(embedding_dim=embedding_dim, num_layers=0, residual_dropout=residual_dropout, **kwargs)

        self.blocks = nn.ModuleList([MambaBlock(dim=embedding_dim, dropout=residual_dropout, use_mlp=mamba_mlp,
                                                impl=mamba_impl, d_state=d_state, expand=expand, headdim=headdim,
                                                rope_fraction=rope_fraction, is_mimo=is_mimo, mimo_rank=mimo_rank,
                                                chunk_size=chunk_size) for _ in range(num_layers)])

        # CDT's init for LayerNorm/MLP only; the Mamba-3 mixer keeps its own init (dt_bias, B/C biases, D)
        for block in self.blocks:
            for name, child in block.named_children():
                if name != "mixer":
                    child.apply(self._init_weights)


    # recurrent mode, only used by efficiency.py
    def init_recurrent(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return [mamba3_init_state(block.mixer, batch_size, device) for block in self.blocks]

    def recurrent_step(self, x, state):
        states = []
        for block, s in zip(self.blocks, state):
            x, s = block.step(x, s)
            states.append(s)

        return x, states


def build(env, model_config, trainer_config, device):
    if model_config.get("mamba_impl", "cuda") == "cuda" and torch.device(device).type != "cuda":
        raise ValueError("CDMamba with mamba_impl: cuda requires device: cuda (mamba_impl: ref runs on CPU).")

    action_dim = env.action_space.shape[0]

    model = CDMamba(state_dim=env.observation_space.shape[0], action_dim=action_dim,
                    max_action=float(env.action_space.high[0]), episode_len=env.spec.max_episode_steps,
                    target_entropy=-action_dim, **model_config).to(device)

    if model.stochastic:
        model.log_temperature = (model.log_temperature.detach().to(device).requires_grad_(True))

    trainer = CDTTrainer(model=model, env=env, device=device, **trainer_config)

    # no weight decay on the Mamba-3 parameters marked _no_weight_decay (dt_bias, D), as in cdm_3.py
    defaults = trainer.optim.defaults
    lr_rule = trainer.scheduler.lr_lambdas[0]
    decay = [p for p in model.parameters() if p.requires_grad and not getattr(p, "_no_weight_decay", False)]
    no_decay = [p for p in model.parameters() if p.requires_grad and getattr(p, "_no_weight_decay", False)]

    trainer.optim = torch.optim.AdamW([{"params": decay, "weight_decay": defaults["weight_decay"]},
                                       {"params": no_decay, "weight_decay": 0.0}],
                                      lr=defaults["lr"], betas=defaults["betas"], eps=defaults["eps"],
                                      weight_decay=defaults["weight_decay"])
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optim, lr_rule)

    return model, trainer
