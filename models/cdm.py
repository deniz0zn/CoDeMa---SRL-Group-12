import math

import torch
from torch import nn
from torch.nn import functional as F

from osrl.algorithms.cdt import CDT, CDTTrainer


class MambaMixer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.inner = int(expand * dim)
        self.d_state = d_state
        self.rank = math.ceil(dim / 16)

        self.in_proj = nn.Linear(dim, 2 * self.inner, bias=False)

        self.conv = nn.Conv1d(self.inner, self.inner, kernel_size=d_conv, groups=self.inner, padding=d_conv - 1)

        self.select = nn.Linear(self.inner, self.rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner)

        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, dim, bias=False)


        # Mamba-1 initialization
        nn.init.uniform_(self.dt_proj.weight, -self.rank ** -0.5, self.rank ** -0.5)

        dt = torch.exp(torch.rand(self.inner)* (math.log(0.1) - math.log(0.001))+ math.log(0.001))

        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, tokens):
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        length = x.shape[1]

        # depthwise convolution
        x = self.conv(x.transpose(1, 2))
        x = x[:, :, :length].transpose(1, 2)
        x = F.silu(x)

        delta, b, c = self.select(x).split([self.rank, self.d_state, self.d_state],dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        a = -self.A_log.exp()

        state = x.new_zeros(x.shape[0], self.inner, self.d_state)
        outputs = []

        # selective state-space recurrence
        for t in range(length):
            step = delta[:, t, :, None]

            state = (torch.exp(step * a) * state + step * b[:, t, None, :] * x[:, t, :, None])

            y = (state * c[:, t, None, :]).sum(-1)
            outputs.append(y + self.D * x[:, t])

        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * F.silu(gate))


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = MambaMixer(dim, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = x + self.dropout(self.mixer(self.norm(x)))

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


class CoDeMa(CDT):
    def __init__(self, embedding_dim=128, num_layers=3, d_state=16, d_conv=4, expand=2,
                 residual_dropout=0.1, **kwargs):
        super().__init__(embedding_dim=embedding_dim, num_layers=0, residual_dropout=residual_dropout, **kwargs)

        self.blocks = nn.ModuleList([MambaBlock(embedding_dim, d_state, d_conv, expand, residual_dropout)
                                     for _ in range(num_layers)])


def build(env, model_config, trainer_config, device):
    action_dim = env.action_space.shape[0]

    model = CoDeMa(state_dim=env.observation_space.shape[0], action_dim=action_dim,
                   max_action=float(env.action_space.high[0]), episode_len=env.spec.max_episode_steps,
                   target_entropy=-action_dim,**model_config).to(device)

    if model.stochastic:
        model.log_temperature = (model.log_temperature.detach().to(device).requires_grad_(True))

    trainer = CDTTrainer(model=model, env=env, device=device, **trainer_config)

    return model, trainer