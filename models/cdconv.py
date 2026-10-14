import torch
from torch import nn
from torch.nn import functional as F
from osrl.algorithms.cdt import CDT, CDTTrainer


class ModalityCausalConv(nn.Module):
    """Decision ConvFormer token mixer: one causal depthwise filter per modality (R, C, s, a).
    The output at position i uses the filter of the modality of token i; left-only padding."""

    def __init__(self, dim, num_modalities, kernel_size):
        super().__init__()
        self.num_modalities = num_modalities
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)
                                    for _ in range(num_modalities)])

    def forward(self, x):
        length = x.shape[1]
        x = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = torch.stack([conv(x) for conv in self.convs], dim=0)  # [modalities, batch, dim, length]
        position = torch.arange(length, device=x.device)

        return y[position % self.num_modalities, :, :, position].permute(1, 0, 2)

    def step(self, x, buffer, position):
        """One token. x [batch, dim]; buffer [batch, kernel - 1, dim] holds the previous inputs."""
        window = torch.cat([buffer, x.unsqueeze(1)], dim=1)
        y = self.convs[position % self.num_modalities](window.transpose(1, 2)).squeeze(-1)

        return y, window[:, 1:]


class ConvBlock(nn.Module):
    def __init__(self, dim, num_modalities, kernel_size, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.mixer = ModalityCausalConv(dim, num_modalities, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, padding_mask=None):
        # causal and padding is on the right only, so the padding mask is not needed
        x = x + self.dropout(self.mixer(self.norm1(x)))
        return x + self.mlp(self.norm2(x))

    def step(self, x, buffer, position):
        y, buffer = self.mixer.step(self.norm1(x), buffer, position)
        x = x + self.dropout(y)

        return x + self.mlp(self.norm2(x)), buffer


class CDConv(CDT):
    def __init__(self, embedding_dim=128, num_layers=3, conv_window=2, residual_dropout=0.1, **kwargs):
        super().__init__(embedding_dim=embedding_dim, num_layers=0, residual_dropout=residual_dropout, **kwargs)

        if conv_window not in (1, 2, 3):
            raise ValueError("conv_window must be 1, 2 or 3 timesteps")

        # kernel of conv_window timesteps, e.g. 4 modalities x 2 = 8 tokens
        self.kernel_size = self.seq_repeat * conv_window
        self.blocks = nn.ModuleList([ConvBlock(embedding_dim, self.seq_repeat, self.kernel_size, residual_dropout)
                                     for _ in range(num_layers)])

        # CDT's init for Linear/LayerNorm; Conv1d keeps the PyTorch init (as CDT's init does not touch it)
        self.blocks.apply(self._init_weights)


    # recurrent mode, only used by efficiency.py
    def init_recurrent(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        buffers = [torch.zeros(batch_size, self.kernel_size - 1, self.embedding_dim, device=device)
                   for _ in self.blocks]
        return {"buffers": buffers, "position": 0}

    def recurrent_step(self, x, state):
        buffers = []
        for block, buffer in zip(self.blocks, state["buffers"]):
            x, buffer = block.step(x, buffer, state["position"])
            buffers.append(buffer)

        return x, {"buffers": buffers, "position": state["position"] + 1}


def build(env, model_config, trainer_config, device):
    action_dim = env.action_space.shape[0]

    model = CDConv(state_dim=env.observation_space.shape[0], action_dim=action_dim,
                   max_action=float(env.action_space.high[0]), episode_len=env.spec.max_episode_steps,
                   target_entropy=-action_dim, **model_config).to(device)

    if model.stochastic:
        model.log_temperature = (model.log_temperature.detach().to(device).requires_grad_(True))

    trainer = CDTTrainer(model=model, env=env, device=device, **trainer_config)

    return model, trainer
