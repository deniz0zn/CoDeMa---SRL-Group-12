import torch
from torch import nn
import mamba_ssm  # keep this here or my house burns down
from mamba_ssm import Mamba
from osrl.algorithms.cdt import CDT, CDTTrainer


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, dropout):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mixer = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand, use_fast_path=True,)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, padding_mask=None):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        x = x + self.dropout(self.mixer(self.norm(x)))

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


class CDM(CDT):
    def __init__(self, embedding_dim=128, num_layers=3, d_state=16, d_conv=4, expand=2,
                 residual_dropout=0.1, **kwargs,):
        super().__init__(embedding_dim=embedding_dim, num_layers=0, residual_dropout=residual_dropout, **kwargs,)


        self.blocks = nn.ModuleList([MambaBlock(dim=embedding_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                                                dropout=residual_dropout,)for _ in range(num_layers)])


def build(env, model_config, trainer_config, device):
    if torch.device(device).type != "cuda":
        raise ValueError("Official CDM requires device: cuda.")

    action_dim = env.action_space.shape[0]

    model = CDM(state_dim=env.observation_space.shape[0], action_dim=action_dim,
                max_action=float(env.action_space.high[0]), episode_len=env.spec.max_episode_steps,
                target_entropy=-action_dim, **model_config, ).to(device)

    if model.stochastic:
        model.log_temperature = (model.log_temperature.detach().to(device).requires_grad_(True))


    trainer = CDTTrainer(model=model, env=env, device=device, **trainer_config,)

    return model, trainer

# if __name__ == "__main__":
#     print(mamba_ssm.__file__)