"""CPU stand-in for mamba_ssm, for CLI smoke tests only (Mamba = Linear)."""
import torch.nn as nn


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kw):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.fc(x)
