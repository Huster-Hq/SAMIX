from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        super().__init__()
        self.module = copy.deepcopy(model)
        self.decay = decay
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            if not value.dtype.is_floating_point:
                value.copy_(model_state[key])
                continue
            value.mul_(self.decay).add_(model_state[key].detach(), alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
