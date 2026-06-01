import math

import torch
from torch import nn


class SemanticAdapter(nn.Module):
    """AdaptFormer-style bottleneck adapter for channels-last tokens."""

    def __init__(self, dim: int, bottleneck: int = 64, scale: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck, dim)
        self.scale = scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.scale


class FeatureMapAdapter(nn.Module):
    """Applies a semantic adapter on a BCHW feature map."""

    def __init__(self, channels: int, bottleneck: int, scale: float = 0.1) -> None:
        super().__init__()
        self.adapter = SemanticAdapter(channels, bottleneck=bottleneck, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        tokens = tokens + self.adapter(tokens)
        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
