from typing import Iterable, Tuple

import torch
from torch import nn

from .adapters import FeatureMapAdapter


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdapterEnhancedEncoder(nn.Module):
    """Small encoder with late-stage semantic adapters."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, ...] = (64, 128, 256, 256),
        adapter_stages: Iterable[int] = (2, 3),
        adapter_bottleneck: int = 64,
        adapter_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        stages = []
        adapters = []
        in_ch = channels[0]
        adapter_stages = set(adapter_stages)
        for index, out_ch in enumerate(channels):
            stride = 1 if index == 0 else 2
            stages.append(ConvBlock(in_ch, out_ch, stride=stride))
            if index in adapter_stages:
                adapters.append(FeatureMapAdapter(out_ch, bottleneck=adapter_bottleneck, scale=adapter_scale))
            else:
                adapters.append(nn.Identity())
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.adapters = nn.ModuleList(adapters)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage, adapter in zip(self.stages, self.adapters):
            x = stage(x)
            x = adapter(x)
        return x
