from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .ema import ModelEMA
from .polyp_pvt import PolypPVT
from .sa_sam2 import SASAM2FewShotSegmentor


@dataclass
class SAMIXModules:
    sa_sam2: SASAM2FewShotSegmentor
    seg_model: PolypPVT
    ema_model: ModelEMA


class SAMIXCoTrainer(nn.Module):
    def __init__(
        self,
        sa_sam2: SASAM2FewShotSegmentor,
        seg_model: PolypPVT,
        ema_decay: float = 0.999,
        sa_device: str | None = None,
        seg_device: str | None = None,
    ) -> None:
        super().__init__()
        self.sa_sam2 = sa_sam2
        self.seg_model = seg_model
        self.ema_model = ModelEMA(seg_model, decay=ema_decay)
        self.sa_device = torch.device(sa_device) if sa_device is not None else next(sa_sam2.parameters()).device
        self.seg_device = torch.device(seg_device) if seg_device is not None else next(seg_model.parameters()).device

    def forward(self, query_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.seg_model(query_images)
