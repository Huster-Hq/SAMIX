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
    def __init__(self, sa_sam2: SASAM2FewShotSegmentor, seg_model: PolypPVT, ema_decay: float = 0.999) -> None:
        super().__init__()
        self.sa_sam2 = sa_sam2
        self.seg_model = seg_model
        self.ema_model = ModelEMA(seg_model, decay=ema_decay)

    def forward(self, query_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.seg_model(query_images)
