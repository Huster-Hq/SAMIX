from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class DecoderOutput:
    low_res_masks: Optional[Tensor] = None
    high_res_masks: Optional[Tensor] = None
    obj_ptr: Optional[Tensor] = None
    pix_feat_with_mem: Optional[Tensor] = None
    masks: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.masks is None:
            self.masks = self.low_res_masks


@dataclass
class BackboneOutput:
    orig_size: List[Tuple[int, int]]
    backbone_out: Dict[str, Any]
    vision_feats: List[Tensor]
    vision_pos_embeds: List[Tensor]
    feat_sizes: List[Tuple[int, int]]

    def get_current_feats(self, idx: int) -> List[Tensor]:
        return [x[:, idx : idx + 1, :] for x in self.vision_feats]

    def get_current_pos_embeds(self, idx: int) -> List[Tensor]:
        return [x[:, idx : idx + 1, :] for x in self.vision_pos_embeds]

    def get_current_feats_x16(self, idx: int) -> Tensor:
        feat = self.vision_feats[-1][:, idx : idx + 1, :]
        height = self.feat_sizes[-1][0]
        return feat.permute(1, 2, 0).reshape(feat.size(1), feat.size(2), height, -1)

    def get_high_res_features(self, current_vision_feats: List[Tensor]) -> List[Tensor]:
        high_res = []
        for tensor, spatial in zip(current_vision_feats[:-1], self.feat_sizes[:-1]):
            tensor = tensor.permute(1, 2, 0).contiguous()
            tensor = tensor.view(tensor.size(0), tensor.size(1), *spatial)
            high_res.append(tensor)
        return high_res


@dataclass
class SASAM2Output:
    support_masks: Tensor
    query_masks: Tensor
    support_memory_bank: List[Dict[int, Dict[str, Tensor]]]
    aux: Dict[str, Any]
