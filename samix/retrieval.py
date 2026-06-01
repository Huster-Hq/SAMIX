from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def masked_average_pool(features: torch.Tensor, masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pool BCHW features with binary masks of shape B1HW."""
    masks = F.interpolate(masks, size=features.shape[-2:], mode="nearest")
    weighted = features * masks
    denom = masks.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    pooled = weighted.sum(dim=(-2, -1), keepdim=True) / denom
    return pooled.flatten(1)


def build_support_bank(features: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Args:
        features: [B, S, C, H, W]
        masks: [B, S, 1, H0, W0]
    """
    b, s, c, h, w = features.shape
    flat_features = features.reshape(b * s, c, h, w)
    flat_masks = masks.reshape(b * s, 1, masks.size(-2), masks.size(-1))
    prototypes = masked_average_pool(flat_features, flat_masks)
    prototypes = F.normalize(prototypes, dim=-1)
    return {
        "prototypes": prototypes.reshape(b, s, c),
        "features": features,
        "masks": masks,
    }


def retrieve_topk(query_features: torch.Tensor, support_bank: Dict[str, torch.Tensor], topk: int = 1) -> Dict[str, torch.Tensor]:
    """
    Args:
        query_features: [B, C, H, W]
        support_bank["prototypes"]: [B, S, C]
    """
    query_proto = F.normalize(query_features.mean(dim=(-2, -1)), dim=-1)
    support_proto = support_bank["prototypes"]
    similarity = torch.einsum("bc,bsc->bs", query_proto, support_proto)
    k = min(topk, support_proto.size(1))
    scores, indices = similarity.topk(k=k, dim=1)

    batch_index = torch.arange(query_features.size(0), device=query_features.device)[:, None]
    retrieved = support_proto[batch_index, indices]

    return {
        "scores": scores,
        "indices": indices,
        "prototypes": retrieved,
    }


def fuse_retrieved_prototypes(query_features: torch.Tensor, retrieved: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        fused_feature_map: [B, 2C, H, W]
        aggregated_prototype: [B, C]
    """
    proto = retrieved["prototypes"].mean(dim=1)
    proto_map = proto[:, :, None, None].expand(-1, -1, query_features.size(2), query_features.size(3))
    fused = torch.cat([query_features, proto_map], dim=1)
    return fused, proto
