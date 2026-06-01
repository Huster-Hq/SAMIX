from typing import Dict

import torch
from torch import nn

from .backbones import AdapterEnhancedEncoder
from .retrieval import build_support_bank, fuse_retrieved_prototypes, retrieve_topk


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SAMIXLite(nn.Module):
    """
    A concise support-retrieval segmentation baseline.

    Inputs:
        support_images: [B, S, 3, H, W]
        support_masks:  [B, S, 1, H, W]
        query_images:   [B, 3, H, W]
    """

    def __init__(self, encoder: nn.Module, topk: int = 1) -> None:
        super().__init__()
        self.encoder = encoder
        self.topk = topk
        out_channels = getattr(encoder, "out_channels")
        self.head = SegmentationHead(out_channels * 2)

    def encode_support(self, support_images: torch.Tensor) -> torch.Tensor:
        b, s, c, h, w = support_images.shape
        features = self.encoder(support_images.reshape(b * s, c, h, w))
        _, c_out, h_out, w_out = features.shape
        return features.reshape(b, s, c_out, h_out, w_out)

    def forward(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        support_features = self.encode_support(support_images)
        support_bank = build_support_bank(support_features, support_masks)

        query_features = self.encoder(query_images)
        retrieved = retrieve_topk(query_features, support_bank, topk=self.topk)
        fused, aggregated_proto = fuse_retrieved_prototypes(query_features, retrieved)
        logits = self.head(fused)
        logits = torch.nn.functional.interpolate(
            logits,
            size=query_images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {
            "logits": logits,
            "retrieval_scores": retrieved["scores"],
            "retrieval_indices": retrieved["indices"],
            "retrieved_prototypes": retrieved["prototypes"],
            "query_prototype": aggregated_proto,
        }


def build_samix_lite(topk: int = 1) -> SAMIXLite:
    encoder = AdapterEnhancedEncoder()
    return SAMIXLite(encoder=encoder, topk=topk)
