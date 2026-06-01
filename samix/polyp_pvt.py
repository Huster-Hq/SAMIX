from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class CFM(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state: int, num_node: int, bias: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SimilarityAggregationModule(nn.Module):
    def __init__(self, num_in: int = 32, plane_mid: int = 16, mids: int = 4, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = F.interpolate(edge, size=x.shape[-2:], mode="bilinear", align_corners=True)
        batch, _, _, _ = x.shape
        edge = F.softmax(edge, dim=1)[:, 1:2]
        x_state = self.conv_state(x).view(batch, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(batch, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(batch, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)
        x_n_state = torch.matmul(x_state, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1.0 / x_state.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_state = torch.matmul(x_n_rel, x_proj_reshaped).view(batch, self.num_s, *x.shape[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class PVTv2Backbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for the Polyp-PVT backbone.") from exc
        self.model = timm.create_model("pvt_v2_b2", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))

    def forward(self, x: torch.Tensor):
        return self.model(x)


class PolypPVT(nn.Module):
    """
    Polyp-PVT-style segmentation model with an optional timm PVTv2-B2 backbone.
    """

    def __init__(self, channel: int = 32, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = PVTv2Backbone(pretrained=pretrained_backbone)
        self.trans2_0 = BasicConv2d(64, channel, 1)
        self.trans2_1 = BasicConv2d(128, channel, 1)
        self.trans3_1 = BasicConv2d(320, channel, 1)
        self.trans4_1 = BasicConv2d(512, channel, 1)
        self.cfm = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.sam = SimilarityAggregationModule(num_in=channel, plane_mid=channel // 2, mids=4)
        self.down05 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.out_sam = nn.Conv2d(channel, 1, 1)
        self.out_cfm = nn.Conv2d(channel, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1, x2, x3, x4 = self.backbone(x)
        x1 = x1.permute(0, 3, 1, 2) if x1.dim() == 4 and x1.shape[1] != 64 else x1
        x2 = x2.permute(0, 3, 1, 2) if x2.dim() == 4 and x2.shape[1] != 128 else x2
        x3 = x3.permute(0, 3, 1, 2) if x3.dim() == 4 and x3.shape[1] != 320 else x3
        x4 = x4.permute(0, 3, 1, 2) if x4.dim() == 4 and x4.shape[1] != 512 else x4

        cim = self.sa(self.ca(x1) * x1) * x1
        x2_t = self.trans2_1(x2)
        x3_t = self.trans3_1(x3)
        x4_t = self.trans4_1(x4)
        cfm = self.cfm(x4_t, x3_t, x2_t)

        t2 = self.down05(self.trans2_0(cim))
        sam_feature = self.sam(cfm, t2)
        pred_cfm = F.interpolate(self.out_cfm(cfm), scale_factor=8, mode="bilinear", align_corners=False)
        pred_sam = F.interpolate(self.out_sam(sam_feature), scale_factor=8, mode="bilinear", align_corners=False)
        return {"aux_logits": pred_cfm, "logits": pred_cfm + pred_sam, "sam_logits": pred_sam}
