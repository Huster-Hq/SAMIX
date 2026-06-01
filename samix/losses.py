from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def _resize_target_like(logits: torch.Tensor, target: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    if target.shape[-2:] == logits.shape[-2:]:
        return target
    if mode == "nearest":
        return F.interpolate(target.float(), size=logits.shape[-2:], mode=mode)
    return F.interpolate(target.float(), size=logits.shape[-2:], mode=mode, align_corners=False)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = _resize_target_like(logits, target, mode="nearest")
    probs = torch.sigmoid(logits)
    numerator = 2 * (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - ((numerator + eps) / denominator).mean()


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _resize_target_like(logits, target, mode="nearest")
    weight = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3)).clamp_min(1e-6)
    probs = torch.sigmoid(logits)
    inter = ((probs * target) * weight).sum(dim=(2, 3))
    union = ((probs + target) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def mask_supervision_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return structure_loss(logits, target) + dice_loss(logits, target)


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))


def pseudo_label_loss(student_logits: torch.Tensor, pseudo_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pseudo_mask = _resize_target_like(student_logits, pseudo_mask, mode="bilinear")
    target = (pseudo_mask >= threshold).float()
    return F.binary_cross_entropy_with_logits(student_logits, target)


def _dice_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = 2 * (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - ((numerator + eps) / denominator).mean()


def _build_box_mask(height: int, width: int, boxes: List[Dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, height, width), device=device, dtype=dtype)
    for box in boxes:
        x0, y0, x1, y1 = box["xyxy"]
        x0 = int(max(0, min(width - 1, round(min(x0, x1)))))
        y0 = int(max(0, min(height - 1, round(min(y0, y1)))))
        x1 = int(max(0, min(width, round(max(x0, x1)))))
        y1 = int(max(0, min(height, round(max(y0, y1)))))
        if x1 <= x0 or y1 <= y0:
            continue
        mask[:, y0:y1, x0:x1] = 1.0
    return mask


def _mask_to_box_transform(prob_map: torch.Tensor, boxes: List[Dict]) -> torch.Tensor:
    """
    Differentiable Mask-to-Box (M2B) transformation.

    For each box region:
    1. project the predicted mask with max along width/height
    2. back-project the 1D descriptors
    3. intersect them with min() to obtain a box-aligned support mask
    """
    transformed = prob_map.clone()
    _, height, width = transformed.shape
    for box in boxes:
        x0, y0, x1, y1 = box["xyxy"]
        x0 = int(max(0, min(width - 1, round(min(x0, x1)))))
        y0 = int(max(0, min(height - 1, round(min(y0, y1)))))
        x1 = int(max(0, min(width, round(max(x0, x1)))))
        y1 = int(max(0, min(height, round(max(y0, y1)))))
        if x1 <= x0 or y1 <= y0:
            continue

        patch = transformed[:, y0:y1, x0:x1]
        proj_w = patch.max(dim=1, keepdim=True).values
        proj_h = patch.max(dim=2, keepdim=True).values
        back_w = proj_w.expand(-1, patch.shape[1], -1)
        back_h = proj_h.expand(-1, -1, patch.shape[2])
        transformed[:, y0:y1, x0:x1] = torch.minimum(back_w, back_h)
    return transformed


def point_supervision_loss(logits: torch.Tensor, points_batch: List[List[Dict]], image_size: int) -> torch.Tensor:
    if not points_batch:
        return logits.new_tensor(0.0)
    losses = []
    probs = torch.sigmoid(logits[:, 0])
    for batch_index, points in enumerate(points_batch):
        if not points:
            continue
        sample_losses = []
        for point in points:
            x_coord, y_coord = point["xy"]
            x_coord = int(max(0, min(image_size - 1, round(x_coord))))
            y_coord = int(max(0, min(image_size - 1, round(y_coord))))
            target = probs.new_tensor(float(point.get("label", 1)))
            sample_losses.append(F.binary_cross_entropy(probs[batch_index, y_coord, x_coord], target))
        losses.append(torch.stack(sample_losses).mean())
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def scribble_supervision_loss(logits: torch.Tensor, scribbles_batch: List[List[Dict]], image_size: int) -> torch.Tensor:
    flattened_points = []
    for scribbles in scribbles_batch:
        points = []
        for stroke in scribbles:
            for xy in stroke["points"]:
                points.append({"xy": xy, "label": 1})
        flattened_points.append(points)
    return point_supervision_loss(logits, flattened_points, image_size=image_size)


def box_supervision_loss(logits: torch.Tensor, boxes_batch: List[List[Dict]], image_size: int) -> torch.Tensor:
    probs = torch.sigmoid(logits[:, 0:1])
    losses = []
    for batch_index, boxes in enumerate(boxes_batch):
        if not boxes:
            continue
        target_box_mask = _build_box_mask(
            height=image_size,
            width=image_size,
            boxes=boxes,
            device=probs.device,
            dtype=probs.dtype,
        ).unsqueeze(0)
        transformed = _mask_to_box_transform(probs[batch_index], boxes).unsqueeze(0)
        bce = F.binary_cross_entropy(transformed.clamp(1e-6, 1 - 1e-6), target_box_mask)
        dice = _dice_loss_from_probs(transformed, target_box_mask)
        losses.append(bce + dice)
    return torch.stack(losses).mean() if losses else logits.new_tensor(0.0)


def class_presence_loss(logits: torch.Tensor, classes_batch: List[List[Dict]]) -> torch.Tensor:
    probs = torch.sigmoid(logits[:, 0])
    pooled = probs.flatten(1).amax(dim=1)
    targets = logits.new_tensor([1.0 if classes else 0.0 for classes in classes_batch])
    return F.binary_cross_entropy(pooled, targets)
