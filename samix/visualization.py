from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def _mask_to_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask[0]
    return mask > 0.5


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: Sequence[float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
) -> torch.Tensor:
    image = denormalize_image(image.detach().cpu())
    mask_tensor = mask.detach().cpu().float()
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.shape[-2:] != image.shape[-2:]:
        mask_tensor = F.interpolate(mask_tensor, size=image.shape[-2:], mode="bilinear", align_corners=False)
    mask_bool = _mask_to_bool(mask_tensor[0])
    overlay = image.clone()
    color_tensor = overlay.new_tensor(color).view(3, 1, 1)
    blended = (1.0 - alpha) * overlay + alpha * color_tensor
    overlay = torch.where(mask_bool.unsqueeze(0), blended, overlay)
    return overlay.clamp(0.0, 1.0)


def _to_pil(image: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(denormalize_image(image.detach().cpu()))


def draw_annotation(
    image: torch.Tensor,
    label_type: Optional[str],
    annotation: Optional[Dict[str, Any]],
) -> torch.Tensor:
    if label_type == "mask" and annotation is not None:
        return overlay_mask(image, torch.as_tensor(annotation["mask"]).float(), color=(0.0, 1.0, 0.0))

    canvas = _to_pil(image)
    draw = ImageDraw.Draw(canvas)

    if label_type == "box" and annotation is not None:
        for box in annotation.get("boxes", []):
            draw.rectangle(box["xyxy"], outline=(255, 255, 0), width=4)
    elif label_type == "point" and annotation is not None:
        for point in annotation.get("points", []):
            x_coord, y_coord = point["xy"]
            radius = 6
            color = (0, 255, 0) if point.get("label", 1) == 1 else (255, 0, 0)
            draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), outline=color, width=3)
    elif label_type == "scribble" and annotation is not None:
        for stroke in annotation.get("scribbles", []):
            points = stroke.get("points", [])
            if len(points) >= 2:
                draw.line([tuple(point) for point in points], fill=(255, 165, 0), width=3)
            for point in points[:: max(1, len(points) // 32 or 1)]:
                radius = 2
                x_coord, y_coord = point
                draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), fill=(255, 165, 0))

    return TF.to_tensor(canvas)


def build_episode_visual(
    support_images: torch.Tensor,
    support_masks: torch.Tensor,
    query_image: torch.Tensor,
    query_label_type: Optional[str],
    query_annotation: Optional[Dict[str, Any]],
    sa_logits: torch.Tensor,
    seg_logits: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    panels = []

    for shot_index in range(support_images.shape[0]):
        panels.append(
            overlay_mask(
                support_images[shot_index],
                support_masks[shot_index],
                color=(0.0, 1.0, 0.0),
            )
        )

    panels.append(denormalize_image(query_image.detach().cpu()))
    panels.append(draw_annotation(query_image, query_label_type, query_annotation))
    panels.append(overlay_mask(query_image, torch.sigmoid(sa_logits), color=(1.0, 0.0, 0.0)))

    if seg_logits is not None:
        panels.append(overlay_mask(query_image, torch.sigmoid(seg_logits), color=(0.0, 0.5, 1.0)))
    if query_mask is not None:
        panels.append(overlay_mask(query_image, query_mask, color=(0.0, 1.0, 0.0)))

    stacked = torch.stack(panels, dim=0)
    return make_grid(stacked, nrow=len(panels), padding=8)


def save_visual(grid: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
