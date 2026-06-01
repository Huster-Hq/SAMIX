from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def _to_device_tensor(value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def build_mask_prompt(mask: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    mask = torch.as_tensor(mask)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return {"prompt_type": "mask", "prompt": mask.to(device=device, dtype=torch.float32)}


def build_point_prompt(points: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    coords = _to_device_tensor([item["xy"] for item in points], torch.float32, device).view(1, -1, 2)
    labels = _to_device_tensor([item.get("label", 1) for item in points], torch.int32, device).view(1, -1)
    return {"prompt_type": "point", "prompt": {"point_coords": coords, "point_labels": labels}}


def build_box_prompt(boxes: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    xyxy = _to_device_tensor([item["xyxy"] for item in boxes], torch.float32, device).view(-1, 4)
    top_left = torch.minimum(xyxy[:, :2], xyxy[:, 2:])
    bottom_right = torch.maximum(xyxy[:, :2], xyxy[:, 2:])
    coords = torch.stack([top_left, bottom_right], dim=1).view(1, -1, 2)
    labels = torch.tensor([2, 3], dtype=torch.int32, device=device).repeat(1, coords.shape[1] // 2)
    return {"prompt_type": "box", "prompt": {"point_coords": coords, "point_labels": labels}}


def build_scribble_prompt(scribbles: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    all_points: List[List[float]] = []
    for stroke in scribbles:
        all_points.extend(stroke["points"])
    coords = _to_device_tensor(all_points, torch.float32, device).view(1, -1, 2)
    labels = torch.ones((1, coords.shape[1]), dtype=torch.int32, device=device)
    return {"prompt_type": "scribble", "prompt": {"point_coords": coords, "point_labels": labels}}


def annotation_to_prompt(
    annotation_type: Optional[str],
    annotation: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    if annotation_type is None or annotation is None:
        return None
    if annotation_type == "mask":
        return build_mask_prompt(annotation["mask"], device)
    if annotation_type == "point":
        return build_point_prompt(annotation["points"], device)
    if annotation_type == "box":
        return build_box_prompt(annotation["boxes"], device)
    if annotation_type == "scribble":
        return build_scribble_prompt(annotation["scribbles"], device)
    return None
