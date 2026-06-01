from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


SUPERVISION_PRIORITY = ["mask", "scribble", "box", "point", "class", "unlabeled"]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_xy(xy: Sequence[float], sx: float, sy: float) -> List[float]:
    return [float(xy[0]) * sx, float(xy[1]) * sy]


def _resize_annotation(annotation_type: str, annotation: Dict[str, Any], sx: float, sy: float) -> Dict[str, Any]:
    if annotation_type == "mask":
        mask = annotation["mask"]
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        pil_mask = pil_mask.resize((int(round(mask.shape[1] * sx)), int(round(mask.shape[0] * sy))), Image.NEAREST)
        annotation["mask"] = np.asarray(pil_mask, dtype=np.float32)
    elif annotation_type == "point":
        for point in annotation["points"]:
            point["xy"] = _scale_xy(point["xy"], sx, sy)
    elif annotation_type == "box":
        for box in annotation["boxes"]:
            x0, y0, x1, y1 = box["xyxy"]
            box["xyxy"] = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
    elif annotation_type == "scribble":
        for stroke in annotation["scribbles"]:
            stroke["points"] = [_scale_xy(point, sx, sy) for point in stroke["points"]]
    return annotation


class MixedSupervisionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        image_size: int = 512,
        normalize: bool = True,
        preferred_label_types: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = _load_json(self.manifest_path)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.preferred_label_types = list(preferred_label_types or SUPERVISION_PRIORITY)
        self.ids = list(self.manifest["splits"][split])

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_relpath: str) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
        image_path = self.root / image_relpath
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size[1], image.size[0]
        sx = self.image_size / image.size[0]
        sy = self.image_size / image.size[1]
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        if self.normalize:
            tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, orig_size, (sx, sy)

    def _load_annotation(self, image_id: str, image_info: Dict[str, Any], scale_xy: tuple[float, float]) -> Dict[str, Dict[str, Any]]:
        labels = image_info.get("labels", {})
        annotations: Dict[str, Dict[str, Any]] = {}
        sx, sy = scale_xy
        for label_type, relpath in labels.items():
            label_path = self.root / relpath
            if label_type == "mask":
                mask = np.asarray(Image.open(label_path).convert("L"), dtype=np.float32)
                mask = (mask > 0).astype(np.float32)
                annotations[label_type] = {"mask": mask}
            else:
                annotations[label_type] = _load_json(label_path)
            annotations[label_type] = _resize_annotation(label_type, annotations[label_type], sx, sy)
        return annotations

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.ids[index]
        image_info = self.manifest["images"][image_id]
        image, orig_size, scale_xy = self._load_image(image_info["file_name"])
        annotations = self._load_annotation(image_id, image_info, scale_xy)
        label_type = next((name for name in self.preferred_label_types if name in annotations), None)
        return {
            "image_id": image_id,
            "image": image,
            "orig_size": orig_size,
            "split": self.split,
            "annotations": annotations,
            "label_type": label_type,
            "annotation": annotations.get(label_type),
        }


class WarmupEpisodeDataset(Dataset):
    def __init__(self, base_dataset: MixedSupervisionDataset, shots: int = 1) -> None:
        self.base_dataset = base_dataset
        self.shots = shots
        self.mask_indices = [idx for idx in range(len(base_dataset)) if "mask" in base_dataset[idx]["annotations"]]
        if len(self.mask_indices) < shots + 1:
            raise ValueError("WarmupEpisodeDataset needs enough mask-labeled samples for support and query.")

    def __len__(self) -> int:
        return len(self.mask_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        query_item = self.base_dataset[self.mask_indices[index]]
        support_pool = [idx for idx in self.mask_indices if idx != self.mask_indices[index]]
        support_indices = random.sample(support_pool, k=self.shots)
        support_items = [self.base_dataset[idx] for idx in support_indices]
        support_images = torch.stack([item["image"] for item in support_items], dim=0)
        support_masks = torch.stack(
            [torch.from_numpy(item["annotations"]["mask"]["mask"]).unsqueeze(0) for item in support_items],
            dim=0,
        ).float()
        query_mask = torch.from_numpy(query_item["annotations"]["mask"]["mask"]).unsqueeze(0).float()
        return {
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_item["image"],
            "query_masks": query_mask,
            "query_annotation": query_item["annotation"],
        }


class CotrainEpisodeDataset(Dataset):
    def __init__(
        self,
        base_dataset: MixedSupervisionDataset,
        shots: int = 1,
        query_label_types: Sequence[str] = ("mask", "scribble", "box", "point", "class", "unlabeled"),
    ) -> None:
        self.base_dataset = base_dataset
        self.shots = shots
        self.query_label_types = set(query_label_types)
        self.mask_indices = [idx for idx in range(len(base_dataset)) if "mask" in base_dataset[idx]["annotations"]]
        self.query_indices = [
            idx
            for idx in range(len(base_dataset))
            if base_dataset[idx]["label_type"] in self.query_label_types
        ]
        if len(self.mask_indices) < shots:
            raise ValueError("CotrainEpisodeDataset needs mask-labeled support samples.")

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        query_item = self.base_dataset[self.query_indices[index]]
        support_pool = [idx for idx in self.mask_indices if self.base_dataset[idx]["image_id"] != query_item["image_id"]]
        support_indices = random.sample(support_pool, k=self.shots)
        support_items = [self.base_dataset[idx] for idx in support_indices]
        support_images = torch.stack([item["image"] for item in support_items], dim=0)
        support_masks = torch.stack(
            [torch.from_numpy(item["annotations"]["mask"]["mask"]).unsqueeze(0) for item in support_items],
            dim=0,
        ).float()
        query_mask = None
        if "mask" in query_item["annotations"]:
            query_mask = torch.from_numpy(query_item["annotations"]["mask"]["mask"]).unsqueeze(0).float()
        return {
            "support_images": support_images,
            "support_masks": support_masks,
            "query_images": query_item["image"],
            "query_masks": query_mask,
            "query_label_type": query_item["label_type"],
            "query_annotation": query_item["annotation"],
        }


def warmup_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "support_images": torch.stack([item["support_images"] for item in batch], dim=0),
        "support_masks": torch.stack([item["support_masks"] for item in batch], dim=0),
        "query_images": torch.stack([item["query_images"] for item in batch], dim=0),
        "query_masks": torch.stack([item["query_masks"] for item in batch], dim=0),
    }


def cotrain_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    query_masks = [item["query_masks"] for item in batch]
    stacked_masks = None
    if all(mask is not None for mask in query_masks):
        stacked_masks = torch.stack(query_masks, dim=0)
    return {
        "support_images": torch.stack([item["support_images"] for item in batch], dim=0),
        "support_masks": torch.stack([item["support_masks"] for item in batch], dim=0),
        "query_images": torch.stack([item["query_images"] for item in batch], dim=0),
        "query_masks": stacked_masks,
        "query_label_types": [item["query_label_type"] for item in batch],
        "query_annotations": [item["query_annotation"] for item in batch],
    }
