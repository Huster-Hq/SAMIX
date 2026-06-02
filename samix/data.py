from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _sample_scribble_points(mask: np.ndarray, max_points: int = 256) -> List[List[float]]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return []
    if len(coords) > max_points:
        indices = np.linspace(0, len(coords) - 1, max_points, dtype=np.int64)
        coords = coords[indices]
    return [[float(x), float(y)] for y, x in coords]


def _parse_box_txt(path: Path, width: int, height: int) -> Dict[str, Any]:
    boxes: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Unexpected box format in {path}: {line}")
        class_id, cx, cy, bw, bh = map(float, parts)
        x0 = (cx - bw / 2.0) * width
        y0 = (cy - bh / 2.0) * height
        x1 = (cx + bw / 2.0) * width
        y1 = (cy + bh / 2.0) * height
        boxes.append(
            {
                "xyxy": [x0, y0, x1, y1],
                "class_id": int(class_id),
                "class_name": "polyp",
            }
        )
    return {"boxes": boxes}


def _parse_point_txt(path: Path, width: int, height: int) -> Dict[str, Any]:
    points: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 2:
            x_norm, y_norm = map(float, parts)
            label = 1
        elif len(parts) == 3:
            x_norm, y_norm, label = map(float, parts)
        else:
            raise ValueError(f"Unexpected point format in {path}: {line}")
        click_label = int(label)
        points.append(
            {
                "xy": [x_norm * width, y_norm * height],
                "label": click_label,
                "class_name": "polyp" if click_label == 1 else "background",
                "class_id": 0 if click_label == 1 else -1,
            }
        )
    return {"points": points}


def _parse_scribble_png(path: Path, max_points: int = 256) -> Dict[str, Any]:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return {
        "scribbles": [
            {
                "points": _sample_scribble_points(mask, max_points=max_points),
                "class_name": "polyp",
                "class_id": 1,
            }
        ]
    }


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


class NearestSupportSelector:
    def __init__(self, ordered_support_indices: Dict[int, List[int]]) -> None:
        self.ordered_support_indices = ordered_support_indices

    def select(self, query_index: int, shots: int, exclude_indices: Optional[Sequence[int]] = None) -> List[int]:
        exclude = set(exclude_indices or [])
        support_indices = [
            idx for idx in self.ordered_support_indices.get(query_index, [])
            if idx not in exclude
        ]
        if len(support_indices) < shots:
            raise ValueError(
                f"Not enough nearest-neighbor support samples for query index {query_index}: "
                f"requested {shots}, found {len(support_indices)}"
            )
        return support_indices[:shots]


def build_nearest_support_selector(
    base_dataset: "MixedSupervisionDataset",
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    candidate_indices: Optional[Sequence[int]] = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> NearestSupportSelector:
    candidate_list = list(candidate_indices) if candidate_indices is not None else list(range(len(base_dataset)))

    images: List[torch.Tensor] = []
    for idx in range(len(base_dataset)):
        item = base_dataset[idx]
        images.append(item["image"])

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(images[start : start + batch_size], dim=0).to(device)
        batch_embeddings = feature_extractor(batch).detach().cpu()
        embeddings.append(batch_embeddings)
    embedding_tensor = torch.cat(embeddings, dim=0)

    candidate_tensor = embedding_tensor[candidate_list]
    similarity = embedding_tensor @ candidate_tensor.T
    ordered_support_indices: Dict[int, List[int]] = {}
    for query_index in range(len(base_dataset)):
        order = torch.argsort(similarity[query_index], descending=True).tolist()
        ordered_support_indices[query_index] = [candidate_list[idx] for idx in order if candidate_list[idx] != query_index]
    return NearestSupportSelector(ordered_support_indices)


class MixedSupervisionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        image_size: int = 512,
        normalize: bool = True,
        preferred_label_types: Optional[Sequence[str]] = None,
        max_scribble_points: int = 256,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = _load_json(self.manifest_path)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.preferred_label_types = list(preferred_label_types or SUPERVISION_PRIORITY)
        self.max_scribble_points = max_scribble_points
        self.ids = list(self.manifest["splits"][split])
        self.uses_sample_manifest = "samples" in self.manifest

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

    def _load_annotation(self, image_info: Dict[str, Any], scale_xy: tuple[float, float]) -> Dict[str, Dict[str, Any]]:
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

    def _load_polyp_annotation(
        self,
        sample_info: Dict[str, Any],
        orig_size: tuple[int, int],
        scale_xy: tuple[float, float],
    ) -> Dict[str, Dict[str, Any]]:
        annotation_type = sample_info["annotation_type"]
        annotation_path = self.root / sample_info["annotation_path"]
        height, width = orig_size
        if annotation_type == "mask":
            mask = np.asarray(Image.open(annotation_path).convert("L"), dtype=np.float32)
            annotation = {"mask": (mask > 0).astype(np.float32)}
        elif annotation_type == "box":
            annotation = _parse_box_txt(annotation_path, width=width, height=height)
        elif annotation_type == "point":
            annotation = _parse_point_txt(annotation_path, width=width, height=height)
        elif annotation_type == "scribble":
            annotation = _parse_scribble_png(annotation_path, max_points=self.max_scribble_points)
        else:
            raise ValueError(f"Unsupported annotation type in sample manifest: {annotation_type}")
        return {annotation_type: _resize_annotation(annotation_type, annotation, scale_xy[0], scale_xy[1])}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.ids[index]
        if self.uses_sample_manifest:
            image_info = self.manifest["samples"][image_id]
            image, orig_size, scale_xy = self._load_image(image_info["image_path"])
            annotations = self._load_polyp_annotation(image_info, orig_size, scale_xy)
        else:
            image_info = self.manifest["images"][image_id]
            image, orig_size, scale_xy = self._load_image(image_info["file_name"])
            annotations = self._load_annotation(image_info, scale_xy)
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
    def __init__(
        self,
        base_dataset: MixedSupervisionDataset,
        shots: int = 1,
        support_selector: Optional[NearestSupportSelector] = None,
        query_label_types: Sequence[str] = ("mask", "box"),
    ) -> None:
        self.base_dataset = base_dataset
        self.shots = shots
        self.support_selector = support_selector
        self.mask_indices = [idx for idx in range(len(base_dataset)) if "mask" in base_dataset[idx]["annotations"]]
        self.query_label_types = set(query_label_types)
        self.query_indices = [
            idx for idx in range(len(base_dataset))
            if base_dataset[idx]["label_type"] in self.query_label_types
        ]
        if len(self.mask_indices) < shots + 1:
            raise ValueError("WarmupEpisodeDataset needs enough mask-labeled samples for support and query.")
        if not self.query_indices:
            raise ValueError("WarmupEpisodeDataset needs at least one query sample for the configured query label types.")

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        query_dataset_index = self.query_indices[index]
        query_item = self.base_dataset[query_dataset_index]
        if self.support_selector is not None:
            support_indices = self.support_selector.select(query_dataset_index, self.shots, exclude_indices=[query_dataset_index])
        else:
            support_pool = [idx for idx in self.mask_indices if idx != query_dataset_index]
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


class CotrainEpisodeDataset(Dataset):
    def __init__(
        self,
        base_dataset: MixedSupervisionDataset,
        shots: int = 1,
        query_label_types: Sequence[str] = ("mask", "scribble", "box", "point", "class", "unlabeled"),
        support_selector: Optional[NearestSupportSelector] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.shots = shots
        self.query_label_types = set(query_label_types)
        self.support_selector = support_selector
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
        query_dataset_index = self.query_indices[index]
        query_item = self.base_dataset[query_dataset_index]
        if self.support_selector is not None:
            support_indices = self.support_selector.select(query_dataset_index, self.shots, exclude_indices=[query_dataset_index])
        else:
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
    query_masks = [item["query_masks"] for item in batch]
    stacked_masks = None
    if all(mask is not None for mask in query_masks):
        stacked_masks = torch.stack(query_masks, dim=0)
    return {
        "support_images": torch.stack([item["support_images"] for item in batch], dim=0),
        "support_masks": torch.stack([item["support_masks"] for item in batch], dim=0),
        "query_images": torch.stack([item["query_images"] for item in batch], dim=0),
        "query_masks": stacked_masks,
        "query_masks_list": query_masks,
        "query_label_types": [item["query_label_type"] for item in batch],
        "query_annotations": [item["query_annotation"] for item in batch],
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
