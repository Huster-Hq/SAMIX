from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


LABEL_TYPES = ("full", "box", "scribble", "point")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_link(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        dst.symlink_to(src)
        return
    if link_mode == "hardlink":
        os.link(src, dst)
        return
    shutil.copy2(src, dst)


def _load_box_txt(path: Path, width: int, height: int) -> List[Dict[str, object]]:
    boxes: List[Dict[str, object]] = []
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
    return boxes


def _load_point_txt(path: Path, width: int, height: int) -> List[Dict[str, object]]:
    points: List[Dict[str, object]] = []
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
        points.append(
            {
                "xy": [x_norm * width, y_norm * height],
                "label": int(label),
                "class_id": 1,
                "class_name": "polyp" if int(label) == 1 else "background",
            }
        )
    return points


def _sample_scribble_points(mask: np.ndarray, max_points: int) -> List[List[int]]:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return []
    if len(coords) > max_points:
        indices = np.linspace(0, len(coords) - 1, max_points, dtype=np.int64)
        coords = coords[indices]
    return [[int(x), int(y)] for y, x in coords]


def _write_json(path: Path, content: Dict[str, object]) -> None:
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def _convert_dataset(
    source_root: Path,
    output_root: Path,
    link_mode: str,
    max_scribble_points: int,
) -> Dict[str, object]:
    _ensure_dir(output_root)
    images_dir = output_root / "images"
    masks_dir = output_root / "masks"
    boxes_dir = output_root / "boxes"
    scribbles_dir = output_root / "scribbles"
    points_dir = output_root / "points"
    for directory in (images_dir, masks_dir, boxes_dir, scribbles_dir, points_dir):
        _ensure_dir(directory)

    manifest_images: Dict[str, object] = {}
    split_ids: List[str] = []
    stats = {label: 0 for label in LABEL_TYPES}

    for label_type in LABEL_TYPES:
        src_image_dir = source_root / label_type / "images"
        src_label_dir = source_root / label_type / "masks"
        for src_image_path in sorted(src_image_dir.iterdir()):
            if not src_image_path.is_file():
                continue
            image_id = src_image_path.stem
            if image_id in manifest_images:
                raise ValueError(f"Duplicate image id detected across label pools: {image_id}")

            with Image.open(src_image_path) as image:
                width, height = image.size

            dst_image_path = images_dir / src_image_path.name
            _safe_link(src_image_path.resolve(), dst_image_path, link_mode)

            labels: Dict[str, str] = {}
            if label_type == "full":
                src_mask_path = src_label_dir / f"{image_id}.png"
                dst_mask_path = masks_dir / src_mask_path.name
                _safe_link(src_mask_path.resolve(), dst_mask_path, link_mode)
                labels["mask"] = f"masks/{dst_mask_path.name}"
            elif label_type == "box":
                src_box_path = src_label_dir / f"{image_id}.txt"
                box_json_path = boxes_dir / f"{image_id}.json"
                box_content = {
                    "image_id": image_id,
                    "boxes": _load_box_txt(src_box_path, width=width, height=height),
                }
                _write_json(box_json_path, box_content)
                labels["box"] = f"boxes/{box_json_path.name}"
            elif label_type == "point":
                src_point_path = src_label_dir / f"{image_id}.txt"
                point_json_path = points_dir / f"{image_id}.json"
                point_content = {
                    "image_id": image_id,
                    "points": _load_point_txt(src_point_path, width=width, height=height),
                }
                _write_json(point_json_path, point_content)
                labels["point"] = f"points/{point_json_path.name}"
            elif label_type == "scribble":
                src_scribble_path = src_label_dir / f"{image_id}.png"
                mask = np.asarray(Image.open(src_scribble_path).convert("L"), dtype=np.uint8)
                scribble_json_path = scribbles_dir / f"{image_id}.json"
                scribble_content = {
                    "image_id": image_id,
                    "scribbles": [
                        {
                            "points": _sample_scribble_points(mask, max_points=max_scribble_points),
                            "class_id": 1,
                            "class_name": "polyp",
                        }
                    ],
                }
                _write_json(scribble_json_path, scribble_content)
                labels["scribble"] = f"scribbles/{scribble_json_path.name}"
            else:
                raise ValueError(f"Unsupported label type: {label_type}")

            manifest_images[image_id] = {
                "file_name": f"images/{dst_image_path.name}",
                "split": "train",
                "width": width,
                "height": height,
                "labels": labels,
                "source_pool": label_type,
            }
            split_ids.append(image_id)
            stats[label_type] += 1

    manifest = {
        "dataset_name": "samix_joint_polyp_train",
        "root": ".",
        "annotation_types": ["mask", "box", "scribble", "point"],
        "splits": {"train": split_ids, "val": [], "test": []},
        "images": manifest_images,
        "stats": stats,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the joint polyp training pools into one SAMIX-style dataset root.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--max-scribble-points", type=int, default=256)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = _convert_dataset(
        source_root=source_root,
        output_root=output_root,
        link_mode=args.link_mode,
        max_scribble_points=args.max_scribble_points,
    )
    _write_json(output_root / "splits.json", manifest)
    print(f"created={output_root}")
    print(json.dumps(manifest["stats"], ensure_ascii=True))
    print(f"train_samples={len(manifest['splits']['train'])}")


if __name__ == "__main__":
    main()
