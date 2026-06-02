from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


SOURCE_LABELS = {
    "full": {"target_dir": "Mask", "label_type": "mask", "ext": ".png"},
    "box": {"target_dir": "Box", "label_type": "box", "ext": ".txt"},
    "scribble": {"target_dir": "Scribble", "label_type": "scribble", "ext": ".png"},
    "point": {"target_dir": "Point", "label_type": "point", "ext": ".txt"},
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def build_layout(source_root: Path, output_root: Path, link_mode: str) -> Dict[str, object]:
    images_dir = output_root / "images"
    _ensure_dir(images_dir)
    for config in SOURCE_LABELS.values():
        _ensure_dir(output_root / config["target_dir"])

    samples: Dict[str, Dict[str, object]] = {}
    train_ids: List[str] = []
    stats = {config["label_type"]: 0 for config in SOURCE_LABELS.values()}

    for source_name, config in SOURCE_LABELS.items():
        src_images = source_root / source_name / "images"
        src_labels = source_root / source_name / "masks"
        target_label_dir = output_root / str(config["target_dir"])
        label_type = str(config["label_type"])
        ext = str(config["ext"])

        for src_image_path in sorted(src_images.iterdir()):
            if not src_image_path.is_file():
                continue
            image_id = src_image_path.stem
            if image_id in samples:
                raise ValueError(f"Duplicate image id across source pools: {image_id}")

            src_label_path = src_labels / f"{image_id}{ext}"
            if not src_label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_id}: {src_label_path}")

            dst_image_path = images_dir / src_image_path.name
            dst_label_path = target_label_dir / src_label_path.name
            _safe_link_or_copy(src_image_path.resolve(), dst_image_path, link_mode)
            _safe_link_or_copy(src_label_path.resolve(), dst_label_path, link_mode)

            samples[image_id] = {
                "image_id": image_id,
                "split": "train",
                "image_path": f"images/{dst_image_path.name}",
                "annotation_type": label_type,
                "annotation_path": f"{config['target_dir']}/{dst_label_path.name}",
                "source_pool": source_name,
            }
            train_ids.append(image_id)
            stats[label_type] += 1

    return {
        "dataset_name": "Polyp",
        "root": ".",
        "annotation_types": ["mask", "box", "scribble", "point"],
        "splits": {"train": train_ids, "val": [], "test": []},
        "samples": samples,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Polyp train dataset layout for SAMIX.")
    parser.add_argument("--source-root", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--manifest-name", type=str, default="train_annotation_manifest.json")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    _ensure_dir(output_root)
    manifest = build_layout(source_root=source_root, output_root=output_root, link_mode=args.link_mode)
    manifest_path = output_root / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"created={output_root}")
    print(json.dumps(manifest["stats"], ensure_ascii=True))
    print(f"train_samples={len(manifest['splits']['train'])}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
