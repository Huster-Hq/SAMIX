import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_FOLDERS = {
    "mask": ("masks", [".png"]),
    "box": ("boxes", [".json"]),
    "scribble": ("scribbles", [".json"]),
    "point": ("points", [".json"]),
    "class": ("classes", [".json"]),
    "unlabeled": ("unlabeled", [".json"]),
}


def parse_split_ids(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_split_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_split_membership(image_ids: list[str], args: argparse.Namespace) -> dict[str, str]:
    membership: dict[str, str] = {}

    for image_id in load_split_file(args.train_file) + parse_split_ids(args.train_ids):
        membership[image_id] = "train"
    for image_id in load_split_file(args.val_file) + parse_split_ids(args.val_ids):
        membership[image_id] = "val"
    for image_id in load_split_file(args.test_file) + parse_split_ids(args.test_ids):
        membership[image_id] = "test"

    default_split = args.default_split
    for image_id in image_ids:
        membership.setdefault(image_id, default_split)
    return membership


def find_label_file(dataset_root: Path, folder_name: str, stem: str, exts: list[str]) -> str | None:
    folder = dataset_root / folder_name
    if not folder.exists():
        return None
    for ext in exts:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate.relative_to(dataset_root).as_posix()
    return None


def infer_image_size(_image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None

    with Image.open(_image_path) as image:
        width, height = image.size
    return width, height


def build_manifest(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_root).resolve()
    image_dir = dataset_root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    image_ids = [path.stem for path in image_paths]
    split_membership = resolve_split_membership(image_ids, args)

    manifest = {
        "dataset_name": args.dataset_name,
        "root": ".",
        "annotation_types": list(LABEL_FOLDERS.keys()),
        "splits": {"train": [], "val": [], "test": []},
        "images": {},
    }

    for image_path in image_paths:
        image_id = image_path.stem
        split = split_membership[image_id]
        width, height = infer_image_size(image_path)

        labels = {}
        for label_type, (folder_name, exts) in LABEL_FOLDERS.items():
            rel_path = find_label_file(dataset_root, folder_name, image_id, exts)
            if rel_path is not None:
                labels[label_type] = rel_path

        manifest["splits"][split].append(image_id)
        manifest["images"][image_id] = {
            "file_name": image_path.relative_to(dataset_root).as_posix(),
            "split": split,
            "width": width,
            "height": height,
            "labels": labels,
        }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SAMIX mixed-supervision dataset manifest.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="samix_dataset")
    parser.add_argument("--output", type=str, default="splits.json")
    parser.add_argument("--default-split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-ids", type=str, default="")
    parser.add_argument("--val-ids", type=str, default="")
    parser.add_argument("--test-ids", type=str, default="")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_manifest(args)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()
