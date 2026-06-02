from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PolypTestDataset(Dataset):
    def __init__(self, dataset_root: str, image_size: int = 512) -> None:
        self.root = Path(dataset_root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.image_size = image_size
        self.image_paths = sorted(p for p in self.image_dir.iterdir() if p.is_file() and not p.name.startswith("."))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Tuple[int, int]]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        orig_hw = (image.size[1], image.size[0])
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        mask = Image.open(mask_path).convert("L")
        mask_tensor = TF.to_tensor(mask)
        mask_tensor = (mask_tensor > 0).float()

        return {
            "image_id": image_path.stem,
            "image": image_tensor,
            "mask": mask_tensor,
            "orig_hw": orig_hw,
        }


def compute_dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    target = (target > 0.5).float()

    intersection = (preds * target).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection

    dice = ((2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return {"dice": dice, "iou": iou}


@torch.no_grad()
def evaluate_seg_model(
    seg_model: torch.nn.Module,
    test_root: str,
    image_size: int,
    device: str,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Dict[str, Dict[str, float]]:
    seg_model_was_training = seg_model.training
    seg_model.eval()

    root = Path(test_root)
    dataset_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "images").exists() and (path / "masks").exists()
    )
    metrics: Dict[str, Dict[str, float]] = {}
    dataset_scores: List[Tuple[float, float]] = []

    for dataset_dir in dataset_dirs:
        dataset = PolypTestDataset(str(dataset_dir), image_size=image_size)
        dice_scores: List[float] = []
        iou_scores: List[float] = []
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            logits = seg_model(image)["logits"]
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            score = compute_dice_iou_from_logits(logits, mask)
            dice_scores.append(score["dice"])
            iou_scores.append(score["iou"])
        dataset_dice = float(sum(dice_scores) / max(1, len(dice_scores)))
        dataset_iou = float(sum(iou_scores) / max(1, len(iou_scores)))
        metrics[dataset_dir.name] = {"dice": dataset_dice, "iou": dataset_iou}
        dataset_scores.append((dataset_dice, dataset_iou))

    if dataset_scores:
        mean_dice = float(sum(score[0] for score in dataset_scores) / len(dataset_scores))
        mean_iou = float(sum(score[1] for score in dataset_scores) / len(dataset_scores))
    else:
        mean_dice = 0.0
        mean_iou = 0.0
    metrics["mean"] = {"dice": mean_dice, "iou": mean_iou}

    if seg_model_was_training:
        seg_model.train()
    return metrics
