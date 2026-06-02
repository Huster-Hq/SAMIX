from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from .framework import SAMIXCoTrainer
from .losses import (
    box_supervision_loss,
    class_presence_loss,
    consistency_loss,
    mask_supervision_loss,
    point_supervision_loss,
    pseudo_label_loss,
    scribble_supervision_loss,
)
from .prompts import annotation_to_prompt


@dataclass
class WarmupOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


@dataclass
class CotrainOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


class SAMIXTrainer:
    """
    Two-stage co-training trainer:
    1. warmup: only SA-SAM2, following the SANSA-style few-shot episode training
    2. joint phase: train Seg-Model + EMA + semantic adapter together
    """

    def __init__(
        self,
        model: SAMIXCoTrainer,
        warmup_optimizer: torch.optim.Optimizer,
        joint_optimizer: torch.optim.Optimizer,
        ema_decay: float = 0.999,
        pseudo_weight: float = 1.0,
        consistency_weight: float = 0.2,
        weak_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.warmup_optimizer = warmup_optimizer
        self.joint_optimizer = joint_optimizer
        self.ema_decay = ema_decay
        self.pseudo_weight = pseudo_weight
        self.consistency_weight = consistency_weight
        self.weak_weight = weak_weight

    def warmup_step(self, batch: Dict[str, torch.Tensor | List]) -> WarmupOutput:
        self.model.sa_sam2.train()
        self.warmup_optimizer.zero_grad(set_to_none=True)
        sa_device = self.model.sa_device
        batch["support_images"] = batch["support_images"].to(sa_device)
        batch["support_masks"] = batch["support_masks"].to(sa_device)
        batch["query_images"] = batch["query_images"].to(sa_device)
        if batch["query_masks"] is not None:
            batch["query_masks"] = batch["query_masks"].to(sa_device)
        query_annotations = [
            [annotation_to_prompt(label_type, annotation, sa_device) if label_type != "mask" else None]
            for label_type, annotation in zip(batch["query_label_types"], batch["query_annotations"])
        ]
        sa_out = self.model.sa_sam2(
            support_images=batch["support_images"],
            support_masks=batch["support_masks"],
            query_images=batch["query_images"],
            query_annotations=query_annotations,
        )
        logits = sa_out.query_masks[:, 0]
        loss = logits.new_tensor(0.0)
        metrics: Dict[str, float] = {}

        dense_indices = [idx for idx, label_type in enumerate(batch["query_label_types"]) if label_type == "mask"]
        if dense_indices:
            dense_logits = logits[dense_indices]
            if batch["query_masks"] is not None and len(dense_indices) == logits.shape[0]:
                dense_target = batch["query_masks"]
            else:
                dense_target = torch.stack(
                    [batch["query_masks_list"][idx] for idx in dense_indices],
                    dim=0,
                ).to(sa_device)
            mask_loss = mask_supervision_loss(dense_logits, dense_target)
            loss = loss + mask_loss
            metrics["warmup_mask_loss"] = float(mask_loss.detach().item())

        weak = self._weak_loss(
            logits,
            batch["query_label_types"],
            batch["query_annotations"],
            image_size=logits.shape[-1],
        )
        loss = loss + weak
        metrics["warmup_weak_loss"] = float(weak.detach().item())
        loss.backward()
        self.warmup_optimizer.step()
        metrics["warmup_loss"] = float(loss.detach().item())
        return WarmupOutput(loss=loss.detach(), metrics=metrics)

    def _weak_loss(self, logits: torch.Tensor, label_types: List[str], annotations: List[Optional[Dict]], image_size: int) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        count = 0
        for label_type in set(label_types):
            indices = [idx for idx, t in enumerate(label_types) if t == label_type]
            if not indices:
                continue
            subset_logits = logits[indices]
            subset_annotations = [annotations[idx] for idx in indices]
            if label_type == "point":
                total = total + point_supervision_loss(subset_logits, [ann["points"] for ann in subset_annotations], image_size=image_size)
                count += 1
            elif label_type == "scribble":
                total = total + scribble_supervision_loss(subset_logits, [ann["scribbles"] for ann in subset_annotations], image_size=image_size)
                count += 1
            elif label_type == "box":
                total = total + box_supervision_loss(subset_logits, [ann["boxes"] for ann in subset_annotations], image_size=image_size)
                count += 1
            elif label_type == "class":
                total = total + class_presence_loss(subset_logits, [ann.get("classes", []) for ann in subset_annotations])
                count += 1
        return total / max(1, count)

    def cotrain_step(self, batch: Dict[str, torch.Tensor | List]) -> CotrainOutput:
        self.model.train()
        self.joint_optimizer.zero_grad(set_to_none=True)

        sa_device = self.model.sa_device
        seg_device = self.model.seg_device
        support_images = batch["support_images"].to(sa_device)
        support_masks = batch["support_masks"].to(sa_device)
        query_images_sa = batch["query_images"].to(sa_device)
        query_images_seg = batch["query_images"].to(seg_device)
        dense_target_sa = batch["query_masks"].to(sa_device) if batch["query_masks"] is not None else None
        dense_target_seg = batch["query_masks"].to(seg_device) if batch["query_masks"] is not None else None
        query_annotations = [
            [annotation_to_prompt(label_type, annotation, sa_device)]
            for label_type, annotation in zip(batch["query_label_types"], batch["query_annotations"])
        ]

        sa_out = self.model.sa_sam2(
            support_images=support_images,
            support_masks=support_masks,
            query_images=query_images_sa,
            query_annotations=query_annotations,
        )
        seg_out = self.model.seg_model(query_images_seg)
        with torch.no_grad():
            ema_out = self.model.ema_model(query_images_seg)

        seg_logits = seg_out["logits"]
        sa_logits = sa_out.query_masks[:, 0]
        ema_logits = ema_out["logits"]
        loss = seg_logits.new_tensor(0.0)
        metrics: Dict[str, float] = {}

        if dense_target_sa is not None and dense_target_seg is not None:
            sa_sup = mask_supervision_loss(sa_logits, dense_target_sa)
            seg_sup = mask_supervision_loss(seg_logits, dense_target_seg)
            loss = loss + sa_sup.to(seg_device) + seg_sup
            metrics["sa_supervised"] = float(sa_sup.detach().item())
            metrics["seg_supervised"] = float(seg_sup.detach().item())

        pseudo = pseudo_label_loss(seg_logits, torch.sigmoid(sa_logits.detach()).to(seg_device), threshold=0.5)
        consistency = consistency_loss(seg_logits, ema_logits.detach())
        weak = self._weak_loss(
            seg_logits,
            batch["query_label_types"],
            batch["query_annotations"],
            image_size=seg_logits.shape[-1],
        )
        loss = loss + self.pseudo_weight * pseudo + self.consistency_weight * consistency + self.weak_weight * weak

        loss.backward()
        self.joint_optimizer.step()
        self.model.ema_model.update(self.model.seg_model)

        metrics.update(
            {
                "pseudo_loss": float(pseudo.detach().item()),
                "consistency_loss": float(consistency.detach().item()),
                "weak_loss": float(weak.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
        )
        return CotrainOutput(loss=loss.detach(), metrics=metrics)
