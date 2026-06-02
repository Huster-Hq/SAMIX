import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from samix.data import (
    CotrainEpisodeDataset,
    MixedSupervisionDataset,
    WarmupEpisodeDataset,
    build_nearest_support_selector,
    cotrain_collate,
    warmup_collate,
)
from samix.eval import evaluate_seg_model
from samix.framework import SAMIXCoTrainer
from samix.polyp_pvt import PolypPVT
from samix.prompts import annotation_to_prompt
from samix.sa_sam2 import build_sa_sam2
from samix.training import SAMIXTrainer
from samix.visualization import build_episode_visual, save_visual


def _make_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _log_message(message: str, log_path: Path) -> None:
    print(message)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _append_metrics(record: dict, metrics_path: Path) -> None:
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_scalars(writer: SummaryWriter, prefix: str, metrics: dict, step: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def _first_query_mask(batch: dict) -> torch.Tensor | None:
    if batch.get("query_masks") is not None:
        return batch["query_masks"][0]
    query_masks_list = batch.get("query_masks_list")
    if query_masks_list:
        return query_masks_list[0]
    return None


@torch.no_grad()
def _build_stage_visual(
    cotrainer: SAMIXCoTrainer,
    batch: dict,
    include_seg_model: bool,
) -> torch.Tensor:
    sa_device = cotrainer.sa_device
    seg_device = cotrainer.seg_device
    query_prompt = None
    if batch["query_label_types"][0] != "mask":
        query_prompt = annotation_to_prompt(batch["query_label_types"][0], batch["query_annotations"][0], sa_device)
    query_annotation = [[query_prompt]]

    sa_out = cotrainer.sa_sam2(
        support_images=batch["support_images"][0:1].to(sa_device),
        support_masks=batch["support_masks"][0:1].to(sa_device),
        query_images=batch["query_images"][0:1].to(sa_device),
        query_annotations=query_annotation,
    )
    seg_logits = None
    if include_seg_model:
        seg_logits = cotrainer.seg_model(batch["query_images"][0:1].to(seg_device))["logits"][0].detach().cpu()
    return build_episode_visual(
        support_images=batch["support_images"][0].detach().cpu(),
        support_masks=batch["support_masks"][0].detach().cpu(),
        query_image=batch["query_images"][0].detach().cpu(),
        query_label_type=batch["query_label_types"][0],
        query_annotation=batch["query_annotations"][0],
        sa_logits=sa_out.query_masks[0, 0].detach().cpu(),
        seg_logits=seg_logits,
        query_mask=_first_query_mask(batch),
    )


def _save_checkpoint(
    path: Path,
    epoch: int,
    stage: str,
    cotrainer: SAMIXCoTrainer,
    warmup_optimizer: torch.optim.Optimizer,
    joint_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    ema_model = cotrainer.ema_model.module if hasattr(cotrainer.ema_model, "module") else cotrainer.ema_model
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "samix_args": vars(args),
        "sa_sam2": cotrainer.sa_sam2.state_dict(),
        "seg_model": cotrainer.seg_model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "warmup_optimizer": warmup_optimizer.state_dict(),
        "joint_optimizer": joint_optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAMIX with warmup and co-training stages.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sam2-root", type=str, default=str(PROJECT_ROOT / "external" / "sam2"))
    parser.add_argument("--sam2-ckpt", type=str, required=True)
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--joint-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sa-device", type=str, default=None)
    parser.add_argument("--seg-device", type=str, default=None)
    parser.add_argument("--support-selection", type=str, choices=("nearest", "random"), default="nearest")
    parser.add_argument("--support-index-batch-size", type=int, default=16)
    parser.add_argument("--warmup-lr", type=float, default=5e-5)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "train_cotrain"))
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--pvt-pretrained-backbone", action="store_true")
    parser.add_argument("--test-root", type=str, default="/memory/huqiang/Polyp/images/Public_Dataset/PraNet_241217/TestDataset")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-images-every", type=int, default=200)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--test-num-workers", type=int, default=2)
    args = parser.parse_args()

    sa_device = args.sa_device or args.device
    seg_device = args.seg_device or args.device
    output_dir = _make_output_dir(Path(args.output_dir))
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard"
    visual_dir = output_dir / "visualizations"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"
    _log_message(f"Output dir: {output_dir}", log_path)
    _log_message(f"Manifest: {args.manifest}", log_path)
    _log_message(f"SAM2 config: {args.sam2_config}", log_path)
    _log_message(f"SAM2 checkpoint: {args.sam2_ckpt}", log_path)
    _log_message(f"Support selection: {args.support_selection}", log_path)
    _log_message(f"SA device: {sa_device}", log_path)
    _log_message(f"Seg device: {seg_device}", log_path)
    _log_message(f"TensorBoard dir: {tensorboard_dir}", log_path)
    _log_message(f"Test root: {args.test_root}", log_path)

    train_dataset = MixedSupervisionDataset(args.manifest, split="train", image_size=args.image_size)
    sa_sam2 = build_sa_sam2(
        sam2_repo_root=args.sam2_root,
        checkpoint_path=args.sam2_ckpt,
        config_name=args.sam2_config,
        device=sa_device,
    )
    support_selector = None
    if args.support_selection == "nearest":
        mask_indices = [idx for idx in range(len(train_dataset)) if "mask" in train_dataset[idx]["annotations"]]
        support_selector = build_nearest_support_selector(
            train_dataset,
            sa_sam2.extract_global_features,
            candidate_indices=mask_indices,
            batch_size=args.support_index_batch_size,
            device=sa_device,
        )
    warmup_dataset = WarmupEpisodeDataset(
        train_dataset,
        shots=args.shots,
        support_selector=support_selector,
        query_label_types=("mask", "box"),
    )
    cotrain_dataset = CotrainEpisodeDataset(train_dataset, shots=args.shots, support_selector=support_selector)
    warmup_loader = DataLoader(
        warmup_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=warmup_collate,
    )
    cotrain_loader = DataLoader(
        cotrain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cotrain_collate,
    )
    seg_model = PolypPVT(pretrained_backbone=args.pvt_pretrained_backbone).to(seg_device)
    cotrainer = SAMIXCoTrainer(
        sa_sam2=sa_sam2,
        seg_model=seg_model,
        sa_device=sa_device,
        seg_device=seg_device,
    )

    warmup_optimizer = torch.optim.AdamW(sa_sam2.trainable_parameters(), lr=args.warmup_lr, weight_decay=1e-4)
    joint_parameters = list(seg_model.parameters()) + list(sa_sam2.trainable_parameters())
    joint_optimizer = torch.optim.AdamW(joint_parameters, lr=args.joint_lr, weight_decay=1e-4)
    trainer = SAMIXTrainer(cotrainer, warmup_optimizer, joint_optimizer)
    warmup_step = 0
    joint_step = 0

    for epoch in range(args.warmup_epochs):
        last_metrics = None
        for batch in warmup_loader:
            output = trainer.warmup_step(batch)
            last_metrics = output.metrics
            _log_message(f"[warmup][epoch {epoch}] loss={output.metrics['warmup_loss']:.4f}", log_path)
            _append_metrics({"stage": "warmup", "epoch": epoch, **output.metrics}, metrics_path)
            _write_scalars(writer, "warmup", output.metrics, warmup_step)
            if args.log_images_every > 0 and warmup_step % args.log_images_every == 0:
                visual = _build_stage_visual(cotrainer, batch, include_seg_model=False)
                writer.add_image("warmup/episode", visual, warmup_step)
                save_visual(visual, visual_dir / "warmup" / f"step_{warmup_step:07d}.png")
            warmup_step += 1
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                checkpoints_dir / f"warmup_epoch_{epoch + 1}.pt",
                epoch=epoch + 1,
                stage="warmup",
                cotrainer=cotrainer,
                warmup_optimizer=warmup_optimizer,
                joint_optimizer=joint_optimizer,
                args=args,
            )
        if last_metrics is not None:
            _save_checkpoint(
                checkpoints_dir / "latest.pt",
                epoch=epoch + 1,
                stage="warmup",
                cotrainer=cotrainer,
                warmup_optimizer=warmup_optimizer,
                joint_optimizer=joint_optimizer,
                args=args,
            )

    for epoch in range(args.joint_epochs):
        last_metrics = None
        for batch in cotrain_loader:
            output = trainer.cotrain_step(batch)
            last_metrics = output.metrics
            _log_message(f"[joint][epoch {epoch}] total={output.metrics['total_loss']:.4f}", log_path)
            _append_metrics({"stage": "joint", "epoch": epoch, **output.metrics}, metrics_path)
            _write_scalars(writer, "joint", output.metrics, joint_step)
            if args.log_images_every > 0 and joint_step % args.log_images_every == 0:
                visual = _build_stage_visual(cotrainer, batch, include_seg_model=True)
                writer.add_image("joint/episode", visual, joint_step)
                save_visual(visual, visual_dir / "joint" / f"step_{joint_step:07d}.png")
            joint_step += 1
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            eval_metrics = evaluate_seg_model(
                cotrainer.seg_model,
                test_root=args.test_root,
                image_size=args.image_size,
                device=seg_device,
                batch_size=args.test_batch_size,
                num_workers=args.test_num_workers,
            )
            _append_metrics({"stage": "eval", "epoch": epoch + 1, "metrics": eval_metrics}, metrics_path)
            summary = ", ".join(
                f"{dataset}: Dice={scores['dice']:.4f}, IoU={scores['iou']:.4f}"
                for dataset, scores in eval_metrics.items()
            )
            _log_message(f"[eval][epoch {epoch}] {summary}", log_path)
            for dataset_name, scores in eval_metrics.items():
                writer.add_scalar(f"eval/{dataset_name}/dice", scores["dice"], epoch + 1)
                writer.add_scalar(f"eval/{dataset_name}/iou", scores["iou"], epoch + 1)
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                checkpoints_dir / f"joint_epoch_{epoch + 1}.pt",
                epoch=epoch + 1,
                stage="joint",
                cotrainer=cotrainer,
                warmup_optimizer=warmup_optimizer,
                joint_optimizer=joint_optimizer,
                args=args,
            )
        if last_metrics is not None:
            _save_checkpoint(
                checkpoints_dir / "latest.pt",
                epoch=epoch + 1,
                stage="joint",
                cotrainer=cotrainer,
                warmup_optimizer=warmup_optimizer,
                joint_optimizer=joint_optimizer,
                args=args,
            )

    _save_checkpoint(
        checkpoints_dir / "final.pt",
        epoch=args.joint_epochs,
        stage="joint",
        cotrainer=cotrainer,
        warmup_optimizer=warmup_optimizer,
        joint_optimizer=joint_optimizer,
        args=args,
    )
    writer.close()
    _log_message(f"Training finished. Artifacts saved to {output_dir}", log_path)


if __name__ == "__main__":
    main()
