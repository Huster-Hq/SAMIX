import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from samix.data import CotrainEpisodeDataset, MixedSupervisionDataset, WarmupEpisodeDataset, cotrain_collate, warmup_collate
from samix.framework import SAMIXCoTrainer
from samix.polyp_pvt import PolypPVT
from samix.sa_sam2 import build_sa_sam2
from samix.training import SAMIXTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAMIX with warmup and co-training stages.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--sam2-root", type=str, required=True)
    parser.add_argument("--sam2-ckpt", type=str, required=True)
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_dataset = MixedSupervisionDataset(args.manifest, split="train", image_size=args.image_size)
    warmup_dataset = WarmupEpisodeDataset(train_dataset, shots=args.shots)
    cotrain_dataset = CotrainEpisodeDataset(train_dataset, shots=args.shots)
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

    sa_sam2 = build_sa_sam2(
        sam2_repo_root=args.sam2_root,
        checkpoint_path=args.sam2_ckpt,
        config_name=args.sam2_config,
        device=args.device,
    )
    seg_model = PolypPVT().to(device)
    cotrainer = SAMIXCoTrainer(sa_sam2=sa_sam2, seg_model=seg_model).to(device)

    warmup_optimizer = torch.optim.AdamW(sa_sam2.trainable_parameters(), lr=1e-4, weight_decay=1e-4)
    joint_parameters = list(seg_model.parameters()) + list(sa_sam2.trainable_parameters())
    joint_optimizer = torch.optim.AdamW(joint_parameters, lr=1e-4, weight_decay=1e-4)
    trainer = SAMIXTrainer(cotrainer, warmup_optimizer, joint_optimizer)

    for epoch in range(args.warmup_epochs):
        for batch in warmup_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = trainer.warmup_step(batch)
            print(f"[warmup][epoch {epoch}] loss={output.metrics['warmup_loss']:.4f}")

    for epoch in range(args.joint_epochs):
        for batch in cotrain_loader:
            batch["support_images"] = batch["support_images"].to(device)
            batch["support_masks"] = batch["support_masks"].to(device)
            batch["query_images"] = batch["query_images"].to(device)
            if batch["query_masks"] is not None:
                batch["query_masks"] = batch["query_masks"].to(device)
            output = trainer.cotrain_step(batch)
            print(f"[joint][epoch {epoch}] total={output.metrics['total_loss']:.4f}")


if __name__ == "__main__":
    main()
