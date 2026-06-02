from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from .model_utils import BackboneOutput, DecoderOutput, SASAM2Output


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def _resolve_sam2_repo_root(sam2_repo_root: Optional[str]) -> Optional[Path]:
    if sam2_repo_root:
        candidate = Path(sam2_repo_root).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"SAM2 repo root does not exist: {candidate}")
        return candidate

    project_root = Path(__file__).resolve().parents[1]
    bundled = project_root / "external" / "sam2"
    if bundled.exists():
        return bundled.resolve()
    return None


def _ensure_sam2_importable(sam2_repo_root: Optional[str]) -> Path | None:
    resolved_root = _resolve_sam2_repo_root(sam2_repo_root)
    if resolved_root is not None:
        resolved_root_str = str(resolved_root)
        if resolved_root_str not in sys.path:
            sys.path.insert(0, resolved_root_str)
    return resolved_root


class SASAM2FewShotSegmentor(nn.Module):
    """
    SAM2-based in-context segmentor with SANSA-style semantic adapters in the image encoder.
    """

    def __init__(self, sam: nn.Module, freeze_non_adapter: bool = True) -> None:
        super().__init__()
        self.sam = sam
        if freeze_non_adapter:
            self.freeze_non_adapter_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_non_adapter_parameters(self) -> None:
        for name, parameter in self.sam.named_parameters():
            parameter.requires_grad = "adapter" in name

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def preprocess_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        orig_size = [tuple(img.shape[-2:]) for img in images]
        if images.max() > 1.0:
            images = images / 255.0
        images = _imagenet_normalize(images)
        if images.shape[-1] != self.sam.image_size or images.shape[-2] != self.sam.image_size:
            images = F.interpolate(
                images,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return images, orig_size

    def _forward_backbone(self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]) -> BackboneOutput:
        backbone_out = self.sam.forward_image(samples)
        _, vision_feats, vision_pos, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        return BackboneOutput(orig_size, backbone_out, vision_feats, vision_pos, feat_sizes)

    @torch.no_grad()
    def extract_global_features(self, images: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        samples, orig_size = self.preprocess_batch(images)
        backbone_out = self._forward_backbone(samples, orig_size)
        features = backbone_out.vision_feats[-1].mean(dim=0)
        if was_training:
            self.train()
        return F.normalize(features, dim=-1)

    def _split_prompt(
        self,
        prompt: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if prompt is None:
            return None, None
        if prompt["prompt_type"] == "mask":
            mask_prompt = prompt["prompt"].float()
            if mask_prompt.shape[-2:] != (self.sam.image_size, self.sam.image_size):
                mask_prompt = F.interpolate(
                    mask_prompt,
                    size=(self.sam.image_size, self.sam.image_size),
                    mode="nearest",
                )
            return None, mask_prompt
        return prompt["prompt"], None

    def forward_episode(
        self,
        support_images: torch.Tensor,
        support_annotations: List[List[Optional[Dict[str, Any]]]],
        query_images: torch.Tensor,
        query_annotations: Optional[List[List[Optional[Dict[str, Any]]]]] = None,
    ) -> SASAM2Output:
        batch_size, num_support = support_images.shape[:2]
        num_query = query_images.shape[1]
        clip = torch.cat([support_images, query_images], dim=1)
        flat_clip = clip.flatten(0, 1)
        flat_clip, orig_size = self.preprocess_batch(flat_clip)
        backbone_out = self._forward_backbone(flat_clip, orig_size)

        support_masks: List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        memory_banks: List[Dict[int, Dict[str, torch.Tensor]]] = []

        for batch_index in range(batch_size):
            output_dict: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            for frame_index in range(num_support + num_query):
                absolute_idx = batch_index * (num_support + num_query) + frame_index
                current_feats = backbone_out.get_current_feats(absolute_idx)
                current_pos = backbone_out.get_current_pos_embeds(absolute_idx)
                if frame_index < num_support:
                    prompt = support_annotations[batch_index][frame_index]
                    point_inputs, mask_inputs = self._split_prompt(prompt)
                    current_out = self.sam.track_step(
                        frame_idx=frame_index,
                        is_init_cond_frame=True,
                        current_vision_feats=current_feats,
                        current_vision_pos_embeds=current_pos,
                        feat_sizes=backbone_out.feat_sizes,
                        point_inputs=point_inputs,
                        mask_inputs=mask_inputs,
                        output_dict=output_dict,
                        num_frames=num_support + num_query,
                        track_in_reverse=False,
                        run_mem_encoder=True,
                        prev_sam_mask_logits=None,
                    )
                    output_dict["cond_frame_outputs"][frame_index] = current_out
                    support_masks.append(current_out["pred_masks_high_res"])
                else:
                    prompt = None
                    if query_annotations is not None:
                        prompt = query_annotations[batch_index][frame_index - num_support]
                    point_inputs, mask_inputs = self._split_prompt(prompt)
                    current_out = self.sam.track_step(
                        frame_idx=frame_index,
                        is_init_cond_frame=False,
                        current_vision_feats=current_feats,
                        current_vision_pos_embeds=current_pos,
                        feat_sizes=backbone_out.feat_sizes,
                        point_inputs=point_inputs,
                        mask_inputs=mask_inputs,
                        output_dict=output_dict,
                        num_frames=num_support + num_query,
                        track_in_reverse=False,
                        run_mem_encoder=True,
                        prev_sam_mask_logits=None,
                    )
                    output_dict["non_cond_frame_outputs"][frame_index] = current_out
                    query_masks.append(current_out["pred_masks_high_res"])
            merged_memory_bank = output_dict["cond_frame_outputs"].copy()
            merged_memory_bank.update(output_dict["non_cond_frame_outputs"])
            memory_banks.append(merged_memory_bank)

        support_masks_tensor = torch.cat(support_masks, dim=0).view(batch_size, num_support, 1, self.sam.image_size, self.sam.image_size)
        query_masks_tensor = torch.cat(query_masks, dim=0).view(batch_size, num_query, 1, self.sam.image_size, self.sam.image_size)
        return SASAM2Output(
            support_masks=support_masks_tensor,
            query_masks=query_masks_tensor,
            support_memory_bank=memory_banks,
            aux={"image_size": self.sam.image_size},
        )

    def forward(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        query_annotations: Optional[List[List[Optional[Dict[str, Any]]]]] = None,
    ) -> SASAM2Output:
        device = support_images.device
        support_annotations: List[List[Optional[Dict[str, Any]]]] = []
        for batch_index in range(support_masks.shape[0]):
            prompt_list = []
            for shot_index in range(support_masks.shape[1]):
                prompt_list.append({"prompt_type": "mask", "prompt": support_masks[batch_index, shot_index : shot_index + 1].to(device)})
            support_annotations.append(prompt_list)
        query_images = query_images.unsqueeze(1) if query_images.dim() == 4 else query_images
        return self.forward_episode(support_images, support_annotations, query_images, query_annotations=query_annotations)


def build_sa_sam2(
    sam2_repo_root: Optional[str],
    checkpoint_path: str,
    config_name: str = "sam2_hiera_l.yaml",
    adaptformer_stages: Sequence[int] = (2, 3),
    adapt_dim: float = 0.3,
    adapter_scale: float = 0.1,
    device: str = "cuda",
) -> SASAM2FewShotSegmentor:
    # Prefer the bundled or explicitly provided SAM2 repo so the project can be
    # reproduced from one checkout. Fall back to any environment-installed
    # `sam2` package if no repo root is supplied.
    _ensure_sam2_importable(sam2_repo_root)

    # Import the official SAM2 package first so its config module is initialized
    # in the same way as upstream `build_sam2`.
    import sam2  # noqa: F401

    stage_list = ",".join(str(stage) for stage in adaptformer_stages)
    overrides = [
        "++model.image_encoder.trunk._target_=samix.sa_hiera.AdapterHiera",
        f"++model.image_encoder.trunk.adaptformer_stages=[{stage_list}]",
        f"++model.image_encoder.trunk.adapt_dim={adapt_dim}",
        f"++model.image_encoder.trunk.adapter_scale={adapter_scale}",
        "++model.pred_obj_scores=false",
        "++model.pred_obj_scores_mlp=false",
        "++model.fixed_no_obj_ptr=false",
    ]

    cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    sam.load_state_dict(state_dict, strict=False)
    model = SASAM2FewShotSegmentor(sam=sam).to(device)
    return model
