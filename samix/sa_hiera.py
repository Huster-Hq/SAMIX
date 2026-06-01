from __future__ import annotations

from functools import partial
from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.utils import PatchEmbed, window_partition, window_unpartition
from sam2.modeling.sam2_utils import DropPath, MLP

from .adapters import SemanticAdapter


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module | None = None) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm is not None:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        qkv = self.qkv(x).reshape(batch, height * width, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool is not None:
            q = do_pool(q.reshape(batch, height, width, -1), self.q_pool)
            height, width = q.shape[1:3]
            q = q.reshape(batch, height * width, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(batch, height, width, -1)
        return self.proj(x)


class AdapterMultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] | None = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        adaptformer: bool = False,
        adapt_dim: float = 0.3,
        adapter_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool = None
        self.q_stride = q_stride
        if q_stride is not None:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)
        self.proj = nn.Linear(dim, dim_out) if dim != dim_out else None

        self.adapter = None
        if adaptformer:
            bottleneck = max(1, int(dim_out * adapt_dim))
            self.adapter = SemanticAdapter(dim_out, bottleneck=bottleneck, scale=adapter_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.proj is not None:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            height, width = x.shape[1:3]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride is not None:
            window_size = self.window_size // self.q_stride[0]
            height, width = shortcut.shape[1:3]
            pad_h = (window_size - height % window_size) % window_size
            pad_w = (window_size - width % window_size) % window_size
            pad_hw = (height + pad_h, width + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (height, width))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.adapter is not None:
            x = x + self.adapter(x)
        return x


class AdapterHiera(nn.Module):
    """
    SAM2 Hiera trunk with SANSA-style semantic adapters injected into chosen stages.
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        weights_path: str | None = None,
        return_interm_layers: bool = True,
        adaptformer_stages: Sequence[int] = (2, 3),
        adapt_dim: float = 0.3,
        adapter_scale: float = 0.1,
    ) -> None:
        super().__init__()
        del weights_path
        assert len(stages) == len(window_spec)

        self.window_spec = window_spec
        self.q_stride = q_stride
        self.global_att_blocks = global_att_blocks
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))

        depth = sum(stages)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        current_stage = 1
        adapter_stage_set = set(adaptformer_stages)
        for index in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[current_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if index in self.global_att_blocks else window_size
            if index - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                current_stage += 1

            self.blocks.append(
                AdapterMultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    drop_path=dpr[index],
                    q_stride=self.q_stride if index in self.q_pool_blocks else None,
                    window_size=window_size,
                    adaptformer=(current_stage - 1) in adapter_stage_set,
                    adapt_dim=adapt_dim,
                    adapter_scale=adapter_scale,
                )
            )
            embed_dim = dim_out

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        height, width = hw
        pos_embed = F.interpolate(self.pos_embed, size=(height, width), mode="bicubic")
        tiled = self.pos_embed_window.tile([x // y for x, y in zip(pos_embed.shape, self.pos_embed_window.shape)])
        return (pos_embed + tiled).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs: List[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == self.stage_ends[-1] or (index in self.stage_ends and self.return_interm_layers):
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs
