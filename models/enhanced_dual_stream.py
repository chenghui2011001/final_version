# -*- coding: utf-8 -*-
"""
Simplified DualStreamBackbone: clean implementation aligned with jiagou.md.

Core Features:
- Simple 3x downsampling for Ribbon stream
- Basic GLA blocks for both streams
- Direct concatenation fusion
- Maintains O(n) complexity
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .enhanced_gla import GLABlock
except ImportError:  # pragma: no cover
    from enhanced_gla import GLABlock


class Conv1dT(nn.Module):
    """1D Convolution with time-first tensor convention [B,T,C]."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        y = self.conv(x.transpose(1, 2))  # -> [B,C,T]
        return y.transpose(1, 2)  # -> [B,T,C]


class DualStream(nn.Module):
    """Simplified DualStream backbone following jiagou.md specification.

    Optionally embeds a lightweight semantic head on a chosen source:
    - semantic_source: 'ribbon' | 'thread' | 'fused'
    - semantic_dim:    output dims for semantic proxy (e.g., 6 for priors)
    Returns either h or (h, aux) with aux['semantic_pred'] when enabled.
    """
    def __init__(
        self,
        d_model: int = 128,
        depth_ribbon: int = 2,
        depth_thread: int = 2,
        use_semantic_head: bool = False,
        semantic_dim: int = 6,
        semantic_source: str = 'fused',
    ):
        super().__init__()
        # Ribbon: 3x downsampling → GLA×depth → upsampling back to T
        self.down = Conv1dT(d_model, d_model, kernel_size=5, stride=3)
        self.ribbon_blocks = nn.ModuleList([
            GLABlock(d_model=d_model, heads=2, local_kernel=5, dilation=4)
            for _ in range(depth_ribbon)
        ])
        self.up = nn.Upsample(scale_factor=3, mode='linear', align_corners=False)
        self.back_to_T = Conv1dT(d_model, d_model, kernel_size=3, dilation=1)

        # Thread: original resolution GLA×depth
        self.thread_blocks = nn.ModuleList([
            GLABlock(d_model=d_model, heads=2, local_kernel=3, dilation=1)
            for _ in range(depth_thread)
        ])

        # Fusion
        self.mix = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Semantic head config
        self.use_semantic_head = use_semantic_head
        self.semantic_dim = int(semantic_dim)
        self.semantic_source = str(semantic_source)
        if self.use_semantic_head:
            self.semantic_head = nn.Linear(d_model, self.semantic_dim)
        else:
            self.semantic_head = None

    def forward(self, h0: torch.Tensor, ribbon_input: Optional[torch.Tensor] = None, thread_input: Optional[torch.Tensor] = None):
        b, t, d = h0.shape

        # Ribbon stream
        base_ribbon = ribbon_input if ribbon_input is not None else h0
        r = self.down(base_ribbon)  # [B,T,128] → [B,T/3,128]
        for blk in self.ribbon_blocks:
            r = blk(r)
        r = self.up(r.transpose(1, 2)).transpose(1, 2)  # Linear interpolation back to T

        # Ensure exact length match by interpolation or padding
        if r.size(1) != t:
            r = F.interpolate(r.transpose(1, 2), size=t, mode='linear', align_corners=False).transpose(1, 2)

        r = self.back_to_T(r)  # [B,T,128]

        # Thread stream
        thread = thread_input if thread_input is not None else h0
        for blk in self.thread_blocks:
            thread = blk(thread)  # [B,T,128]

        # Fusion
        h = torch.cat([r, thread], dim=-1)  # [B,T,256]
        h = self.mix(h)  # [B,T,128]

        # Prepare auxiliary outputs
        aux = {
            "ribbon_stream": r,      # [B,T,128] 长语义带
            "thread_stream": thread, # [B,T,128] 微音段带
            "fused_stream": h,       # [B,T,128] 融合特征
        }

        if self.use_semantic_head and self.semantic_head is not None:
            # Select semantic source
            if self.semantic_source == 'ribbon':
                sem_src = r
            elif self.semantic_source == 'thread':
                sem_src = thread
            else:
                sem_src = h  # fused
            sem_pred = self.semantic_head(sem_src)  # [B,T,semantic_dim]
            aux["semantic_pred"] = sem_pred

        return h, aux


class CompatibleDualStream(DualStream):
    """Backward compatible DualStream with simplified interface."""
    def __init__(self, d_model: int, gla_depth: int = 2, n_heads: int = 2, dropout: float = 0.0):
        super().__init__(d_model=d_model, depth_ribbon=gla_depth, depth_thread=gla_depth)


if __name__ == "__main__":
    pass
