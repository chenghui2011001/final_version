# -*- coding: utf-8 -*-
"""
Gated Linear Attention backbone used in the AETHER encoder/decoder.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .utils import phi_feature_map


class _LinearAttention(nn.Module):
    """
    Linear time attention implemented with feature map factorisation.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, d = x.shape
        h = self.n_heads

        q = self.q_proj(x).view(b, t, h, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = self.k_proj(x).view(b, t, h, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, self.head_dim).transpose(1, 2)

        qf = phi_feature_map(q)
        kf = phi_feature_map(k)

        if mask is not None:
            mask = mask.to(dtype=qf.dtype).view(b, 1, t, 1)
            qf = qf * mask
            kf = kf * mask
            v = v * mask

        kv = torch.einsum("bhtd,bhtk->bhdk", kf, v)
        z = kf.sum(dim=2)
        numerator = torch.einsum("bhtd,bhdk->bhtk", qf, kv)
        denominator = torch.einsum("bhtd,bhd->bht", qf, z).unsqueeze(-1).clamp_min_(1e-6)
        attn = numerator / denominator

        if mask is not None:
            attn = attn * mask

        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        return self.dropout(attn)


class GLABlock(nn.Module):
    """
    Single Gated Linear Attention block mixing global and local context.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        dropout: float = 0.0,
        local_kernel: int = 3,
        local_dilation: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attn = _LinearAttention(d_model, n_heads, dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        padding = ((local_kernel - 1) // 2) * local_dilation
        self.local = nn.Conv1d(d_model, d_model, kernel_size=local_kernel, padding=padding, groups=1, dilation=local_dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        attn_out = self.attn(x, attn_mask)
        local_out = self.local(x.transpose(1, 2)).transpose(1, 2)
        gate = self.gate(x)
        mixed = gate * attn_out + (1.0 - gate) * local_out
        return residual + self.dropout(self.out_proj(mixed))


class GLABackbone(nn.Module):
    """
    Stacked GLA blocks with a final LayerNorm for stable readout.
    """

    def __init__(
        self,
        d_model: int,
        depth: int = 2,
        n_heads: int = 2,
        dropout: float = 0.0,
        local_kernel: int = 3,
        local_dilation: int = 1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GLABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    local_kernel=local_kernel,
                    local_dilation=local_dilation,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.out_norm(x)


__all__ = ["GLABlock", "GLABackbone"]
