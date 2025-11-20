# -*- coding: utf-8 -*-
"""
Simplified GLA: basic gated linear attention aligned with jiagou.md.

Core Features:
- Linear attention with O(n) complexity
- Local convolution for detail preservation
- Gated fusion between global and local paths
- Simple multi-head structure without RoPE
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLABlock(nn.Module):
    """Basic GLA block following jiagou.md specification."""
    def __init__(self, d_model: int, heads: int = 2, local_kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        # Pre-normalization
        self.norm = nn.LayerNorm(d_model)

        # Linear attention components
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Local convolution path
        self.local_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=local_kernel,
            padding=(local_kernel - 1) // 2 * dilation,
            dilation=dilation,
            groups=d_model  # Depthwise convolution for efficiency
        )

        # Gated fusion
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute linear attention with ELU feature mapping."""
        # Apply ELU + 1 feature mapping to ensure positivity
        phi_q = F.elu(q) + 1.0  # [B, H, T, D_h]
        phi_k = F.elu(k) + 1.0  # [B, H, T, D_h]

        # Cumulative attention: kv = sum_t( phi(k_t) ⊗ v_t )
        kv = torch.einsum('bhtd,bhte->bhde', phi_k, v)  # [B, H, D_h, D_h]

        # Attention output: y = sum_t( phi(q_t) @ kv )
        y_att = torch.einsum('bhtd,bhde->bhte', phi_q, kv)  # [B, H, T, D_h]

        return y_att

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape

        # Pre-normalization
        x_norm = self.norm(x)

        # Linear attention path
        qkv = self.qkv(x_norm)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, T, D]

        # Reshape for multi-head attention
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_h]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
            k = k.masked_fill(~mask, 0.0)
            v = v.masked_fill(~mask, 0.0)

        # Linear attention
        y_att = self.linear_attention(q, k, v)  # [B, H, T, D_h]

        # Merge heads
        y_att = y_att.transpose(1, 2).contiguous().view(b, t, d)  # [B, T, D]

        # Local convolution path
        y_local = self.local_conv(x_norm.transpose(1, 2)).transpose(1, 2)  # [B, T, D]

        # Gated fusion: g * global + (1 - g) * local
        g = self.gate_proj(x)  # [B, T, D]
        y = g * y_att + (1 - g) * y_local

        # Output projection with residual connection
        out = x + self.dropout(self.out_proj(y))

        return out


class GLABackbone(nn.Module):
    """Stack of GLA blocks forming a backbone."""
    def __init__(self, d_model: int, depth: int, heads: int = 2, local_kernel: int = 3, local_dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GLABlock(
                d_model=d_model,
                heads=heads,
                local_kernel=local_kernel,
                dilation=local_dilation,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class CompatibleGLABlock(GLABlock):
    """Backward compatible GLA block."""
    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1, **kwargs):
        super().__init__(d_model=d_model, heads=n_heads, dropout=dropout, **kwargs)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, mask)


if __name__ == "__main__":
    pass