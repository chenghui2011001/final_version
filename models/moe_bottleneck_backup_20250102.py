# -*- coding: utf-8 -*-
"""
Micro-MoE bottleneck (compat wrapper).

This module now provides a thin compatibility wrapper that forwards to the
Specialized dual-routing MoE implemented in `enhanced_moe.CompatibleMicroMoE`.
The original TinyExpert-based implementation is kept for reference but the
exported `MicroMoE` class uses the enhanced router and experts for better
specialization and interpretability, with Top-2 sparse activation.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyExpert(nn.Module):
    """
    Minimalist expert that keeps parameters light-weight yet expressive.
    Supports scale factor for sparsity optimization.
    """

    def __init__(self, d_model: int, scale_factor: float = 1.0) -> None:
        super().__init__()
        hidden_dim = max(32, int(d_model * 2 * scale_factor))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MicroMoE(nn.Module):
    """Compatibility wrapper delegating to enhanced_moe.CompatibleMicroMoE.

    Preserves the original call pattern: forward(h, router_in) -> (h_out, aux).
    The last 10 dims of router_in are interpreted as CSI when available.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 4,
        top_k: int = 2,
        d_router_in: Optional[int] = None,  # kept for API compatibility (unused)
        balance_weight: float = 0.5,
        expert_dropout: float = 0.1,
        d_csi: int = 10,
        router_use_csi: bool = True,
        use_token_level: bool = True,
    ) -> None:
        super().__init__()
        try:
            from .enhanced_moe import CompatibleMicroMoE as _Compat
        except Exception:
            from enhanced_moe import CompatibleMicroMoE as _Compat

        self._compat = _Compat(
            d_model=d_model,
            n_experts=n_experts,
            top_k=top_k,
            d_csi=d_csi,
            expert_dropout=expert_dropout,
            balance_weight=balance_weight,
            router_use_csi=router_use_csi,
        )

    def forward(self, h: torch.Tensor, router_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Extract CSI from the tail for compatibility
        b = h.size(0)
        csi_dim = self._compat.specialized_moe.d_csi
        if router_in.size(-1) >= csi_dim:
            csi_vec = router_in[:, -csi_dim:]
        else:
            csi_vec = None

        out = self._compat.specialized_moe(h, csi_vec)
        aux = self._compat.specialized_moe.get_aux_losses(h, csi_vec)

        # Normalize keys for training pipeline
        if isinstance(aux, dict) and 'balance_loss' in aux:
            aux['moe_balance_loss'] = aux['balance_loss']
        return out, aux

    def get_expert_utilization(self):
        return self._compat.get_expert_utilization()


__all__ = ["MicroMoE", "TinyExpert"]
