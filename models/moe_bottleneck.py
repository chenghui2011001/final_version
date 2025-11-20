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
        # 直流通路相关参数
        enable_direct_pathway: bool = False,
        initial_bypass_weight: float = 0.8,
        adaptive_threshold: float = 0.05,
        pathway_warmup_steps: int = 1500,
    ) -> None:
        super().__init__()
        self.enable_direct_pathway = enable_direct_pathway

        if enable_direct_pathway:
            # 使用带直流通路的增强版本
            try:
                from .enhanced_moe import EnhancedMicroMoEWithBypass as _Enhanced
            except Exception:
                from enhanced_moe import EnhancedMicroMoEWithBypass as _Enhanced

            self._compat = _Enhanced(
                D=d_model,
                n_experts=n_experts,
                topk=top_k,
                d_csi=d_csi,
                expert_dropout=expert_dropout,
                balance_weight=balance_weight,
                router_use_csi=router_use_csi,
                use_token_level=use_token_level,
                enable_direct_pathway=enable_direct_pathway,
                initial_bypass_weight=initial_bypass_weight,
                adaptive_threshold=adaptive_threshold,
                pathway_warmup_steps=pathway_warmup_steps,
            )
        else:
            # 使用标准版本
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

    def forward(self, h: torch.Tensor, router_in: torch.Tensor, training_step: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Extract CSI from the tail for compatibility
        b = h.size(0)

        if self.enable_direct_pathway:
            # 直流通路模式：调用EnhancedMicroMoEWithBypass
            csi_dim = self._compat.d_model if hasattr(self._compat, 'd_model') else 10
            if router_in.size(-1) >= csi_dim:
                csi_vec = router_in[:, -csi_dim:]
            else:
                csi_vec = None

            # 调用增强版本的forward，传递training_step
            out = self._compat(h, csi_vec, training_step=training_step)
            aux = self._compat.get_aux_losses(h, csi_vec)
        else:
            # 标准模式：调用CompatibleMicroMoE
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

    def get_performance_stats(self):
        """获取性能统计信息（仅直流通路模式）"""
        if self.enable_direct_pathway and hasattr(self._compat, 'get_performance_stats'):
            return self._compat.get_performance_stats()
        return {}

    def get_separated_outputs(self):
        """获取分离的输出用于损失计算（仅直流通路模式）"""
        if self.enable_direct_pathway and hasattr(self._compat, 'get_separated_outputs'):
            return self._compat.get_separated_outputs()
        return None, None

    def update_performance_ema(self, direct_loss: float, expert_loss: float):
        """更新性能EMA（仅直流通路模式）"""
        if self.enable_direct_pathway and hasattr(self._compat, 'update_performance_ema'):
            self._compat.update_performance_ema(direct_loss, expert_loss)


__all__ = ["MicroMoE", "TinyExpert"]
