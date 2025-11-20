#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waveform loss utilities for the staged pipeline.

Only the portions that are actively used by the new training scripts are
ported here; the legacy trainer still provides the full reference logic.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from ..fargan_losses import (
    compute_fargan_comprehensive_loss,
    compute_fargan_original_style_loss,
    compute_fargan_training_loss,
)


def fargan_wave_losses(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    period: torch.Tensor,
    *,
    comprehensive_weight: float = 0.0,
    original_weight: float = 0.0,
    train_weights: Dict[str, float] | None = None,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the composite waveform loss used during FARGAN stages.

    The baseline training objective is ``compute_fargan_training_loss``.
    Optional comprehensive/original penalties can be blended in with the
    supplied weights.
    """
    pred_audio = pred_audio.squeeze(1) if pred_audio.dim() == 3 else pred_audio
    target_audio = target_audio.squeeze(1) if target_audio.dim() == 3 else target_audio

    min_len = min(pred_audio.size(-1), target_audio.size(-1))
    pred_audio = pred_audio[..., :min_len]
    target_audio = target_audio[..., :min_len]

    primary_loss, breakdown = compute_fargan_training_loss(
        pred_audio, target_audio, period, device=device, weights=train_weights
    )
    total = primary_loss
    # 🔧 修复: 不要使用.item()断开梯度，只在需要时转为float
    details = {"primary": primary_loss}  # 保持tensor保留梯度

    if comprehensive_weight > 0.0:
        comp_loss, comp_terms = compute_fargan_comprehensive_loss(
            pred_audio, target_audio, device=device
        )
        total = total + comprehensive_weight * comp_loss
        details["comprehensive"] = comp_loss  # 保持tensor
        for key, value in comp_terms.items():
            details[f"comp_{key}"] = value  # 保持tensor

    if original_weight > 0.0:
        orig_loss, orig_terms = compute_fargan_original_style_loss(
            pred_audio, target_audio, device=device
        )
        total = total + original_weight * orig_loss
        details["original"] = orig_loss  # 保持tensor
        for key, value in orig_terms.items():
            details[f"orig_{key}"] = value  # 保持tensor

    for key, value in breakdown.items():
        details[f"primary_{key}"] = value  # 保持tensor

    return total, details


__all__ = ["fargan_wave_losses"]
