# -*- coding: utf-8 -*-
"""
Lightweight placeholder metrics for quick experiments.
"""

from __future__ import annotations

import torch


def pseudo_pesq(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Cheap proxy metric mirroring PESQ trends.
    """
    diff = torch.nn.functional.l1_loss(y_hat, y)
    return float(torch.clamp(1.0 - diff, 0.0, 1.0).item())


def pseudo_stoi(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """
    Cheap proxy metric mirroring STOI behaviour.
    """
    diff = torch.nn.functional.mse_loss(y_hat, y)
    return float(torch.clamp(1.0 - diff, 0.0, 1.0).item())


__all__ = ["pseudo_pesq", "pseudo_stoi"]
