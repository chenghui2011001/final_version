# -*- coding: utf-8 -*-
"""
Mask-as-Channel (MAC) schedule used during training.
"""

from __future__ import annotations

from typing import Dict

import torch


def make_time_mask(
    batch_size: int,
    seq_len: int,
    ratio: float,
    burst_ratio: float,
    csi: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Generate time masks mixing random and burst style dropouts.
    """
    device = next(iter(csi.values())).device if csi else torch.device("cpu")
    mask = torch.ones(batch_size, seq_len, device=device)
    if ratio <= 0:
        return mask

    snr = csi.get("snr_db", torch.zeros(batch_size, device=device)).view(batch_size, 1)
    adaptive_ratio = torch.clamp(ratio + 0.02 * (5.0 - snr), 0.3, 0.95)
    burst_len = max(2, seq_len // 20)

    for b in range(batch_size):
        r = adaptive_ratio[b].item()
        rand_keep = max(0, int(seq_len * r * (1.0 - burst_ratio)))
        bursts_per_sample = max(1, int((r * burst_ratio * seq_len) / burst_len))
        if rand_keep > 0:
            drop_idx = torch.randperm(seq_len, device=device)[:rand_keep]
            mask[b, drop_idx] = 0.0
        for _ in range(bursts_per_sample):
            start = torch.randint(0, max(1, seq_len - burst_len + 1), (1,), device=device).item()
            mask[b, start : start + burst_len] = 0.0
    return mask


__all__ = ["make_time_mask"]
