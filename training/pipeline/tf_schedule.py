#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher forcing helpers shared across staged trainers.

The monolithic pipeline used bespoke logic for the FARGAN warm-up stage.
This module captures the essential pieces so Stage 2/3 scripts can keep the
same semantics without depending on the original implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class TeacherForcingSchedule:
    """Linear teacher forcing envelope."""

    start_step: int
    end_step: int
    max_ratio: float = 1.0

    def ratio(self, step: int) -> float:
        """Compute the teacher-forcing ratio for a global optimisation step."""
        if step <= self.start_step:
            return float(self.max_ratio)
        if step >= self.end_step:
            return 0.0
        span = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / span
        return float(self.max_ratio * max(0.0, 1.0 - progress))


def inject_fargan_pre(
    csi_dict: Optional[Dict[str, torch.Tensor]],
    target_audio: Optional[torch.Tensor],
    teacher_ratio: float,
    frames: int = 2,
    frame_hop: int = 160,
    min_ratio: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Build the ``csi_dict`` consumed by ``FarganWaveHead`` so that teacher forcing
    can feed the vocoder with ground-truth excitation during warm-up.
    """
    csi = {} if csi_dict is None else dict(csi_dict)
    if teacher_ratio <= min_ratio:
        csi.pop("fargan_pre", None)
        return csi

    if target_audio is None:
        csi.pop("fargan_pre", None)
        return csi

    # ``target_audio`` is expected to be [B, T] PCM at 16 kHz.
    audio = target_audio
    if audio.dim() == 3 and audio.size(1) == 1:
        audio = audio.squeeze(1)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    pre_frames = max(0, min(frames, audio.size(-1) // frame_hop))
    if pre_frames <= 0:
        csi.pop("fargan_pre", None)
        return csi

    pre_len = pre_frames * frame_hop
    csi["fargan_pre"] = audio[..., :pre_len].detach()
    return csi


__all__ = [
    "TeacherForcingSchedule",
    "inject_fargan_pre",
]
