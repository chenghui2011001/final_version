#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation helpers shared by the staged pipeline.

The original trainer implemented an extensive validation stack. Here we expose
the pieces that are useful for the new modular scripts with conservative
defaults and robust error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

# Use absolute import from final_version root to support script execution
from utils.audio_validation_generator import (
    export_validation_audio,
    integrate_audio_validation,
)


def run_audio_validation(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    wave_head: Optional[torch.nn.Module],
    dataset,
    feature_spec_type: str,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Execute the validation sweep and return the aggregated metrics."""
    try:
        return integrate_audio_validation(
            encoder=encoder,
            decoder=decoder,
            wave_head=wave_head,
            dataset=dataset,
            feature_spec_type=feature_spec_type,
            output_dir=str(output_dir),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] validation skipped due to error: {exc}")
        return {}


def dump_audio_examples(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    wave_head: Optional[torch.nn.Module],
    batch: Dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
) -> None:
    """Generate preview audio artefacts for the current batch."""
    try:
        export_validation_audio(
            encoder=encoder,
            decoder=decoder,
            wave_head=wave_head,
            batch=batch,
            output_dir=str(output_dir),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] audio export skipped due to error: {exc}")


__all__ = ["run_audio_validation", "dump_audio_examples"]
