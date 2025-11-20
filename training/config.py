# -*- coding: utf-8 -*-
"""
Dataclass for configuring AETHER training runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    # Model
    d_model: int = 128
    dz: int = 24
    gla_depth: int = 2
    n_heads: int = 2
    d_csi: int = 16  # snr_db(1) + ber(1) + fading_onehot(8) + acoustic_priors(6)
    dropout: float = 0.0
    use_film: bool = True
    use_moe: bool = True
    n_experts: int = 4
    top_k: int = 2
    latent_bits: int = 4
    quantize_latent: bool = True
    frame_rate_hz: float = 100.0

    # Optimisation
    lr: float = 2e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    grad_clip: float = 5.0
    stage_schedule: Tuple[int, int, int] = (1000, 1500, 2500)

    # Curriculum
    mask_ratio: float = 0.75
    burst_ratio: float = 0.15

    # Loss weights
    lambda_stft: float = 0.25
    lambda_rate: float = 1e-1  # Increased from 1e-3 for stronger rate control
    lambda_rate_rho: float = 1e-2  # Increased adaptation step size
    target_kbps: float = 1.0
    kbps_tolerance: float = 0.2
    lambda_update_interval: int = 50
    lambda_balance: float = 1e-2
    lambda_cons: float = 1e-2

    # Stage toggles
    stage: str = "C"  # {"A": baseline, "B": +FiLM, "C": full model}

    # FiLM progressive activation
    film_warmup_steps: int = 100      # 前多少 step 线性从 start_ratio→1.0
    film_start_ratio: float = 0.5     # 第一步先只激活 50% 通道
    film_beta_scale: float = 0.2      # 初期把 beta 缩小，减小漂移

    # Logging & checkpoints
    run_dir: str = "dnn/torch/final_version/runs"
    checkpoint_interval: int = 500
    log_interval: int = 50
    flush_interval: int = 200
    insight_buffer_limit: int = 256


__all__ = ["TrainConfig"]
