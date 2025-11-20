#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage configuration helpers extracted from the legacy progressive trainer.

These utilities provide a light-weight view of the staged training flow so
the new pipeline scripts can share common semantics without depending on the
monolithic ``progressive_train_clean.py`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class StageConfig:
    """Configuration payload for a staged training phase."""

    name: str
    description: str

    # Scheduling
    steps: Optional[int] = None
    batches: Optional[int] = None
    epochs: Optional[float] = None

    # Architectural switches
    use_film: bool = False
    use_moe: bool = False
    use_quantization: bool = False
    apply_channel: bool = False
    channel_type: str = "clean"

    # Loss wiring
    layered_loss: bool = False
    learning_rate: float = 2e-4
    lambda_rate: float = 0.0
    lambda_balance: float = 0.0
    lambda_cons: float = 0.0

    # Audio quality gates
    enable_audio_quality: bool = False
    min_snr_db: float = 5.0
    min_mel_cos: float = 0.85
    max_mel_l2: float = 0.15
    max_spectral_distortion: float = 0.65
    max_rms_delta_db: float = 3.0

    # Teacher forcing / waveform controls
    wave_start_step: int = 0
    wave_full_start_step: int = 0
    wave_lowpass_weight: float = 0.5
    wave_full_weight: float = 1.0
    wave_lowpass_schedule: List[Tuple[int, float]] = field(default_factory=list)
    wave_full_schedule: List[Tuple[int, float]] = field(default_factory=list)
    train_wave_head_only: bool = False
    preheat_mix_start_step: int = 0
    preheat_mix_end_step: int = 0
    preheat_chunk_frames: int = 0
    layered_warmup_steps: int = 0
    anti_static_weight: float = 0.0

    # FiLM scheduler knobs
    use_advanced_scheduler: bool = False
    film_warmup_steps: int = 0
    film_start_ratio: float = 1.0
    film_beta_scale_start: float = 1.0

    # Analytics / convergence
    min_convergence_rate: float = 0.5
    max_final_loss: float = 2.0
    early_stop_loss: float = 0.01
    target_kbps: float = 0.0
    max_kbps_p90: float = 0.0

    def total_steps(self, total_batches: int) -> int:
        """Resolve the intended number of optimisation steps for the stage."""
        if self.steps is not None:
            return max(1, int(self.steps))
        if self.batches is not None:
            return max(1, int(self.batches))
        if self.epochs is not None:
            return max(1, int(round(self.epochs * total_batches)))
        return max(1, total_batches)

    def layered_enabled(self, step: int) -> bool:
        """Return True if layered loss should be active at the given step."""
        return self.layered_loss and step >= self.layered_warmup_steps

    @staticmethod
    def scheduled_value(schedule: List[Tuple[int, float]], step: int, default: float) -> float:
        """Evaluate a staircase schedule (step -> value)."""
        if not schedule:
            return default
        value = default
        for boundary, candidate in sorted(schedule, key=lambda item: item[0]):
            if step >= boundary:
                value = candidate
            else:
                break
        return value


def _stage_configurations() -> Dict[str, StageConfig]:
    """Snapshot of the four classic progressive stages."""
    return {
        "stage1": StageConfig(
            name="clean_baseline",
            description="Feature reconstruction without channel constraints.",
            epochs=1.0,
            use_film=False,  # 禁用FiLM，避免潜在特征z被过度正则化
            use_moe=False,   # 禁用MoE，避免潜在特征z被过度正则化
            learning_rate=5e-4,
            lambda_rate=0.0,  # 禁用rate损失，专注特征重建质量
            min_convergence_rate=0.5,
            max_final_loss=0.2,
            early_stop_loss=0.01,
            layered_warmup_steps=0,
            anti_static_weight=1e-3,  # 增加anti_static_weight，避免模型输出低方差结果
        ),
        "stage2": StageConfig(
            name="wave_preheat",
            description="Warm up FARGAN vocoder with teacher forcing.",
            epochs=2.0,
            layered_loss=True,
            learning_rate=1e-4,
            min_convergence_rate=-20.0,
            max_final_loss=3.0,
            train_wave_head_only=True,
            wave_start_step=0,
            wave_full_start_step=1200,
            wave_lowpass_weight=0.9,
            wave_full_weight=1.0,
            wave_lowpass_schedule=[(0, 1.2), (800, 0.8), (1200, 0.6)],
            wave_full_schedule=[(0, 0.2), (600, 0.5), (1200, 0.8), (1800, 1.0)],
            preheat_mix_start_step=0,
            preheat_mix_end_step=15000,
            preheat_chunk_frames=128,
            layered_warmup_steps=3000,
            anti_static_weight=2e-3,
        ),
        "stage3": StageConfig(
            name="channel_adapt",
            description="Introduce MoE (no FiLM, no channel) for isolated ablation.",
            epochs=3.0,
            use_film=False,   # Stage3: 禁用FiLM（单变量验证）
            use_moe=True,
            apply_channel=False,  # Stage3: 禁用信道模拟
            channel_type="clean",
            layered_loss=True,
            learning_rate=5e-5,  # 大幅降低学习率防止梯度爆炸 (从3e-4降到5e-5)
            lambda_rate=0.1,
            lambda_balance=0.02,  # 提高MoE均衡损失权重，缓解专家塌缩
            min_convergence_rate=1.0,
            max_final_loss=1.0,
            enable_audio_quality=True,
            min_snr_db=3.0,
            min_mel_cos=0.75,
            max_mel_l2=0.30,
            max_spectral_distortion=0.90,
            max_rms_delta_db=5.0,
            use_advanced_scheduler=True,
            film_warmup_steps=500,
            film_start_ratio=0.1,
            film_beta_scale_start=0.1,
            layered_warmup_steps=3000,
            anti_static_weight=1e-4,
        ),
        "stage4": StageConfig(
            name="full_optimization",
            description="Full end-to-end optimisation with quantisation.",
            epochs=5.0,
            use_film=True,
            use_moe=True,
            use_quantization=True,
            apply_channel=True,
            channel_type="fading",
            layered_loss=True,
            learning_rate=2e-4,
            lambda_rate=0.2,
            lambda_balance=0.1,
            lambda_cons=0.05,
            min_convergence_rate=0.5,
            max_final_loss=1.0,
            enable_audio_quality=True,
            min_snr_db=10.0,
            min_mel_cos=0.90,
            max_mel_l2=0.12,
            max_spectral_distortion=0.60,
            max_rms_delta_db=2.5,
            target_kbps=1.2,
            max_kbps_p90=1.6,
            layered_warmup_steps=3000,
            anti_static_weight=1.5e-4,
        ),
    }


def get_stage_config(key: str) -> StageConfig:
    """Return a copied stage configuration by registry key."""
    configs = _stage_configurations()
    if key not in configs:
        raise KeyError(f"Unknown stage key: {key}")
    cfg = configs[key]
    # shallow dataclass copy so callers can mutate without touching defaults
    return StageConfig(**cfg.__dict__)


def iter_stage_configs(keys: Iterable[str]) -> List[StageConfig]:
    """Materialise a list of stage configs matching the provided keys."""
    return [get_stage_config(k) for k in keys]


__all__ = [
    "StageConfig",
    "get_stage_config",
    "iter_stage_configs",
]
