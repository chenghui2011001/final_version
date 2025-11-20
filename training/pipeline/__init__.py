"""
Staged training pipeline for AETHER + FARGAN.

Stage 1: AETHER feature reconstruction
Stage 2: Standalone FARGAN vocoder training (see ``train_fargan.py``)
Stage 3: End-to-end optimisation with FiLM/MoE
Stage 4: Quantised full optimisation with channel perturbations
"""

from .stages import StageConfig, get_stage_config, iter_stage_configs
from .tf_schedule import TeacherForcingSchedule, inject_fargan_pre
from .wave_loss import fargan_wave_losses
from .eval_hooks import dump_audio_examples, run_audio_validation

__all__ = [
    "StageConfig",
    "get_stage_config",
    "iter_stage_configs",
    "TeacherForcingSchedule",
    "inject_fargan_pre",
    "fargan_wave_losses",
    "dump_audio_examples",
    "run_audio_validation",
]
