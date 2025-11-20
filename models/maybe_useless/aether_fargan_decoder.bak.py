# Backup of aether_fargan_decoder before adding decoder-side residual MoE
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import math

try:
    from ..aether_encoder_decoder import AETHERDecoder
    from ..feature_adapter import Feature48To36Adapter, FARGANFeatureSpec
    from ..fargan_components import FARGANCore
except Exception:
    from models.aether_encoder_decoder import AETHERDecoder
    from models.feature_adapter import Feature48To36Adapter, FARGANFeatureSpec
    from models.fargan_components import FARGANCore

try:
    from utils.feature_spec import get_default_feature_spec
except Exception:
    from dnn.torch.final_version.utils.feature_spec import get_default_feature_spec


class PeriodEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.period_proj = nn.Sequential(
            nn.Linear(20, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, fargan_features: torch.Tensor) -> torch.Tensor:
        features_20 = fargan_features[..., :20]
        dnn_pitch = fargan_features[..., 18:19]
        period_raw = 256.0 / torch.pow(2.0, dnn_pitch + 1.5)
        period = torch.round(torch.clamp(period_raw, 32.0, 255.0)).long().squeeze(-1)
        return period


class AETHERFARGANDecoder(AETHERDecoder):
    def __init__(
        self,
        dz: int = 24,
        d_out: int = 36,
        d_hidden: int = 128,
        d_csi: int = 32,
        decoder_heads: int = 2,
        enable_synth: bool = True,
        fargan_subframe_size: int = 40,
        fargan_nb_subframes: int = 4,
        frame_rate_hz: float = 100.0,
        feature_spec_type: str = "fargan",
        use_film: bool = False,
    ):
        super().__init__(
            dz=dz,
            d_out=d_out,
            d_hidden=d_hidden,
            d_csi=d_csi,
            decoder_heads=decoder_heads,
            enable_synth=False,
            feature_spec_type=feature_spec_type,
            use_film=use_film,
        )
        self.enable_fargan_synth = enable_synth
        self.frame_rate_hz = frame_rate_hz
        self.fargan_frame_size = fargan_subframe_size * fargan_nb_subframes
        self.period_estimator = PeriodEstimator()
        if enable_synth:
            self.fargan_core = FARGANCore(
                subframe_size=fargan_subframe_size,
                nb_subframes=fargan_nb_subframes,
                feature_dim=20,
                cond_size=256
            )

    def _estimate_period(self, fargan_features: torch.Tensor) -> torch.Tensor:
        return self.period_estimator(fargan_features)

    def _generate_waveform(self, fargan_features: torch.Tensor, period: torch.Tensor, target_len: Optional[int] = None, fargan_pre: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = fargan_features.shape
        max_available_frames = T - 4
        if target_len is not None:
            target_frames = (target_len + self.fargan_frame_size - 1) // self.fargan_frame_size
            nb_frames = min(max_available_frames, target_frames)
        else:
            nb_frames = max_available_frames
        nb_frames = max(1, nb_frames)
        features_20 = fargan_features[..., :20]
        audio, _ = self.fargan_core(features_20, period, nb_frames, pre=fargan_pre)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if target_len is not None and audio.size(-1) > target_len:
            audio = audio[..., :target_len]
        return audio

    def forward(self, z: torch.Tensor, csi_dict: Dict[str, torch.Tensor], attn_mask: Optional[torch.Tensor] = None, return_wave: bool = False, target_len: Optional[int] = None):
        fargan_features = super()._forward_features(z, csi_dict, attn_mask)
        if not return_wave:
            return fargan_features
        period = self._estimate_period(fargan_features)
        if self.enable_fargan_synth:
            fargan_pre = csi_dict.get('fargan_pre') if csi_dict else None
            audio = self._generate_waveform(fargan_features, period, target_len, fargan_pre)
            return fargan_features, audio
        else:
            audio_len = target_len or (fargan_features.size(1) * self.fargan_frame_size)
            audio = torch.zeros(fargan_features.size(0), 1, audio_len, device=fargan_features.device, dtype=fargan_features.dtype)
            return fargan_features, audio
