#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Feature Extractor for 16-dimensional semantic/style latent space

This module extracts semantic features from raw audio using pre-trained SSL models
(HuBERT/WavLM) and projects them to a 16-dimensional semantic space that aligns
with FARGAN's frame rate for unified training.

Key Features:
- Low-rate semantic extraction (25Hz) for content stability
- Frame rate alignment to FARGAN (100Hz) via broadcasting
- Frozen SSL teacher models for consistent semantic supervision
- 16-dimensional projection for semantic/style latent representation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import warnings

from .ssl_utils import load_ssl_model, get_ssl_model_info, SSLModelWrapper

class SemanticFeatureExtractor(nn.Module):
    """
    Extract 16-dimensional semantic/style features from raw audio

    Architecture:
    Raw Audio -> SSL Model -> Layer Selection -> Subsampling (25Hz) ->
    Projection (16D) -> Broadcasting (100Hz) -> FARGAN-aligned Features
    """

    def __init__(self,
                 ssl_model: Union[str, SSLModelWrapper],
                 layer_idx: int = 9,
                 proj_dim: int = 16,
                 frame_subsample: int = 4,
                 fargan_frame_rate: float = 100.0,
                 sample_rate: int = 16000,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize Semantic Feature Extractor

        Args:
            ssl_model: SSL model name (str) or pre-loaded SSLModelWrapper
            layer_idx: Layer index for feature extraction (9 for HuBERT-base)
            proj_dim: Output semantic dimension (16 for 16-dim semantic space)
            frame_subsample: Subsampling factor (4 = 100Hz->25Hz)
            fargan_frame_rate: Target FARGAN frame rate (100Hz)
            sample_rate: Audio sample rate (16kHz)
            device: Target device for computation
        """
        super().__init__()

        # Device management
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load SSL model if string provided
        if isinstance(ssl_model, str):
            self.ssl_model = load_ssl_model(ssl_model, device=self.device)
            self.ssl_model_name = ssl_model
        else:
            self.ssl_model = ssl_model.to(self.device)
            self.ssl_model_name = getattr(ssl_model, 'model_type', 'unknown')

        # Configuration
        self.layer_idx = layer_idx
        self.proj_dim = proj_dim
        self.frame_subsample = frame_subsample
        self.fargan_frame_rate = fargan_frame_rate
        self.sample_rate = sample_rate

        # SSL model configuration
        if hasattr(self.ssl_model.model, 'config'):
            self.ssl_hidden_size = self.ssl_model.model.config.hidden_size
        elif isinstance(ssl_model, str):
            # Get info from SSL utils
            ssl_info = get_ssl_model_info(ssl_model)
            self.ssl_hidden_size = ssl_info['hidden_size']
        else:
            # Fallback for models without config attribute
            self.ssl_hidden_size = 768  # Common size for HuBERT-base/WavLM-base

        # Projection network: SSL hidden -> 16D semantic space
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.ssl_hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.proj_dim)
        )

        # Initialize projection weights
        self._init_projection_weights()
        # Ensure projection and all submodules are on target device
        try:
            self.to(self.device)
        except Exception:
            # Fallback: at least move the projection to target device
            try:
                self.semantic_projection.to(self.device)
            except Exception:
                pass

        # Calculate frame timing
        self.ssl_frame_rate = self._estimate_ssl_frame_rate()
        self.semantic_frame_rate = self.ssl_frame_rate / self.frame_subsample

        print(f"SemanticFeatureExtractor initialized:")
        print(f"  SSL model: {self.ssl_model_name}")
        print(f"  SSL frame rate: {self.ssl_frame_rate:.1f} Hz")
        print(f"  Semantic frame rate: {self.semantic_frame_rate:.1f} Hz")
        print(f"  FARGAN frame rate: {self.fargan_frame_rate:.1f} Hz")
        print(f"  Projection: {self.ssl_hidden_size} -> {self.proj_dim}")
        print(f"  Device: {self.device}")

    def _init_projection_weights(self):
        """Initialize projection network weights"""
        for module in self.semantic_projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _estimate_ssl_frame_rate(self) -> float:
        """Estimate SSL model frame rate"""
        # Most SSL models operate at ~50Hz (20ms frame shift)
        # HuBERT/WavLM typically use 20ms frame shift at 16kHz
        return 50.0

    @torch.no_grad()
    def extract_lowrate_semantic(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract low-rate semantic features (25Hz) from audio

        Args:
            audio: [B, T_audio] Raw audio at 16kHz

        Returns:
            semantic_low: [B, T_semantic, 16] Low-rate semantic features
        """
        # Ensure audio is in correct format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [T] -> [1, T]

        # Extract SSL features with error handling
        try:
            ssl_output = self.ssl_model(audio, output_hidden_states=True)
            ssl_features = ssl_output.hidden_states[self.layer_idx]  # [B, T_ssl, D_ssl]
        except Exception as e:
            warnings.warn(f"SSL feature extraction failed: {e}")
            # Fallback: create dummy features
            B, T_audio = audio.shape
            T_ssl_est = T_audio // 320  # Rough estimate for 50Hz
            ssl_features = torch.zeros(B, T_ssl_est, self.ssl_hidden_size,
                                     device=audio.device, dtype=audio.dtype)

        # Subsample to low-rate semantic features (25Hz)
        ssl_lowrate = ssl_features[:, ::self.frame_subsample, :]  # [B, T_semantic, D_ssl]

        # Project to 16-dimensional semantic space
        semantic_features = self.semantic_projection(ssl_lowrate)  # [B, T_semantic, 16]

        return semantic_features

    def broadcast_to_fargan_rate(self,
                                semantic_low: torch.Tensor,
                                target_frames: int) -> torch.Tensor:
        """
        Broadcast low-rate semantic features to FARGAN frame rate

        Args:
            semantic_low: [B, T_semantic, 16] Low-rate semantic features
            target_frames: Target number of FARGAN frames

        Returns:
            semantic_fargan: [B, target_frames, 16] FARGAN-rate semantic features
        """
        B, T_semantic, D = semantic_low.shape

        # Method 1: Simple repetition (preserves semantic consistency)
        if self.frame_subsample == 4:
            # Each semantic frame covers 4 FARGAN frames
            semantic_repeated = semantic_low.unsqueeze(2).repeat(1, 1, self.frame_subsample, 1)
            semantic_full = semantic_repeated.reshape(B, T_semantic * self.frame_subsample, D)
        else:
            # General case: linear interpolation
            semantic_full = F.interpolate(
                semantic_low.transpose(1, 2),  # [B, 16, T_semantic]
                size=target_frames,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, target_frames, 16]

        # Trim or pad to exact target length
        if semantic_full.size(1) > target_frames:
            semantic_full = semantic_full[:, :target_frames, :]
        elif semantic_full.size(1) < target_frames:
            # Pad with last frame
            last_frame = semantic_full[:, -1:, :].expand(-1, target_frames - semantic_full.size(1), -1)
            semantic_full = torch.cat([semantic_full, last_frame], dim=1)

        return semantic_full

    def forward(self,
                audio: torch.Tensor,
                target_frames: Optional[int] = None) -> torch.Tensor:
        """
        Extract semantic features aligned to FARGAN frame rate

        Args:
            audio: [B, T_audio] Raw audio at 16kHz
            target_frames: Target number of FARGAN frames (optional)

        Returns:
            semantic_features: [B, target_frames, 16] FARGAN-aligned semantic features
        """
        # Extract low-rate semantic features
        semantic_low = self.extract_lowrate_semantic(audio)  # [B, T_semantic, 16]

        # Determine target frames if not provided
        if target_frames is None:
            # Estimate based on audio length and FARGAN frame rate
            T_audio = audio.size(-1)
            target_frames = int(T_audio * self.fargan_frame_rate / self.sample_rate)

        # Broadcast to FARGAN frame rate
        semantic_fargan = self.broadcast_to_fargan_rate(semantic_low, target_frames)

        return semantic_fargan

    def get_semantic_info(self) -> dict:
        """Get semantic extractor information for debugging"""
        return {
            'ssl_model_name': self.ssl_model_name,
            'ssl_model_type': self.ssl_model.model_type,
            'layer_idx': self.layer_idx,
            'ssl_hidden_size': self.ssl_hidden_size,
            'proj_dim': self.proj_dim,
            'frame_subsample': self.frame_subsample,
            'ssl_frame_rate': self.ssl_frame_rate,
            'semantic_frame_rate': self.semantic_frame_rate,
            'fargan_frame_rate': self.fargan_frame_rate,
            'device': str(self.device)
        }


def create_semantic_extractor(model_name: str = "hubert-base",
                              proj_dim: int = 16,
                              device: Optional[Union[str, torch.device]] = None) -> SemanticFeatureExtractor:
    """
    Convenience factory for creating SemanticFeatureExtractor with sensible defaults

    Args:
        model_name: SSL model name (default: "hubert-base")
        proj_dim: Output semantic dimension (default: 16)
        device: Target device (default: auto-detect)

    Returns:
        SemanticFeatureExtractor: Ready-to-use semantic extractor
    """
    return SemanticFeatureExtractor(
        ssl_model=model_name,
        layer_idx=9,  # Good layer for HuBERT/WavLM
        proj_dim=proj_dim,
        frame_subsample=4,  # 25Hz semantic rate
        fargan_frame_rate=100.0,
        sample_rate=16000,
        device=device
    )


def test_semantic_extractor():
    """Test function for SemanticFeatureExtractor"""
    print("Testing SemanticFeatureExtractor...")

    # Test setup with SSL utils (will use dummy if transformers not available)
    extractor = create_semantic_extractor("hubert-base")

    # Test input
    batch_size = 2
    audio_length = 16000  # 1 second at 16kHz
    audio = torch.randn(batch_size, audio_length)

    # Test extraction
    semantic_features = extractor(audio, target_frames=100)

    print(f"Input audio shape: {audio.shape}")
    print(f"Output semantic features shape: {semantic_features.shape}")
    print(f"Expected: [2, 100, 16]")

    # Test low-rate extraction
    semantic_low = extractor.extract_lowrate_semantic(audio)
    print(f"Low-rate semantic shape: {semantic_low.shape}")

    # Test info
    info = extractor.get_semantic_info()
    print("Extractor info:", info)

    print("SemanticFeatureExtractor test completed!")


if __name__ == "__main__":
    test_semantic_extractor()
