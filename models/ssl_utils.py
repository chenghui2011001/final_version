#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSL Model Loading Utilities

This module provides unified SSL model loading for semantic feature extraction.
Supports HuBERT, WavLM, and other transformers SSL models with consistent
interface for the SemanticFeatureExtractor.

Key Features:
- Unified model loading interface for different SSL architectures
- Model caching for efficient memory usage
- Device management and model optimization
- Fallback mechanisms for model loading failures
"""

from __future__ import annotations
import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any, Union
from pathlib import Path
import os

try:
    from transformers import (
        HubertModel, HubertConfig,
        WavLMModel, WavLMConfig,
        Wav2Vec2Model, Wav2Vec2Config,
        AutoModel, AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. SSL models will use dummy fallback.")


class SSLModelWrapper(nn.Module):
    """Wrapper for SSL models to ensure consistent interface"""

    def __init__(self, model, model_type: str = "unknown"):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True):
        """Forward pass with consistent output format"""
        with torch.no_grad():
            if self.model_type in ["hubert", "wavlm", "wav2vec2"]:
                return self.model(audio, output_hidden_states=output_hidden_states)
            else:
                # Generic transformers model
                return self.model(audio, output_hidden_states=output_hidden_states)


class DummySSLModel(nn.Module):
    """Dummy SSL model for testing and fallback"""

    def __init__(self, hidden_size: int = 768, num_layers: int = 13):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, audio: torch.Tensor, output_hidden_states: bool = True):
        """Create dummy hidden states matching SSL model output"""
        B, T_audio = audio.shape
        # Estimate SSL frame rate (typically 50Hz for 16kHz audio)
        T_ssl = T_audio // 320  # 320 samples per frame at 50Hz

        # Create dummy hidden states for all layers
        hidden_states = []
        for layer_idx in range(self.num_layers):
            layer_output = torch.randn(
                B, T_ssl, self.hidden_size,
                device=audio.device,
                dtype=audio.dtype
            )
            hidden_states.append(layer_output)

        return type('Output', (), {'hidden_states': hidden_states})()


# Global model cache to avoid reloading models
_SSL_MODEL_CACHE: Dict[str, SSLModelWrapper] = {}


def load_ssl_model(model_name: str,
                   device: Optional[Union[str, torch.device]] = None,
                   cache: bool = True) -> SSLModelWrapper:
    """
    Load SSL model with unified interface

    Args:
        model_name: Model identifier (e.g., "hubert-base", "wavlm-base-plus")
        device: Target device (default: auto-detect)
        cache: Whether to cache loaded models

    Returns:
        SSLModelWrapper: Wrapped SSL model with consistent interface
    """
    # Device management
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Check cache first
    cache_key = f"{model_name}_{device}"
    if cache and cache_key in _SSL_MODEL_CACHE:
        return _SSL_MODEL_CACHE[cache_key]

    # Load model
    try:
        model = _load_ssl_model_impl(model_name, device)
    except Exception as e:
        warnings.warn(f"Failed to load SSL model {model_name}: {e}")
        warnings.warn("Using dummy SSL model for testing")
        model = SSLModelWrapper(DummySSLModel(), "dummy")

    # Move to device
    model = model.to(device)

    # Cache if requested
    if cache:
        _SSL_MODEL_CACHE[cache_key] = model

    return model


def _load_ssl_model_impl(model_name: str, device: torch.device) -> SSLModelWrapper:
    """Implementation of SSL model loading"""

    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available")

    # Normalize model name
    model_name = model_name.lower().replace("_", "-")

    # HuBERT models
    if "hubert" in model_name:
        if "base" in model_name:
            model_id = "facebook/hubert-base-ls960"
        elif "large" in model_name:
            model_id = "facebook/hubert-large-ls960-ft"
        else:
            model_id = "facebook/hubert-base-ls960"  # Default

        model = HubertModel.from_pretrained(model_id)
        return SSLModelWrapper(model, "hubert")

    # WavLM models
    elif "wavlm" in model_name:
        if "large" in model_name:
            model_id = "microsoft/wavlm-large"
        elif "base-plus" in model_name:
            model_id = "microsoft/wavlm-base-plus"
        else:
            model_id = "microsoft/wavlm-base"  # Default

        model = WavLMModel.from_pretrained(model_id)
        return SSLModelWrapper(model, "wavlm")

    # Wav2Vec2 models
    elif "wav2vec2" in model_name:
        if "large" in model_name:
            model_id = "facebook/wav2vec2-large-960h"
        else:
            model_id = "facebook/wav2vec2-base-960h"  # Default

        model = Wav2Vec2Model.from_pretrained(model_id)
        return SSLModelWrapper(model, "wav2vec2")

    # Try generic AutoModel loading
    else:
        try:
            model = AutoModel.from_pretrained(model_name)
            return SSLModelWrapper(model, "auto")
        except Exception as e:
            raise ValueError(f"Unknown SSL model: {model_name}") from e


def get_ssl_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about SSL model without loading it"""

    model_name = model_name.lower().replace("_", "-")

    if "hubert" in model_name:
        if "large" in model_name:
            return {
                "model_type": "hubert",
                "model_id": "facebook/hubert-large-ls960-ft",
                "hidden_size": 1024,
                "num_layers": 25,
                "estimated_frame_rate": 50.0
            }
        else:
            return {
                "model_type": "hubert",
                "model_id": "facebook/hubert-base-ls960",
                "hidden_size": 768,
                "num_layers": 13,
                "estimated_frame_rate": 50.0
            }

    elif "wavlm" in model_name:
        if "large" in model_name:
            return {
                "model_type": "wavlm",
                "model_id": "microsoft/wavlm-large",
                "hidden_size": 1024,
                "num_layers": 25,
                "estimated_frame_rate": 50.0
            }
        elif "base-plus" in model_name:
            return {
                "model_type": "wavlm",
                "model_id": "microsoft/wavlm-base-plus",
                "hidden_size": 768,
                "num_layers": 13,
                "estimated_frame_rate": 50.0
            }
        else:
            return {
                "model_type": "wavlm",
                "model_id": "microsoft/wavlm-base",
                "hidden_size": 768,
                "num_layers": 13,
                "estimated_frame_rate": 50.0
            }

    elif "wav2vec2" in model_name:
        if "large" in model_name:
            return {
                "model_type": "wav2vec2",
                "model_id": "facebook/wav2vec2-large-960h",
                "hidden_size": 1024,
                "num_layers": 25,
                "estimated_frame_rate": 50.0
            }
        else:
            return {
                "model_type": "wav2vec2",
                "model_id": "facebook/wav2vec2-base-960h",
                "hidden_size": 768,
                "num_layers": 13,
                "estimated_frame_rate": 50.0
            }

    else:
        return {
            "model_type": "unknown",
            "model_id": model_name,
            "hidden_size": 768,
            "num_layers": 13,
            "estimated_frame_rate": 50.0
        }


def clear_ssl_cache():
    """Clear SSL model cache to free memory"""
    global _SSL_MODEL_CACHE
    _SSL_MODEL_CACHE.clear()


def list_available_models() -> list:
    """List available pre-configured SSL models"""
    return [
        "hubert-base",
        "hubert-large",
        "wavlm-base",
        "wavlm-base-plus",
        "wavlm-large",
        "wav2vec2-base",
        "wav2vec2-large"
    ]


def test_ssl_utils():
    """Test SSL utilities functionality"""
    print("Testing SSL utilities...")

    # Test model info
    for model_name in ["hubert-base", "wavlm-base", "wav2vec2-base"]:
        info = get_ssl_model_info(model_name)
        print(f"{model_name}: {info}")

    # Test dummy model loading
    print("\nTesting dummy SSL model...")
    dummy_model = SSLModelWrapper(DummySSLModel(), "dummy")

    # Test input
    audio = torch.randn(2, 16000)  # 2 batches, 1 second audio

    # Test forward pass
    output = dummy_model(audio, output_hidden_states=True)
    print(f"Input audio shape: {audio.shape}")
    print(f"Number of hidden layers: {len(output.hidden_states)}")
    print(f"Hidden state shape: {output.hidden_states[0].shape}")

    # Test model loading (will use dummy if transformers not available)
    try:
        model = load_ssl_model("hubert-base")
        print(f"Loaded model type: {model.model_type}")

        # Test with the loaded model
        output = model(audio, output_hidden_states=True)
        print(f"Loaded model output shape: {output.hidden_states[9].shape}")

    except Exception as e:
        print(f"Model loading test failed (expected): {e}")

    # Test cache clearing
    clear_ssl_cache()
    print("Cache cleared successfully")

    print("SSL utilities test completed!")


if __name__ == "__main__":
    test_ssl_utils()