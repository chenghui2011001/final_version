# -*- coding: utf-8 -*-
"""
Utility helpers used across AETHER model components.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def phi_feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Kernel feature map used for linear attention.

    This ELU based mapping keeps values positive which avoids the need for
    explicit softmax normalisation while retaining numerical stability.
    """
    return torch.nn.functional.elu(x) + 1.0


def build_csi_vec(csi_dict: Dict[str, torch.Tensor], target_dim: int = 10, warn_mismatch: bool = True) -> torch.Tensor:
    """
    Concatenate CSI metadata tensors into a dense vector of size ``d_out``.
    Missing dimensions are padded with zeros so the caller always receives a
    deterministic width vector.

    Args:
        csi_dict: Dictionary containing CSI tensors (snr_db, fading_onehot, ber, acoustic_priors, etc.)
        d_out: Target output dimension
        warn_mismatch: Print warning if actual CSI size doesn't match d_out
    """
    # Handle None CSI dict
    if csi_dict is None or not csi_dict:
        device = torch.device("cpu")
        dtype = torch.float32
        # Return zero vector for missing CSI
        return torch.zeros(target_dim, device=device, dtype=dtype)

    device = next(iter(csi_dict.values())).device
    dtype = next(iter(csi_dict.values())).dtype

    items = []
    # Core SNR (normalised)
    if "snr_db" in csi_dict:
        snr = csi_dict["snr_db"].to(device=device, dtype=dtype)
        # map dB [-5,30] -> [-1,1] approximately
        snr_n = ((snr.clamp(-5.0, 30.0) + 5.0) / 35.0) * 2.0 - 1.0
        items.append(snr_n.unsqueeze(-1))
    # Universal4 proxies support
    if "snr_proxy" in csi_dict:
        snr_p = csi_dict["snr_proxy"].to(device=device, dtype=dtype)
        snr_pn = ((snr_p.clamp(-5.0, 30.0) + 5.0) / 35.0) * 2.0 - 1.0
        items.append(snr_pn.unsqueeze(-1))
    if "time_selectivity" in csi_dict:
        ts = csi_dict["time_selectivity"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        items.append(ts.unsqueeze(-1))
    if "freq_selectivity" in csi_dict:
        fs = csi_dict["freq_selectivity"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        items.append(fs.unsqueeze(-1))
    if "los_ratio" in csi_dict:
        lr = csi_dict["los_ratio"].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        items.append(lr.unsqueeze(-1))

    # Optional fading severity stats (prioritise compact scalar set over one-hot type)
    for key, scale, clip, name in [
        ("k_factor_db", 1.0/20.0, (-10.0, 20.0), 'k_factor_db'),
        ("doppler_norm", 1.0, (0.0, 0.2), 'doppler_norm'),
        ("tau_rms_ms", 1.0/10.0, (0.0, 10.0), 'tau_rms_ms'),
        ("coherence_frames", 1.0/100.0, (0.0, 100.0), 'coherence_frames'),
        ("loss_prob", 1.0, (0.0, 1.0), 'loss_prob'),
        ("burst_len_mean", 1.0/10.0, (0.0, 10.0), 'burst_len_mean'),
        ("rate_margin", 1.0, (0.0, 1.0), 'rate_margin'),
        ("buffer_level", 1.0, (0.0, 1.0), 'buffer_level'),
    ]:
        if key in csi_dict:
            v = csi_dict[key].to(device=device, dtype=dtype)
            lo, hi = clip
            v = v.clamp(lo, hi) * scale
            items.append(v.unsqueeze(-1))

    # Back-compat: append type one-hot / BER only if present and we still have room
    if "fading_onehot" in csi_dict and len(items) < target_dim:
        items.append(csi_dict["fading_onehot"].to(device=device, dtype=dtype))
    if "ber" in csi_dict and len(items) < target_dim:
        items.append(csi_dict["ber"].to(device=device, dtype=dtype).unsqueeze(-1))
    if items:
        csi_vec = torch.cat(items, dim=-1)
    else:
        csi_vec = torch.zeros((1, 0), device=device, dtype=dtype)

    width = csi_vec.shape[-1]

    # Warn about dimension mismatches to help maintain consistency
    if warn_mismatch and width != target_dim and width > 0:
        print(f"CSI dimension mismatch: actual={width}, expected={target_dim}")
        if width < target_dim:
            print(f"  Padding with {target_dim - width} zeros")
        else:
            print(f"  Truncating {width - target_dim} dimensions")

    if width < target_dim:
        pad = torch.zeros(csi_vec.shape[0], target_dim - width, device=device, dtype=dtype)
        csi_vec = torch.cat([csi_vec, pad], dim=-1)
    elif width > target_dim:
        csi_vec = csi_vec[..., :target_dim]

    return csi_vec


def global_pool(h: torch.Tensor) -> torch.Tensor:
    """
    Simple time-average pooling with masking support.

    Args:
        h: Tensor with shape ``[B, T, D]``.

    Returns:
        Tensor with shape ``[B, D]`` containing the mean along the temporal axis.
    """
    return h.mean(dim=1)


def extract_acoustic_priors(x: torch.Tensor) -> torch.Tensor:
    """基于特征维度自动选择布局提炼 6 维低开销先验：voiced率、F0稳定度、能量均值/方差、韵律能量、调制度"""
    B, T, D = x.shape

    if D == 36:
        # 36-dim FARGAN layout: ceps(18) + dnn_pitch(1) + frame_corr(1) + lpc(16)
        ceps = x[..., :18]                        # [B,T,18]
        dnn_pitch = x[..., 18:19]                 # [B,T,1]
        frame_corr = x[..., 19:20]                # [B,T,1]
        lpc = x[..., 20:36]                       # [B,T,16]

        c0 = ceps[..., :1]                        # 近似能量项

        # 从 dnn_pitch 推导 voicing (dnn_pitch > 阈值认为有声)
        voiced_rate   = (dnn_pitch > -1.0).float().mean(dim=1)  # [B,1]
        f0_stability  = 1.0 / (1.0 + dnn_pitch.std(dim=1))     # [B,1]
        energy_mean   = c0.mean(dim=1)                          # [B,1]
        energy_var    = c0.var(dim=1)                           # [B,1]
        # 使用 frame_corr 作为韵律特征
        prosody_energy= frame_corr.abs().mean(dim=1)            # [B,1]
        # 使用 lpc 前两维作为调制特征
        modulation    = lpc[..., 0:1].mean(dim=1)               # [B,1]

    else:
        raise ValueError(f"Unsupported feature dimension: {D}. Expected 36 (FARGAN)")

    return torch.cat([voiced_rate, f0_stability, energy_mean,
                      energy_var, prosody_energy, modulation], dim=-1)  # [B,6]


def straight_through_latent_quantizer(z: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Improved uniform quantiser with adaptive dynamic range and higher precision.
    """
    if bits <= 0:
        return z

    # Use higher effective bits for stage1 to preserve energy
    effective_bits = max(bits, 6)  # Minimum 6 bits for better precision
    levels = 2**effective_bits

    # Adaptive dynamic range instead of fixed tanh compression
    # Use percentile-based scaling to better utilize quantization levels
    z_flat = z.view(-1)
    if z_flat.numel() > 1:
        # Convert to float32 for quantile computation (AMP compatibility)
        z_flat_f32 = z_flat.float()
        # Use 99.5th percentile to determine dynamic range (more adaptive than tanh)
        q_low = torch.quantile(z_flat_f32, 0.005)
        q_high = torch.quantile(z_flat_f32, 0.995)
        scale = torch.max(torch.abs(q_low), torch.abs(q_high)) + 1e-6
        z_scaled = z / scale.to(z.dtype)  # Convert scale back to original dtype
    else:
        z_scaled = torch.tanh(z)  # Fallback for single values
        scale = 1.0

    # Clamp to [-1, 1] and quantize
    z_clamped = torch.clamp(z_scaled, -1.0, 1.0)
    scaled = (z_clamped + 1.0) * 0.5 * (levels - 1)
    quantised = torch.round(scaled).clamp(0, levels - 1)
    dequant = quantised / (levels - 1) * 2.0 - 1.0

    # Scale back and apply straight-through
    dequant_scaled = dequant * scale if isinstance(scale, (int, float)) else dequant * scale.detach()
    return z + (dequant_scaled - z).detach()


def estimated_bitrate(z: torch.Tensor, bits: int = 4, frame_rate_hz: float = 100.0) -> float:
    """
    Compute estimated bitrate in kbps for quantized latent tensor.
    """
    if z.dim() == 3:  # [B, T, D]
        dz = z.shape[-1]
    else:
        dz = z.numel()

    bits_per_frame = dz * bits
    kbps = bits_per_frame * frame_rate_hz / 1000.0
    return kbps


def validate_csi_config(csi_dict: Dict[str, torch.Tensor], config_d_csi: int) -> Dict[str, int]:
    """
    Validate CSI configuration consistency and return dimension breakdown.

    Args:
        csi_dict: Sample CSI dictionary
        config_d_csi: Expected d_csi from configuration

    Returns:
        Dictionary with dimension breakdown and recommendations
    """
    # Handle None CSI dict
    if csi_dict is None or not csi_dict:
        return {
            "total_actual": 0,
            "config_d_csi": config_d_csi,
            "recommendation": f"CSI is None, use zeros({config_d_csi})"
        }

    breakdown = {}
    total_dims = 0

    if "snr_db" in csi_dict:
        breakdown["snr_db"] = 1
        total_dims += 1

    if "fading_onehot" in csi_dict:
        fading_dims = csi_dict["fading_onehot"].shape[-1]
        breakdown["fading_onehot"] = fading_dims
        total_dims += fading_dims

    if "ber" in csi_dict:
        breakdown["ber"] = 1
        total_dims += 1

    if "acoustic_priors" in csi_dict:
        pri_dim = csi_dict["acoustic_priors"].shape[-1]
        breakdown["acoustic_priors"] = pri_dim
        total_dims += pri_dim

    breakdown["total_actual"] = total_dims
    breakdown["config_d_csi"] = config_d_csi
    breakdown["padding_needed"] = max(0, config_d_csi - total_dims)
    breakdown["truncation_needed"] = max(0, total_dims - config_d_csi)
    breakdown["is_consistent"] = (total_dims == config_d_csi)

    return breakdown


__all__ = [
    "phi_feature_map",
    "build_csi_vec",
    "validate_csi_config",
    "global_pool",
    "extract_acoustic_priors",
    "straight_through_latent_quantizer",
    "estimated_bitrate",
]
