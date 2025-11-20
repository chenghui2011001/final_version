#!/usr/bin/env python3
"""
Channel simulation and CSI synthesis utilities for Stage4 FiLM testing.

Design goals
- Provide lightweight, physically-inspired channel factors with a few stable scalars
- Return per-batch CSI dict for FiLM (aggregated), and per-frame factors to perturb z
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


class ChannelSimulator:
    def __init__(self, sample_rate: int = 16000, frame_hz: int = 100):
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

    @staticmethod
    def _lowpass_noise(shape, alpha: float, device, dtype, kernel_len: int = 256):
        """First-order IIR low-pass colored noise with pole alpha in (0,1).

        Vectorized approximation using FIR of length ``kernel_len`` with impulse
        response h[k] = (1-alpha) * alpha^k. This avoids Python loops over T and
        dramatically reduces per-batch overhead.
        """
        B, T = shape
        eps = torch.randn(B, T, device=device, dtype=dtype)
        # Build FIR kernel once per call; length sufficient for alpha≈0.98 (tau≈50)
        K = max(16, int(kernel_len))
        k = torch.arange(K, device=device, dtype=dtype)
        h = (1.0 - float(alpha)) * torch.pow(torch.tensor(float(alpha), device=device, dtype=dtype), k)
        # Convolve along time with padding to keep length T
        eps3 = eps.unsqueeze(1)  # [B,1,T]
        h3 = h.view(1, 1, -1)    # [1,1,K]
        y = torch.nn.functional.conv1d(eps3, h3, padding=K - 1).squeeze(1)  # [B, T+K-1]
        return y[:, :T]

    def sample_csi(self, B: int, T: int, channel: str = "fading",
                   snr_min_db: float | None = None,
                   snr_max_db: float | None = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Create CSI dict (aggregated scalars) and per-frame factors for z-perturbation.

        Universal4 proxy (pilot-free friendly) outputs in ``csi``:
          - snr_proxy         (dB): proxy of SNR, derived from simulated SNR trajectory
          - time_selectivity  (0..1): 1 - corr(amp_t[t-1], amp_t[t])
          - freq_selectivity  (0..1): normalised RMS delay spread proxy
          - los_ratio         (0..1): K/(K+1) from Rician K-factor

        Also returns:
          amp_t: [B, T] multiplicative fading envelope (≈1.0 ±)
          snr_db_t: [B, T] per-frame SNR in dB
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32

        # SNR (dB): time-varying around a base in [snr_min_db, snr_max_db]
        lo = -5.0 if snr_min_db is None else float(snr_min_db)
        hi = 15.0 if snr_max_db is None else float(snr_max_db)
        if hi < lo:
            lo, hi = hi, lo
        base_snr = lo + torch.rand(B, device=device, dtype=dtype) * max(hi - lo, 1e-3)
        # doppler_norm ~ [0.0, 0.1]
        doppler_norm = torch.rand(B, device=device, dtype=dtype) * 0.1
        # low-pass color: alpha close to 1.0 for slow variation
        alpha = torch.clamp(1.0 - doppler_norm, 0.85, 0.999)
        alphas = alpha.view(B, 1).expand(B, T)
        # colored noise for snr fluctuation (~ ±3 dB)
        snr_delta = self._lowpass_noise((B, T), alpha=0.98, device=device, dtype=dtype) * 3.0
        snr_db_t = base_snr.view(B, 1) + snr_delta

        # Rician K-factor (dB): [-3, +10] (Rayleigh≈-inf → use small negative)
        k_factor_db = -3.0 + torch.rand(B, device=device, dtype=dtype) * 13.0
        K_lin = torch.pow(10.0, k_factor_db / 10.0)  # [B]

        # Fading amplitude envelope (Rician approx):
        # a(t) ≈ sqrt( (sqrt(K/(K+1)) + n)^2 + n2^2 ), simplified as mean 1.0 +/- colored fluctuation
        fade_lp = self._lowpass_noise((B, T), alpha=0.98, device=device, dtype=dtype) * 0.1
        amp_t = (1.0 + fade_lp).clamp(0.7, 1.3)

        # RMS delay spread (ms) and coherence time (frames)
        tau_rms_ms = 0.1 + torch.rand(B, device=device, dtype=dtype) * 3.0  # 0.1..3.1 ms
        # Coherence time ~ 1/(2 f_D), use doppler_norm ≈ f_D/f_s, frames at 100 Hz
        coh_time_s = (1.0 / (2.0 * (doppler_norm * self.sample_rate + 1e-3))).clamp(0.01, 1.0)
        coherence_frames = (coh_time_s * self.frame_hz).clamp(1.0, 100.0)

        # Burst/loss model (for completeness)
        loss_prob = torch.rand(B, device=device, dtype=dtype) * 0.02  # ≤2%
        burst_len_mean = 1.0 + torch.rand(B, device=device, dtype=dtype) * 4.0  # 1..5 frames

        # System-related auxiliaries (optional)
        rate_margin = torch.rand(B, device=device, dtype=dtype) * 0.5  # 0..0.5
        buffer_level = 0.2 + torch.rand(B, device=device, dtype=dtype) * 0.6  # 0.2..0.8

        # Aggregate CSI scalars (use means)
        snr_db = snr_db_t.mean(dim=1)

        # Universal4 proxies
        # 1) snr_proxy (dB)
        snr_proxy = snr_db

        # 2) time_selectivity (0..1): 1 - corr(amp_t[t-1], amp_t[t])
        x = amp_t[:, :-1]
        y = amp_t[:, 1:]
        x_c = x - x.mean(dim=1, keepdim=True)
        y_c = y - y.mean(dim=1, keepdim=True)
        num = (x_c * y_c).mean(dim=1)
        den = (x_c.std(dim=1) * y_c.std(dim=1)).clamp_min(1e-6)
        corr = (num / den).clamp(-1.0, 1.0)
        time_selectivity = (1.0 - corr).clamp(0.0, 1.0)

        # 3) freq_selectivity (0..1): normalised RMS delay spread proxy
        # tau_rms_ms in [0.1, 3.1] → normalise to [0,1]
        freq_selectivity = ((tau_rms_ms - 0.1) / (3.1 - 0.1)).clamp(0.0, 1.0)

        # 4) los_ratio (0..1): K_lin/(K_lin+1)
        los_ratio = (K_lin / (K_lin + 1.0)).clamp(0.0, 1.0)

        csi = {
            'snr_proxy': snr_proxy,
            'time_selectivity': time_selectivity,
            'freq_selectivity': freq_selectivity,
            'los_ratio': los_ratio,
        }
        return csi, amp_t, snr_db_t

    def apply(self, z: torch.Tensor, amp_t: torch.Tensor, snr_db_t: torch.Tensor) -> torch.Tensor:
        """Apply channel fading/noise to latent z: z'[b,t,d] = a[b,t] * z + n (per-frame SNR control)."""
        B, T, D = z.shape
        device = z.device
        dtype = z.dtype
        # Expand amp to [B,T,1]
        amp = amp_t.to(device=z.device, dtype=z.dtype).unsqueeze(-1)
        z_amp = amp * z
        # Per-frame noise std from SNR(dB): snr = 20*log10(sig/noise), noise = sig / 10^(snr/20)
        # Estimate signal std per-batch (avoid zero):
        sig_std = (z_amp.detach().float().pow(2).mean(dim=(1, 2)).sqrt() + 1e-3).to(dtype)
        # Broadcast to [B,T]
        snr_lin = torch.pow(10.0, (snr_db_t.to(device) / 20.0))
        noise_std_bt = (sig_std.view(B, 1) / (snr_lin + 1e-3)).clamp(1e-6, 1e3)
        noise = torch.randn_like(z_amp) * noise_std_bt.unsqueeze(-1).to(dtype)
        return (z_amp + noise).to(dtype)
