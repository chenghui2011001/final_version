#!/usr/bin/env python3
"""
Lightweight 2-FSK modem for simulation: z -> bits -> 2FSK waveform -> bits -> approx z.

Notes
- This is a non-differentiable test harness intended for channel/FiLM robustness checks.
- Default FS and tones respect Nyquist for 16 kHz: f0=1850 Hz, f1=1950 Hz.
- Symbol rate default 2000 sym/s (symbol_len=8 samples).
- To keep runtime reasonable, we encode only a few bits per frame (default 4).
"""
from __future__ import annotations

from typing import Dict, Tuple
import math
import torch


def _goertzel_power(x: torch.Tensor, freq: float, fs: float) -> torch.Tensor:
    """Compute Goertzel power of vector x (last dim) at freq.
    x: [..., N]
    returns: [...]
    """
    N = x.size(-1)
    k = int(0.5 + (N * freq) / fs)
    w = (2.0 * math.pi / N) * k
    coeff = 2.0 * math.cos(w)
    s_prev = torch.zeros_like(x[..., 0])
    s_prev2 = torch.zeros_like(x[..., 0])
    for n in range(N):
        s = x[..., n] + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2.pow(2) + s_prev.pow(2) - coeff * s_prev * s_prev2
    return power


class TwoFSKModem:
    def __init__(self, sample_rate: int = 16000, f0: float = 1850.0, f1: float = 1950.0, sym_rate: int = 2000):
        assert f0 < sample_rate / 2 and f1 < sample_rate / 2, "Tones must be < Nyquist"
        assert sym_rate > 0 and sample_rate % sym_rate == 0, "Use sym_rate dividing sample_rate for integer symbol_len"
        self.fs = float(sample_rate)
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.sym_rate = int(sym_rate)
        self.sym_len = int(sample_rate // sym_rate)

    def bits_from_z(self, z: torch.Tensor, bits_per_frame: int = 4) -> Tuple[torch.Tensor, Dict]:
        """Map z[B,T,D] to bits[B, T*bits_per_frame].
        Use group-mean sign of D split into 'bits_per_frame' chunks.
        Also return meta for approximate reconstruction.
        """
        B, T, D = z.shape
        device = z.device
        dtype = z.dtype
        g = bits_per_frame
        assert g >= 1 and D >= g
        # split along D into g groups (last group may be larger)
        sizes = [D // g] * g
        sizes[-1] += D - sum(sizes)
        z_split = torch.split(z, sizes, dim=-1)
        means = torch.stack([zi.mean(dim=-1) for zi in z_split], dim=-1)  # [B,T,g]
        # meta scales for reconstruction
        rms = torch.stack([zi.float().pow(2).mean(dim=-1).sqrt() + 1e-3 for zi in z_split], dim=-1)  # [B,T,g]
        # Convert means sign to bits
        bits = (means > 0).to(torch.uint8)  # [B,T,g]
        bits_flat = bits.reshape(B, T * g)
        meta = {
            'shape': (B, T, D),
            'group_sizes': sizes,
            'rms': rms.detach().cpu(),  # store on CPU
            'dtype': str(dtype),
        }
        return bits_flat.cpu(), meta

    def z_from_bits(self, bits_flat: torch.Tensor, meta: Dict) -> torch.Tensor:
        """Approximate reconstruct z from bits and meta. Output shape [B,T,D].
        Recreate group means +/- rms and fill all dims in the group to that mean.
        """
        B, T, D = meta['shape']
        g = len(meta['group_sizes'])
        bits = bits_flat.view(B, T, g).to(torch.float32)
        rms = meta['rms'].to(bits.device).to(torch.float32)  # [B,T,g]
        means = torch.where(bits > 0.5, rms, -rms)  # [B,T,g]
        # expand back to D dims
        chunks = []
        for gi, size in enumerate(meta['group_sizes']):
            m = means[..., gi:gi+1].expand(B, T, size)
            chunks.append(m)
        z_rec = torch.cat(chunks, dim=-1)
        return z_rec.to(torch.float32)

    def modulate(self, bits_flat: torch.Tensor, amp: float = 0.8) -> torch.Tensor:
        """2FSK modulation. bits_flat: [B, Nbits] uint8 -> waveform [B, Nbits*sym_len] float32.
        Run on CPU to avoid CUDA fragmentation.
        """
        bits = bits_flat.to(torch.int64).cpu()
        B, N = bits.shape
        L = N * self.sym_len
        t = torch.arange(self.sym_len, dtype=torch.float32).view(1, -1)
        wave = torch.empty(B, L, dtype=torch.float32)
        w0 = 2.0 * math.pi * self.f0 / self.fs
        w1 = 2.0 * math.pi * self.f1 / self.fs
        s0 = torch.sin(w0 * t)
        s1 = torch.sin(w1 * t)
        for b in range(B):
            segs = []
            for n in range(N):
                segs.append(s1 if bits[b, n] > 0 else s0)
            seg = torch.cat(segs, dim=1)
            wave[b] = (amp * seg).view(-1)
        return wave

    def demodulate(self, wave: torch.Tensor) -> torch.Tensor:
        """Goertzel demod. wave [B, L] -> bits [B, Nbits]. Uses symbol_len. CPU path."""
        x = wave.cpu().to(torch.float32)
        B, L = x.shape
        assert L % self.sym_len == 0
        N = L // self.sym_len
        bits = torch.empty(B, N, dtype=torch.uint8)
        for b in range(B):
            xb = x[b].view(N, self.sym_len)
            p0 = _goertzel_power(xb, self.f0, self.fs)
            p1 = _goertzel_power(xb, self.f1, self.fs)
            bits[b] = (p1 > p0).to(torch.uint8)
        return bits

