# -*- coding: utf-8 -*-
"""Minimal FarGAN-inspired waveform head and loss helpers.

This module provides a lightweight waveform synthesiser that mirrors the key
ideas of FarGAN (period embedding + subframe autoregressive generation), but it
fits directly into the existing AETHER decoder interface.  It also exposes a
compact multi-resolution STFT loss that can be used as an all-in-one waveform
objective.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.feature_spec import get_default_feature_spec


class FarGANWaveHeadLite(nn.Module):
    """FarGAN-like waveform decoder with period embeddings and GRU stack.

    The module expects decoder features shaped ``[B, T, 48]`` (matching the
    current FeatureSpec layout) and returns a waveform tensor of shape
    ``[B, 1, T * hop]``.  Internally it splits each frame into four subframes and
    generates 40 audio samples per subframe (160 samples / 10 ms hop).
    """

    def __init__(
        self,
        d_in: int = 48,
        sample_rate: int = 16000,
        hop: int = 160,
        subframes: int = 4,
        period_min: int = 32,
        period_max: int = 255,
        period_embed_dim: int = 64,
        cond_dim: int = 128,
        gru_dim: int = 192,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.subframes = int(subframes)
        assert self.hop % self.subframes == 0, "hop must be divisible by subframes"
        self.samples_per_subframe = self.hop // self.subframes

        self.period_min = int(period_min)
        self.period_max = int(period_max)
        self.period_bins = self.period_max - self.period_min + 1
        self.period_embed_dim = int(period_embed_dim)
        self.cond_dim = int(cond_dim)
        self.gru_dim = int(gru_dim)

        self.train_noise_scale = 0.02
        self.eval_noise_scale = 0.003

        self.feature_spec = get_default_feature_spec()

        self.period_embed = nn.Embedding(self.period_bins, self.period_embed_dim)

        frame_input_dim = d_in + self.period_embed_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_input_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.cond_dim * self.subframes),
        )

        gru_input_dim = self.cond_dim + self.period_embed_dim + 2  # +sin, +cos
        self.gru = nn.GRU(gru_input_dim, self.gru_dim, num_layers=3, batch_first=True)
        self.out_proj = nn.Linear(self.gru_dim, 1)
        self.log_gain = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()
        self.register_buffer("subframe_steps", torch.arange(self.samples_per_subframe, dtype=torch.float32), persistent=False)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.period_embed.weight, -0.02, 0.02)
        for module in self.frame_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.2)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    def _dnn_pitch_to_period(self, dnn_pitch: torch.Tensor) -> torch.Tensor:
        period = 256.0 * torch.pow(2.0, -(dnn_pitch + 2.0))
        return period.clamp(float(self.period_min), float(self.period_max))

    def _dnn_pitch_to_hz(self, dnn_pitch: torch.Tensor) -> torch.Tensor:
        period = self._dnn_pitch_to_period(dnn_pitch)
        return self.sample_rate / period.clamp(min=1.0)

    # ------------------------------------------------------------------
    def forward(self, feats: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"Expected [B, T, D] features, got {feats.shape}")

        B, T, _ = feats.shape
        device = feats.device

        f0_dnn = self.feature_spec.extract_feature(feats, "f0").squeeze(-1)
        voicing = self.feature_spec.extract_feature(feats, "voicing").squeeze(-1)

        period = self._dnn_pitch_to_period(f0_dnn)
        period_indices = (period.round().long().clamp(self.period_min, self.period_max) - self.period_min)

        hidden = feats.new_zeros(self.gru.num_layers, B, self.gru_dim)
        phase = torch.zeros(B, device=device)

        total_samples = T * self.subframes * self.samples_per_subframe
        samples = []

        for t in range(T):
            feat_t = feats[:, t, :]
            pe_vec = self.period_embed(period_indices[:, t])  # [B, E]
            frame_in = torch.cat([feat_t, pe_vec], dim=-1)
            cond_flat = self.frame_proj(frame_in)  # [B, subframes*cond_dim]
            cond_frame = cond_flat.view(B, self.subframes, self.cond_dim)
            f0_hz_t = self._dnn_pitch_to_hz(f0_dnn[:, t])
            v_gate = voicing[:, t].clamp(0.0, 1.0) 
            delta = 2.0 * math.pi * torch.clamp(f0_hz_t, min=60.0) / self.sample_rate

            for sf in range(self.subframes):
                cond_sf = cond_frame[:, sf, :].unsqueeze(1).expand(-1, self.samples_per_subframe, -1)
                pe_sf = pe_vec.unsqueeze(1).expand(-1, self.samples_per_subframe, -1)
                phase_seq = phase.unsqueeze(-1) + delta.unsqueeze(-1) * v_gate.unsqueeze(-1) * self.subframe_steps.unsqueeze(0)
                sin_seq = torch.sin(phase_seq).unsqueeze(-1)
                cos_seq = torch.cos(phase_seq).unsqueeze(-1)
                gru_in = torch.cat([cond_sf, pe_sf, sin_seq, cos_seq], dim=-1)

                out, hidden = self.gru(gru_in, hidden)
                sample = self.out_proj(out)  # [B,S,1]

                noise_scale = self.train_noise_scale if self.training else self.eval_noise_scale
                noise = torch.randn_like(sample) * noise_scale
                gate = v_gate.view(B, 1, 1)
                sample = gate * sample + (1.0 - gate) * (0.6 * sample + 0.4 * noise)
                samples.append(sample.squeeze(-1))

                phase = phase + delta * v_gate * self.samples_per_subframe
                phase = torch.remainder(phase, 2.0 * math.pi)

        wave = torch.cat(samples, dim=1)  # [B, L]
        #wave = wave * torch.exp(self.log_gain).clamp(max=3.0)
        if self.training:
            with torch.no_grad():
                rms = torch.sqrt((wave.pow(2)).mean(dim=-1, keepdim=True) + 1e-8)
                target = 0.08
                scale = (target / (rms + 1e-8)).clamp(0.6, 1.6)
            wave = wave * scale
        if self.training:
            wave = torch.tanh(wave)
        else:
            wave = torch.clamp(wave, -1.0, 1.0)

        if target_len is not None:
            if wave.size(-1) > target_len:
                wave = wave[..., :target_len]
            elif wave.size(-1) < target_len:
                pad = target_len - wave.size(-1)
                wave = F.pad(wave, (0, pad))

        gain = torch.exp(self.log_gain).clamp(0.5, 1.5)
        wave = 1.1 * torch.tanh(0.9 * gain * wave)
        return wave.unsqueeze(1)  # [B, 1, L]


class FarGANLikeLoss(nn.Module):
    """Compact MR-STFT + L1 loss used by FarGAN-like training."""

    def __init__(
        self,
        fft_sizes: Iterable[int] = (1024, 512, 256),
        hop_sizes: Iterable[int] = (256, 128, 64),
        win_lengths: Iterable[int] = (1024, 512, 256),
        l1_weight: float = 0.03,
        crop_samples: int = 16384,
        sc_floor: float = 1e-2,
        dc_remove: bool = True,
    ) -> None:
        super().__init__()
        self.configs: Tuple[Tuple[int, int, int], ...] = tuple(
            (int(f), int(h), int(w)) for f, h, w in zip(fft_sizes, hop_sizes, win_lengths)
        )
        self.l1_weight = float(l1_weight)
        self.crop_samples = int(crop_samples)
        self.sc_floor = float(sc_floor)
        self.dc_remove = bool(dc_remove)
        self.log_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
        for idx, (_, _, win) in enumerate(self.configs):
            window = torch.hann_window(win)
            self.register_buffer(f"window_{idx}", window, persistent=False)

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        if x.dim() == 2:
            return x
        if x.dim() == 3 and x.size(1) == 1:
            return x[:, 0]
        raise ValueError(f"Expected waveform tensor with shape [B, T] or [B, 1, T], got {x.shape}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_wave = self._ensure_2d(pred).float()
        target_wave = self._ensure_2d(target).float()

        min_len = min(pred_wave.size(-1), target_wave.size(-1))
        if min_len > self.crop_samples > 0:
            start = torch.randint(0, min_len - self.crop_samples + 1, (1,), device=pred_wave.device).item()
            sl = slice(start, start + self.crop_samples)
            pred_wave = pred_wave[..., sl]
            target_wave = target_wave[..., sl]
        else:
            pred_wave = pred_wave[..., :min_len]
            target_wave = target_wave[..., :min_len]

        if self.dc_remove:
            pred_wave = pred_wave - pred_wave.mean(dim=-1, keepdim=True)
            target_wave = target_wave - target_wave.mean(dim=-1, keepdim=True)

        total = pred_wave.new_zeros(())

        for idx, (fft_size, hop_size, win_length) in enumerate(self.configs):
            window = getattr(self, f"window_{idx}")
            pred_spec = torch.stft(
                pred_wave,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            )
            target_spec = torch.stft(
                target_wave,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            )

            mag_pred = pred_spec.abs()
            mag_target = target_spec.abs()

            mag_loss = F.l1_loss(mag_pred, mag_target)

            sc_num = torch.linalg.norm(mag_pred - mag_target, dim=(-2, -1))
            sc_den = torch.linalg.norm(mag_target, dim=(-2, -1)).clamp_min(self.sc_floor)
            sc_loss = (sc_num / sc_den).mean()

            total = total + (mag_loss + sc_loss)

        total = total / len(self.configs)
        total = total + self.l1_weight * F.l1_loss(pred_wave, target_wave)
        return total


__all__ = ["FarGANWaveHeadLite", "FarGANLikeLoss"]
