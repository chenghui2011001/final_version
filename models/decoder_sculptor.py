# -*- coding: utf-8 -*-
"""
Decoder refinement stack that sculpts local detail under CSI aware FiLM control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_field import KANLiteFiLM


class GatedConv1DBlock(nn.Module):
    """
    Lightweight gated 1D convolution block.
    """

    def __init__(self, d_in: int, d_out: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(d_in, d_out * 2, kernel_size, padding=padding, dilation=dilation)
        self.proj = nn.Conv1d(d_out, d_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        a, g = y.chunk(2, dim=1)
        y = torch.tanh(a) * torch.sigmoid(g)
        return self.proj(y)


class ConvRefineDecoder(nn.Module):
    """
    Local sculpting module for the decoder with CSI driven FiLM modulation.

    该模块原本只是在卷积栈入口处做一次固定强度的FiLM，CLI 提供的
    `--dec-film-*` 选项也不会真正影响forward。为了和编码端形成对称，
    这里实现了可配置的“pre/post”双段FiLM、训练步控制以及诊断输出。
    """

    def __init__(
        self,
        d_z: int,
        d_out: int = 48,
        d_hidden: int = 128,
        csi_dim: int = 32,
        use_film: bool = True,  # 新增：可选禁用FiLM
        # FiLM对信道劣化的额外敏感度（0~1，越大表示坏信道下FiLM越强）
        film_channel_gain: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_film = use_film
        self.in_proj = nn.Conv1d(d_z, d_hidden, 1)
        self.block1 = GatedConv1DBlock(d_hidden, d_hidden, kernel_size=5, dilation=1)
        self.block2 = GatedConv1DBlock(d_hidden, d_hidden, kernel_size=5, dilation=2)
        self.block3 = GatedConv1DBlock(d_hidden, d_hidden, kernel_size=5, dilation=4)
        # Replace single head with three-head design: mu, eps, logstd
        self.out_mu = nn.Conv1d(d_hidden, d_out, 1)
        self.out_eps = nn.Conv1d(d_hidden, d_out, 1)
        self.out_logstd = nn.Conv1d(d_hidden, d_out, 1)
        # Std floor to prevent variance collapse; tunable via attribute
        self.std_floor: float = 0.2
        self.film = KANLiteFiLM(d_csi=csi_dim, d_feat=d_hidden, time_dependent=True) if use_film else None
        # FiLM scaling knobs，和编码端保持功能对称
        self.film_gain_scale: float = 1.0
        self.film_bias_scale: float = 0.0
        self.film_invert: bool = False
        self.film_channel_gain: float = float(max(0.0, min(1.0, film_channel_gain)))

        # 为了支持“pre/post” 双阶段调度，引入 ramp 配置
        self.film_pre_start: float = 1.0
        self.film_pre_end: float = 1.0
        self.film_pre_warmup: int = 0
        self.film_post_start: float = 0.0
        self.film_post_end: float = 0.0
        self.film_post_warmup: int = 0

        # 帧级先验（Acoustic Priors）驱动的FiLM增强
        self.use_acoustic_priors: bool = True
        self.ap_channels: int = 16
        self.ap_blend: float = 0.5  # 0..1 越大越依赖先验
        self.ap_conv_t = nn.Conv1d(d_hidden, self.ap_channels, kernel_size=3, padding=1)
        self.ap_conv_out_alpha = nn.Conv1d(self.ap_channels, d_hidden, 1)
        self.ap_conv_out_beta = nn.Conv1d(self.ap_channels, d_hidden, 1)

        # 训练步/诊断缓存
        self._film_step: int | None = None
        self._last_film_stats: dict[str, float] | None = None
        # 可由上游训练脚本传入的batch级JSCC“坏度”指标（0~1）
        self._jscc_bad: float | None = None

    @staticmethod
    def _ramp(step: int | None, start: float, end: float, warmup: int) -> float:
        if step is None or warmup <= 0:
            return end
        ratio = max(0.0, min(1.0, float(step) / float(warmup)))
        return start + (end - start) * ratio

    def configure_film_schedule(
        self,
        pre_start: float = 1.0,
        pre_end: float = 1.0,
        pre_warmup: int = 0,
        post_start: float = 0.0,
        post_end: float = 0.0,
        post_warmup: int = 0,
    ) -> None:
        """Configure decoder FiLM gain/bias schedules (对称于编码端)."""
        self.film_pre_start = float(pre_start)
        self.film_pre_end = float(pre_end)
        self.film_pre_warmup = int(max(0, pre_warmup))
        self.film_post_start = float(post_start)
        self.film_post_end = float(post_end)
        self.film_post_warmup = int(max(0, post_warmup))

    def set_training_step(self, step: int) -> None:
        self._film_step = int(step)

    def get_film_stats(self) -> dict[str, float] | None:
        return self._last_film_stats

    def forward(self, z: torch.Tensor, csi_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor ``[B, T, d_z]``.
            csi_vec: Tensor ``[B, d_csi]``.
        """
        b, t, _ = z.shape
        device = z.device

        # Ensure all modules are on the same device as input tensors
        if self.in_proj.weight.device != device:
            self.to(device)

        # 确保 csi_vec 是 [B, d_csi] 并在正确设备上
        if csi_vec.dim() == 1:
            csi_vec = csi_vec.unsqueeze(0)  # [1, d_csi]
        if csi_vec.device != device:
            csi_vec = csi_vec.to(device)
        if csi_vec.size(0) != b:
            if csi_vec.size(0) == 1:
                csi_vec = csi_vec.expand(b, -1)  # 广播到 B
            else:
                csi_vec = csi_vec[:b]  # 截断到 B
        # C. NaN防御：对解码端FiLM的CSI进行显式清洗
        csi_vec = torch.nan_to_num(csi_vec, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.in_proj(z.transpose(1, 2))  # [B, d_hidden, T]
        h_in = h  # 保存pre-FiLM特征用于帧级先验提炼
        pre_strength = self._ramp(self._film_step, self.film_pre_start, self.film_pre_end, self.film_pre_warmup)
        post_strength = self._ramp(self._film_step, self.film_post_start, self.film_post_end, self.film_post_warmup)

        # Day3: 让解码端FiLM对CSI“坏度”更敏感。
        # 训练脚本会在启用随机CSI后，把 batch 级 badness 写入 _jscc_bad（0~1）。
        # 这里按 encoder 同样的规则做温和放大：pre/post_strength *= 1 + film_channel_gain * bad。
        if self._jscc_bad is not None:
            try:
                bad = float(self._jscc_bad)
                bad = max(0.0, min(1.0, bad))
                gain = 1.0 + self.film_channel_gain * bad
                pre_strength *= gain
                post_strength *= gain
            except Exception:
                pass

        # 可选的FiLM调制（仅在use_film=True时执行）
        if self.use_film and self.film is not None:
            # 生成 FiLM 参数（理想形状 [B, T, d_hidden]）
            alpha, beta = self.film(csi_vec, T=t)
            # 统计非有限计数并进行清洗，确保后续数值稳定
            try:
                nan_count_alpha = int((~torch.isfinite(alpha)).sum().detach().cpu().item())
                nan_count_beta  = int((~torch.isfinite(beta)).sum().detach().cpu().item())
            except Exception:
                nan_count_alpha = 0
                nan_count_beta = 0
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            beta  = torch.nan_to_num(beta,  nan=0.0, posinf=0.0, neginf=0.0)

            # —— 新增：强制把 2D/维度不匹配的情况修正为 3D [B, T, d_hidden] —— #
            if alpha.dim() == 2:           # [B, d_hidden] -> [B, T, d_hidden]
                alpha = alpha.unsqueeze(1).expand(-1, t, -1)
            if beta.dim() == 2:
                beta = beta.unsqueeze(1).expand(-1, t, -1)

            # 卷积是通道优先，转成 [B, d_hidden, T]
            alpha_cf = alpha.permute(0, 2, 1).contiguous()
            beta_cf  = beta.permute(0, 2, 1).contiguous()

            # 帧级先验：从 h_in 提炼 per-frame 表示，映射到 α/β 并与 KAN 输出融合
            if self.use_acoustic_priors:
                ap = torch.tanh(self.ap_conv_t(h_in))                   # [B, C_ap, T]
                ap_alpha_raw = self.ap_conv_out_alpha(ap)               # [B, d_hidden, T]
                ap_beta_raw  = self.ap_conv_out_beta(ap)                # [B, d_hidden, T]
                ap_alpha = torch.sigmoid(ap_alpha_raw)                  # (0,1)
                ap_beta  = torch.tanh(ap_beta_raw)                      # (-1,1)
                w = float(max(0.0, min(1.0, self.ap_blend)))
                alpha_cf = (1.0 - w) * alpha_cf + w * ap_alpha
                beta_cf  = (1.0 - w) * beta_cf  + w * ap_beta

            def _apply_film(x: torch.Tensor, strength: float, stage: str) -> torch.Tensor:
                if strength <= 0.0:
                    return x
                scale = strength * self.film_gain_scale
                bias_scale = strength * self.film_bias_scale
                # Compute multiplicative gamma and additive bias
                if self.film_invert:
                    gamma = (2.0 - alpha_cf)
                    bias = -abs(bias_scale) * beta_cf
                else:
                    gamma = alpha_cf
                    bias = bias_scale * beta_cf
                gamma = (1.0 + scale * (gamma - 1.0))
                # Optional clamp from FiLM module if provided via set_clamp
                try:
                    s_lo = getattr(self.film, '_clamp_scale_lo', None)
                    s_hi = getattr(self.film, '_clamp_scale_hi', None)
                    sh_lo = getattr(self.film, '_clamp_shift_lo', None)
                    sh_hi = getattr(self.film, '_clamp_shift_hi', None)
                    if isinstance(s_lo, float) and isinstance(s_hi, float):
                        gamma = gamma.clamp_(1.0 + s_lo, 1.0 + s_hi)
                    if isinstance(sh_lo, float) and isinstance(sh_hi, float):
                        bias = bias.clamp_(sh_lo, sh_hi)
                except Exception:
                    pass
                out = gamma * x + bias
                return out

            h = _apply_film(h, pre_strength, 'pre')

        # 后续保持不变
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)

        if self.use_film and self.film is not None:
            h = _apply_film(h, post_strength, 'post')
            # 统计量：alpha/beta与最终scale/bias的均值，便于可视化
            try:
                a_mean = float(alpha_cf.mean().item())
                b_mean = float(beta_cf.mean().item())
                nan_ratio_alpha = float(max(0.0, 1.0 - torch.isfinite(alpha_cf).float().mean().item()))
                nan_ratio_beta  = float(max(0.0, 1.0 - torch.isfinite(beta_cf).float().mean().item()))
            except Exception:
                a_mean = 0.0
                b_mean = 0.0
                nan_ratio_alpha = 0.0
                nan_ratio_beta = 0.0
            self._last_film_stats = {
                'pre': float(pre_strength),
                'post': float(post_strength),
                'gain_scale': float(self.film_gain_scale),
                'bias_scale': float(self.film_bias_scale),
                'invert': bool(self.film_invert),
                'ap_blend': float(self.ap_blend if self.use_acoustic_priors else 0.0),
                'a_mean': a_mean,
                'b_mean': b_mean,
                'nan_ratio_alpha': nan_ratio_alpha,
                'nan_ratio_beta': nan_ratio_beta,
                'nan_count_alpha': int(nan_count_alpha),
                'nan_count_beta': int(nan_count_beta),
            }
        else:
            self._last_film_stats = None
        # Three-head synthesis with Stage3 compatibility mode
        mu = self.out_mu(h)                         # [B, d_out, T]

        # Check if we're in Stage3 compatibility mode (revival period)
        use_stage3_compat = getattr(self, '_stage3_compat_mode', False)

        if use_stage3_compat:
            # Stage3 compatibility: direct output without stochastic components
            y_hat = mu.transpose(1, 2)  # [B, T, d_out] - pure deterministic like Stage3
        else:
            # Stage4 normal: stochastic three-head synthesis
            eps = torch.tanh(self.out_eps(h))           # [-1,1]
            # Softplus for positive std, add floor; compute logstd for supervision
            std = torch.nn.functional.softplus(self.out_logstd(h)) + self.std_floor
            y_hat = (mu + std * eps).transpose(1, 2)    # [B, T, d_out]

        # Cache for loss/debug without breaking interfaces
        self._last_mu = mu.transpose(1, 2).detach()
        if not use_stage3_compat:
            self._last_logstd = std.log().transpose(1, 2).detach()
        else:
            # In Stage3 compat mode, no std is computed
            self._last_logstd = torch.zeros_like(mu).transpose(1, 2).detach()

        return y_hat



__all__ = ["ConvRefineDecoder", "GatedConv1DBlock"]
