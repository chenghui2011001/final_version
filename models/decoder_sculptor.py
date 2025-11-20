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
    """

    def __init__(
        self,
        d_z: int,
        d_out: int = 48,
        d_hidden: int = 128,
        csi_dim: int = 32,
        use_film: bool = True,  # 新增：可选禁用FiLM
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
        # FiLM scaling knobs (functional symmetry: decoder侧以增益为主，偏置为辅)
        self.film_gain_scale: float = 1.0
        self.film_bias_scale: float = 0.0  # 默认不使用偏置，保持“pre γ 为主”
        # Optional inverse mode: encourage CSI removal (gamma -> 2-alpha, beta sign -> negative)
        self.film_invert: bool = False

        # 帧级先验（Acoustic Priors）→ 时变FiLM增强（最小入侵式）
        # 从时间维提炼轻量先验，映射为 per-frame α/β，再与 KANLiteFiLM 输出融合
        self.use_acoustic_priors: bool = True
        self.ap_channels: int = 16
        self.ap_blend: float = 0.5  # 0..1，越大越依赖帧级先验
        self.ap_conv_t = nn.Conv1d(d_hidden, self.ap_channels, kernel_size=3, padding=1)
        self.ap_conv_out_alpha = nn.Conv1d(self.ap_channels, d_hidden, 1)
        self.ap_conv_out_beta = nn.Conv1d(self.ap_channels, d_hidden, 1)

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

        h = self.in_proj(z.transpose(1, 2))  # [B, d_hidden, T]
        h_in = h  # 保存pre-FiLM特征用于帧级先验提炼

        # 可选的FiLM调制（仅在use_film=True时执行）
        if self.use_film and self.film is not None:
            # 生成 FiLM 参数（理想形状 [B, T, d_hidden]）
            alpha, beta = self.film(csi_vec, T=t)

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

            # FiLM 调制（decoder侧：以增益为主，偏置缩放）
            if self.film_invert:
                gamma = (2.0 - alpha_cf)
                bias = -abs(self.film_bias_scale) * beta_cf
                h = (1.0 + self.film_gain_scale * (gamma - 1.0)) * h + bias
            else:
                h = (1.0 + self.film_gain_scale * (alpha_cf - 1.0)) * h + self.film_bias_scale * beta_cf

        # 后续保持不变
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)
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
