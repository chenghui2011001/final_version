# -*- coding: utf-8 -*-
"""
High level encoder/decoder assembly for the AETHER speech codec.
数值稳健加强版：
- 编码/解码前后均做 NaN/Inf 清洗
- F0+Voicing 专用分支，解码侧 tanh/sigmoid + 夹紧 + 清洗
- 保留 MoE/FiLM 与双流主干
- 解码输出回写到主特征前做双重兜底，避免污染
"""

from __future__ import annotations

from typing import Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容相对/绝对导入
try:
    from .decoder_sculptor import ConvRefineDecoder
    from .gla_block import GLABackbone, GLABlock
    from .kan_field import KANLiteFiLM
    # Use CompatibleMicroMoE wrapper to specialized MoE (dual routing)
    from .enhanced_moe import CompatibleMicroMoE as MicroMoE
    from .utils import (
        build_csi_vec,
        global_pool,
        straight_through_latent_quantizer,
        estimated_bitrate,
    )
except ImportError:  # pragma: no cover
    from decoder_sculptor import ConvRefineDecoder
    from gla_block import GLABackbone, GLABlock
    from kan_field import KANLiteFiLM
    from enhanced_moe import CompatibleMicroMoE as MicroMoE
    from utils import (
        build_csi_vec,
        global_pool,
        straight_through_latent_quantizer,
        estimated_bitrate,
    )

try:
    # Prefer absolute import from final_version root
    from utils.feature_spec import get_default_feature_spec
except Exception:  # pragma: no cover
    # Fallback to package-relative import when running as a module
    from ..utils.feature_spec import get_default_feature_spec


# ------------------------------
# DualStream Backbone
# ------------------------------
class DualStreamBackbone(nn.Module):
    """Two-stream Transformer backbone capturing coarse and fine temporal cues."""
    def __init__(self, d_model: int, gla_depth: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ribbon_down = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3, padding=1)
        self.ribbon = GLABackbone(
            d_model=d_model, depth=gla_depth, n_heads=n_heads, dropout=dropout,
            local_kernel=5, local_dilation=4,
        )
        self.thread = GLABackbone(
            d_model=d_model, depth=gla_depth, n_heads=n_heads, dropout=dropout,
            local_kernel=3, local_dilation=1,
        )
        self.mix = nn.Linear(d_model * 2, d_model)

    @staticmethod
    def _downsample_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.unsqueeze(1)  # [B,1,T]
        pooled = F.max_pool1d(mask, kernel_size=3, stride=3, padding=1)
        return (pooled.squeeze(1) > 0.5).to(mask.dtype)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ribbon_mask = self._downsample_mask(attn_mask) if attn_mask is not None else None
        r = self.ribbon_down(x.transpose(1, 2)).transpose(1, 2)
        r = self.ribbon(r, ribbon_mask)
        r = F.interpolate(r.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False).transpose(1, 2)

        t = self.thread(x, attn_mask)
        fused = torch.cat([r, t], dim=-1)
        mixed = self.mix(fused)

        stream_logs = {
            "ribbon_energy": r.pow(2).mean(dim=(1, 2)),
            "thread_energy": t.pow(2).mean(dim=(1, 2)),
            # Expose streams for MoE acoustic-aware routing
            "ribbon_stream": r,
            "thread_stream": t,
            "fused_stream": mixed,
        }
        return mixed, stream_logs


# ------------------------------
# 轻量 OLA 合成器（可选）
# ------------------------------
# ------------------------------
# 轻量 OLA 合成器（可选）
# ------------------------------
class FARGANSynthWrapper(nn.Module):
    """
    FARGAN声码器的包装器，提供标准合成接口
    """
    def __init__(self, fargan_core):
        super().__init__()
        self.fargan_core = fargan_core
        self.sample_rate = 16000
        self.sr = 16000

        # 周期估计器：从特征中估计周期
        self.period_estimator = nn.Sequential(
            nn.Linear(20, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        features: torch.Tensor,
        target_len: int = None,
        **kwargs
    ) -> torch.Tensor:
        """
        使用FARGAN合成波形

        Args:
            features: 特征张量 [B, T, feature_dim]
            target_len: 目标波形长度

        Returns:
            wave: 合成的波形 [B, T_wave]
        """
        B, T, feature_dim = features.shape

        # 只使用前20维特征（FARGAN标准）
        fargan_features = features[..., :20]  # [B, T, 20]

        # 估计周期
        period_raw = self.period_estimator(fargan_features)  # [B, T, 1]
        period = torch.round(period_raw.squeeze(-1)).clamp(32, 255).long()  # [B, T]

        # Stage3兼容的FARGAN调用策略：特征+4帧冗余，目标输出=实际长度
        min_len = min(fargan_features.size(1), period.size(1))

        # 确保有足够的特征长度（至少5帧，与Stage3对齐）
        if min_len < 5:
            print(f"[WARNING] FARGAN synthesis failed: min_len={min_len} < 5, using zero padding")
            wave_len = target_len if target_len is not None else min_len * 160
            wave = torch.zeros(B, wave_len, device=features.device, dtype=features.dtype)
            return wave

        # 计算目标帧数：需要确保有足够的冗余特征
        # FARGAN需要nb_frames+4帧特征来输出nb_frames帧音频
        if min_len < 9:  # 至少需要5+4=9帧才能输出5帧
            print(f"[WARNING] FARGAN synthesis failed: min_len={min_len} < 9, using zero padding")
            wave_len = target_len if target_len is not None else min_len * 160
            wave = torch.zeros(B, wave_len, device=features.device, dtype=features.dtype)
            return wave

        # 计算可输出的最大帧数（考虑+4冗余需求）
        max_output_frames = min_len - 4
        nb_frames = (max_output_frames // 5) * 5  # 5帧对齐

        if nb_frames < 5:
            print(f"[WARNING] FARGAN synthesis failed: nb_frames={nb_frames} < 5, using zero padding")
            wave_len = target_len if target_len is not None else nb_frames * 160
            wave = torch.zeros(B, wave_len, device=features.device, dtype=features.dtype)
            return wave

        # 准备FARGAN输入：特征和周期都给nb_frames+4帧（确保有足够冗余）
        input_len = nb_frames + 4
        aligned_features = fargan_features[:, :input_len, :].contiguous()  # [B, input_len, 20]
        aligned_period = period[:, :input_len].clamp(32, 255).long().contiguous()  # [B, input_len]

        # 使用FARGAN合成，输出目标帧数
        try:
            wave, _ = self.fargan_core(aligned_features, aligned_period, nb_frames=nb_frames)
        except Exception as e:
            # 如果FARGAN合成失败，使用零填充
            print(f"[WARNING] FARGAN synthesis failed: {e}, using zero padding")
            wave_len = target_len if target_len is not None else nb_frames * 160
            wave = torch.zeros(B, wave_len, device=features.device, dtype=features.dtype)
            return wave

        # 长度调整
        if target_len is not None:
            current_len = wave.size(-1)
            if current_len > target_len:
                wave = wave[..., :target_len]
            elif current_len < target_len:
                # 零填充
                pad_len = target_len - current_len
                wave = F.pad(wave, (0, pad_len), mode='constant', value=0.0)

        return wave

# ------------------------------
# SimpleOLASynth class removed - using FARGAN synthesis only

# ------------------------------
# Encoder
# ------------------------------
class AETHEREncoder(nn.Module):
    """Encoder: x[B,T,d_in] + CSI -> latent z[B,T,dz] with optional FiLM+MoE."""
    def __init__(
        self,
        d_in: int = 36,
        d_model: int = 128,
        dz: int = 24,
        gla_depth: int = 2,
        n_heads: int = 2,
        d_csi: int = 10,
        dropout: float = 0.0,
        use_film: bool = True,
        use_moe: bool = True,
        n_experts: int = 4,
        top_k: int = 2,
        latent_bits: int = 4,
        frame_rate_hz: float = 100.0,
        quantize_latent: bool = True,
        feature_spec_type: str = "fargan",  # "fargan" or "aether"
    ) -> None:
        super().__init__()
        self.d_csi = d_csi
        self.d_in = d_in
        self.latent_bits = latent_bits
        self.quantize_latent = quantize_latent and latent_bits > 0
        self.frame_rate_hz = frame_rate_hz
        self.bits_per_frame, self.kbps = estimated_bitrate(dz, max(latent_bits, 1), frame_rate_hz)
        self.prior_dim = 6
        self.feature_spec_type = feature_spec_type

        self.in_proj = nn.Linear(d_in, d_model)
        self.film = KANLiteFiLM(d_csi=d_csi, d_feat=d_model, time_dependent=False) if use_film else None

        # 根据特征规范类型设置不同的条件投影
        if feature_spec_type == "fargan":
            from .feature_adapter import get_fargan_feature_spec
            self.spec = get_fargan_feature_spec()
            self._crit_names = self.spec.get_perceptually_critical_features()
            # 对于FARGAN，使用前18维倒谱作为关键特征
            crit_in = 18  # ceps维度
        else:
            from utils.feature_spec import get_default_feature_spec
            self.spec = get_default_feature_spec()
            self._crit_names = self.spec.get_perceptually_critical_features()
            def _dim(sl): return sl.stop - sl.start
            crit_in = sum(_dim(self.spec.get_feature_slice(n)) for n in self._crit_names)

        self.cond_proj = nn.Sequential(nn.Linear(crit_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))


        self.backbone = DualStreamBackbone(d_model=d_model, gla_depth=gla_depth, n_heads=n_heads, dropout=dropout)
        # Specialized dual-routing MoE (sample-level acoustic+CSI and optional token-level routing)
        self.moe = (
            MicroMoE(
                d_model=d_model,
                n_experts=n_experts,
                top_k=top_k,
                d_csi=d_csi,
                router_use_csi=True,
                expert_dropout=0.1,
                balance_weight=0.5,
                use_token_level=True,
            ) if use_moe else None
        )
        self.out_proj = nn.Linear(d_model, dz)
        self.base_top_k = top_k
        self.latent_dim = dz

        self.base_use_film = use_film
        self.base_use_moe = use_moe
        self.active_stage = "C"
        self.active_use_film = use_film
        self.active_use_moe = use_moe
        self.film_ratio = 1.0
        self.film_beta_scale = 1.0
        self.register_buffer("film_perm", torch.randperm(d_model))

        self._safe_init_weights()

    def _safe_init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        for module in self.cond_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_stage(self, stage: str, inference: bool = False) -> None:
        self.active_stage = stage
        self.active_use_film = self.base_use_film and stage in {"B", "C"}
        self.active_use_moe = self.base_use_moe
        if self.moe is not None:
            self.moe.top_k = 1 if inference else self.base_top_k

    def set_film_activation(self, ratio: float, beta_scale: float = 1.0):
        self.film_ratio = float(max(0.0, min(1.0, ratio)))
        self.film_beta_scale = float(max(0.0, beta_scale))

    def forward(
        self,
        x: torch.Tensor,
        csi_dict: Dict[str, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, _ = x.shape
        self.set_stage(self.active_stage, inference=inference)

        # 根据特征类型提取条件特征
        if self.feature_spec_type == "fargan":
            # 对于FARGAN特征，使用前18维倒谱作为条件
            crit = x[..., :18]  # ceps特征
        else:
            # 对于AETHER特征，使用原有的关键特征
            crit = torch.cat([x[..., self.spec.get_feature_slice(n)] for n in self._crit_names], dim=-1)

        cond = self.cond_proj(crit)

        # CSI: use unified 10-dim layout (snr_norm + fading_onehot(8) + ber)
        csi_vec = build_csi_vec(csi_dict, target_dim=self.d_csi)
        # C. NaN防御：对解码端FiLM的CSI进行显式清洗
        csi_vec = torch.nan_to_num(csi_vec, nan=0.0, posinf=0.0, neginf=0.0)
        # C. NaN防御：对FiLM前的CSI进行显式清洗（NaN/Inf→0）
        csi_vec = torch.nan_to_num(csi_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if csi_vec.size(0) != b:
            csi_vec = csi_vec.expand(b, -1)

        x_safe = torch.clamp(x, -10.0, 10.0)
        x_safe = torch.nan_to_num(x_safe, nan=0.0, posinf=1.0, neginf=-1.0)

        h = self.in_proj(x_safe) + cond

        logs: Dict[str, torch.Tensor] = {}
        if csi_dict is not None and "snr_db" in csi_dict:
            logs["snr_db"] = csi_dict["snr_db"]

        alpha = beta = None
        if self.active_use_film and self.film is not None:
            T, D = h.shape[1], h.shape[2]
            alpha, beta = self.film(csi_vec, T=T)
            # 记录非有限计数，并清洗 a/b 数值，便于后续可观测
            try:
                nan_count_a = int((~torch.isfinite(alpha)).sum().detach().cpu().item())
                nan_count_b = int((~torch.isfinite(beta)).sum().detach().cpu().item())
            except Exception:
                nan_count_a = 0
                nan_count_b = 0
            alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1.0, neginf=1.0)
            beta  = torch.nan_to_num(beta,  nan=0.0, posinf=0.0, neginf=0.0)
            k = int(round(self.film_ratio * D))
            if k > 0:
                perm = self.film_perm
                mask = torch.zeros(D, device=h.device, dtype=h.dtype)
                mask.scatter_(0, perm[:k], 1.0)
                mask = mask.view(1, 1, D).expand(h.size(0), T, D)
            else:
                mask = torch.zeros_like(h)
            alpha_eff = 1.0 + (alpha - 1.0) * mask * self.film_ratio
            beta_eff = beta * mask * (self.film_ratio * self.film_beta_scale)
            # Respect optional clamp ranges from FiLM (scale clamp applies to gamma-1)
            try:
                s_lo = getattr(self.film, '_clamp_scale_lo', None)
                s_hi = getattr(self.film, '_clamp_scale_hi', None)
                sh_lo = getattr(self.film, '_clamp_shift_lo', None)
                sh_hi = getattr(self.film, '_clamp_shift_hi', None)
                if isinstance(s_lo, float) and isinstance(s_hi, float):
                    alpha_eff = alpha_eff.clamp_(1.0 + s_lo, 1.0 + s_hi)
                if isinstance(sh_lo, float) and isinstance(sh_hi, float):
                    beta_eff = beta_eff.clamp_(sh_lo, sh_hi)
            except Exception:
                pass
            h = alpha_eff * h + beta_eff
            # 轻量FiLM诊断（可观测统计）
            try:
                self._last_film_stats = {
                    'a_mean': float(alpha.detach().mean().item()),
                    'b_mean': float(beta.detach().mean().item()),
                    'scale_mean': float(alpha_eff.detach().mean().item()),
                    'shift_mean': float(beta_eff.detach().mean().item()),
                    'nan_count_a': int(nan_count_a),
                    'nan_count_b': int(nan_count_b),
                }
            except Exception:
                self._last_film_stats = {'nan_count_a': int(nan_count_a), 'nan_count_b': int(nan_count_b)}

        h, stream_logs = self.backbone(h, attn_mask)
        logs.update(stream_logs)

        if self.active_use_moe and self.moe is not None:
            pooled = global_pool(h)
            # Optional acoustic priors (6-dim); keep CSI at tail for parser compatibility
            try:
                from .utils import extract_acoustic_priors
                priors = extract_acoustic_priors(x)  # [B,6]
            except Exception:
                priors = torch.zeros(pooled.size(0), 6, device=pooled.device, dtype=pooled.dtype)
            router_in = torch.cat([pooled, priors, csi_vec], dim=-1)

            # Provide raw and dual streams for full acoustic-aware routing
            ribbon_stream = stream_logs.get('ribbon_stream', h)
            thread_stream = stream_logs.get('thread_stream', h)
            dual_streams = {
                'ribbon_stream': ribbon_stream,
                'thread_stream': thread_stream,
                'fused_stream': h,
            }

            h_moe, aux = self.moe(h, router_in, x_raw=x, dual_streams=dual_streams)
            h = h + h_moe  # Residual fusion

            # Normalize aux keys for training pipeline
            if isinstance(aux, dict):
                if 'balance_loss' in aux:
                    logs['moe_balance_loss'] = aux['balance_loss']
                if 'moe_balance_loss' in aux:
                    logs['moe_balance_loss'] = aux['moe_balance_loss']
                if 'token_balance_loss' in aux:
                    logs['moe_token_balance_loss'] = aux['token_balance_loss']
                if 'moe_token_balance_loss' in aux:
                    logs['moe_token_balance_loss'] = aux['moe_token_balance_loss']
                # Metrics passthrough
                for k in ('expert_entropy', 'min_expert_usage'):
                    if k in aux:
                        logs[f"moe_{k}"] = aux[k]

        # 简化的latent生成，移除F0专用分支
        z_cont = self.out_proj(h)  # [B,T,dz]
        logs["latent_norm"] = z_cont.norm(dim=-1).mean()

        if self.quantize_latent:
            z_quant = straight_through_latent_quantizer(z_cont, bits=self.latent_bits)
        else:
            z_quant = z_cont

        logs["latent_continuous"] = z_cont
        logs["bits_per_frame_nominal"] = torch.tensor(self.bits_per_frame, device=z_cont.device)
        logs["kbps_nominal"] = torch.tensor(self.kbps, device=z_cont.device)
        if alpha is not None:
            logs["alpha_mean"] = alpha.mean(dim=(1, 2))
            logs["beta_mean"] = beta.mean(dim=(1, 2))
        return z_quant, logs


# ------------------------------
# Decoder
# ------------------------------
class AETHERDecoder(nn.Module):
    """Decoder: latent + CSI -> reconstructed features."""
    def __init__(
        self,
        dz: int = 24,
        d_out: int = 36,
        d_hidden: int = 128,
        d_csi: int = 10,
        decoder_heads: int = 2,
        enable_synth: bool = False,
        feature_spec_type: str = "fargan",  # "fargan" or "aether"
        use_film: bool = True,  # 新增：可选禁用FiLM
    ) -> None:
        super().__init__()
        self.d_csi = d_csi
        self.d_out = d_out
        self.enable_synth = bool(enable_synth)
        self.dz = dz
        self.feature_spec_type = feature_spec_type

        heads = max(1, min(decoder_heads, dz))
        self.global_recompose = GLABlock(d_model=dz, n_heads=heads, dropout=0.0)
        self.refiner = ConvRefineDecoder(d_z=dz, d_out=d_out, d_hidden=d_hidden, csi_dim=d_csi, use_film=use_film)

        # 根据特征类型设置不同的规范
        if feature_spec_type == "fargan":
            from .feature_adapter import get_fargan_feature_spec
            self.spec = get_fargan_feature_spec()
        else:
            from utils.feature_spec import get_default_feature_spec
            self.spec = get_default_feature_spec()

        if self.enable_synth:
            # 根据特征类型选择合适的合成器
            if feature_spec_type == "fargan" and d_out >= 20:
                # 使用真正的FARGAN声码器
                from .fargan_components import FARGANCore
                self.fargan_core = FARGANCore(
                    subframe_size=40,
                    nb_subframes=4,
                    feature_dim=min(20, d_out),  # FARGAN最多使用20维特征
                    cond_size=256,
                )
                # 包装为synth接口以保持兼容性
                self.synth = FARGANSynthWrapper(self.fargan_core)
            else:
                # 对于其他情况，禁用合成器（只使用FARGAN）
                self.synth = None
                print(f"[WARNING] 非FARGAN特征类型 ({feature_spec_type}) 或特征维度不足 (<20), 合成器已禁用")


    def _forward_features(self, z: torch.Tensor, csi_dict: Dict[str, torch.Tensor], attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, _, _ = z.shape
        # CSI: unified 10-dim vector
        csi_vec = build_csi_vec(csi_dict, target_dim=self.d_csi)
        # Ensure shape [B, d_csi] even when no CSI is provided
        if csi_vec.dim() == 1:
            csi_vec = csi_vec.unsqueeze(0)
        if csi_vec.size(0) != b:
            # Safely expand a single row to batch; otherwise truncate to B
            if csi_vec.size(0) == 1:
                csi_vec = csi_vec.expand(b, -1)
            else:
                csi_vec = csi_vec[:b]

        # 简化的解码路径，移除F0专用分支
        z_global = self.global_recompose(z, attn_mask)
        features = self.refiner(z_global, csi_vec)
        # Avoid collapsing variance by zeroing inf; only sanitize NaNs/Infs to finite range
        features = torch.nan_to_num(features)

        return features

    def forward(self, z: torch.Tensor, csi_dict: Dict[str, torch.Tensor], attn_mask: torch.Tensor | None = None,
                return_wave: bool = False, target_len: int = None):
        y = self._forward_features(z, csi_dict, attn_mask)
        if return_wave and self.enable_synth:
            wav = self.synth(y, target_len=target_len)
            return y, wav
        return y


__all__ = ["AETHEREncoder", "AETHERDecoder"]
