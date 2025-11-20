# -*- coding: utf-8 -*-
"""
Simplified MicroMoE: basic sample-level routing aligned with jiagou.md.

Core Features:
- Sample-level top-k routing (not token-level)
- Simple load balancing with uniform prior
- Expert dropout for regularization
- Compatible with CSI conditioning
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import global_pool
except ImportError:  # pragma: no cover
    from utils import global_pool


# --- 强化后的 RobustLSTM：出现非数也会回退到FP32 ---
class RobustLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        self.force_fp32 = False

    def forward(self, x, *args, **kwargs):
        x = x.contiguous()
        orig = x.dtype

        def _run(inp):
            out, hid = self.lstm(inp, *args, **kwargs)
            return out, hid

        try:
            if self.force_fp32 and orig in (torch.bfloat16, torch.float16):
                out, hid = _run(x.float())
            else:
                out, hid = _run(x)
            # 关键：产出后数值自检
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError("non-finite in LSTM output")
        except RuntimeError:
            if orig in (torch.bfloat16, torch.float16):
                self.force_fp32 = True
                out, hid = _run(x.float())
            else:
                raise

            # 再做一次自检（极端保险）
            if torch.isnan(out).any() or torch.isinf(out).any():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # 回到原dtype
        if isinstance(hid, tuple):
            hid = tuple(h.to(orig) for h in hid)
        else:
            hid = hid.to(orig)
        return out.to(orig), hid


# --- 新增 RobustGRU：同样的策略 ---
class RobustGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)
        self.force_fp32 = False

    def forward(self, x, *args, **kwargs):
        x = x.contiguous()
        orig = x.dtype

        def _run(inp):
            out, hid = self.gru(inp, *args, **kwargs)
            return out, hid

        try:
            if self.force_fp32 and orig in (torch.bfloat16, torch.float16):
                out, hid = _run(x.float())
            else:
                out, hid = _run(x)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError("non-finite in GRU output")
        except RuntimeError:
            if orig in (torch.bfloat16, torch.float16):
                self.force_fp32 = True
                out, hid = _run(x.float())
            else:
                raise
            if torch.isnan(out).any() or torch.isinf(out).any():
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if isinstance(hid, tuple):
            hid = tuple(h.to(orig) for h in hid)
        else:
            hid = hid.to(orig)
        return out.to(orig), hid


class AcousticFeatureExtractor(nn.Module):
    """三层特征提取器 - 利用原始声学、Ribbon语义、Thread微音段特征"""
    def __init__(self, d_raw: int = 36, d_model: int = 128, feature_dim: int = 64, n_experts: int = 4):
        super().__init__()
        self.d_raw = d_raw
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.n_experts = n_experts  # 动态专家数量

        # 原始声学特征分析器 - 直接从36维特征提取声学信息
        # E1: Harmonic - F0轨迹分析 (假设特征维度包含F0相关信息)
        self.raw_f0_extractor = nn.Conv1d(d_raw, 16, kernel_size=7, padding=3)  # 原始F0特征
        self.raw_pitch_tracker = RobustLSTM(d_raw, 8, batch_first=True, bidirectional=True)  # F0轨迹跟踪

        # E2: Transient - 高频瞬态分析
        self.raw_transient_detector = nn.Conv1d(d_raw, 16, kernel_size=3, padding=1)  # 短时瞬态
        self.raw_energy_analyzer = nn.Conv1d(d_raw, 16, kernel_size=5, padding=2)  # 能量分布

        # E3: Burst-Inpaint - 连续性分析
        self.raw_continuity_check = nn.Conv1d(d_raw, 16, kernel_size=9, padding=4, dilation=2)  # 连续性
        self.raw_gap_detector = RobustGRU(d_raw, 8, batch_first=True, bidirectional=True) # 缺失检测

        # E4: Low-SNR - 信号质量分析
        self.raw_snr_estimator = nn.Sequential(
            nn.Linear(d_raw, d_raw//2),
            nn.ReLU(),
            nn.Linear(d_raw//2, 16)
        )
        self.raw_noise_profiler = nn.Conv1d(d_raw, 16, kernel_size=11, padding=5)  # 噪声轮廓

        # 编码特征分析器 - 从128维语义特征补充高层信息
        # E1: Harmonic - 谐波语义理解
        self.encoded_harmonic_semantic = nn.Conv1d(d_model, 8, kernel_size=7, padding=3)

        # E2: Transient - 瞬态语义理解
        self.encoded_transient_semantic = nn.Conv1d(d_model, 8, kernel_size=3, padding=1)

        # E3: Burst-Inpaint - 上下文语义理解
        self.encoded_context_semantic = nn.Conv1d(d_model, 8, kernel_size=11, padding=5)

        # E4: Low-SNR - 语义稳定性理解
        self.encoded_stability_semantic = nn.Conv1d(d_model, 8, kernel_size=9, padding=4)

        # 双层特征融合：原始声学(16) + 编码语义(8) = 24 per expert - 动态数量
        self.expert_fusion = nn.ModuleList([
            nn.Linear(24, 16) for _ in range(n_experts)  # 动态专家融合器
        ])

        # 最终特征融合 - 动态输入维度
        expert_feature_dim = n_experts * 16  # 专家数量 * 16特征
        self.feature_fusion = nn.Linear(expert_feature_dim, feature_dim)

    def forward(
        self,
        x_raw: torch.Tensor,
        ribbon_stream: torch.Tensor,
        thread_stream: torch.Tensor,
        fused_stream: torch.Tensor = None
    ) -> torch.Tensor:
        """
        三层特征提取：原始声学 + Ribbon语义带 + Thread微音段带

        Args:
            x_raw: [B, T, 36] 原始输入特征
            ribbon_stream: [B, T, 128] Ribbon长语义带 (3x下采样处理)
            thread_stream: [B, T, 128] Thread微音段带 (原分辨率处理)
            fused_stream: [B, T, 128] 融合特征 (可选)
        Returns:
            features: [B, feature_dim] 三层融合的专家偏好特征
        """
        b, t, _ = x_raw.shape

        # 转置为conv1d格式
        x_raw_conv = x_raw.transpose(1, 2)  # [B, 36, T]
        ribbon_conv = ribbon_stream.transpose(1, 2)  # [B, 128, T]
        thread_conv = thread_stream.transpose(1, 2)  # [B, 128, T]

        expert_features = []

        # E1: Harmonic Expert - 谐波分析
        # 层1: 原始声学F0轨迹分析
        raw_f0 = self.raw_f0_extractor(x_raw_conv)  # [B, 16, T]
        # 使用更安全的统计计算避免fp16下的NaN
        with torch.no_grad():
            raw_f0_safe = torch.clamp(raw_f0.float(), min=-100.0, max=100.0)
            f0_var = raw_f0_safe.var(dim=-1, unbiased=False, keepdim=False)
            f0_stability = torch.clamp(f0_var + 1e-6, min=1e-6, max=100.0).to(raw_f0.dtype)  # [B, 16] F0稳定性

        # RobustLSTM会自动处理tensor连续性和dtype兼容性
        pitch_out, _ = self.raw_pitch_tracker(x_raw)  # [B, T, 16]

        # 检查LSTM输出是否有NaN
        if torch.isnan(pitch_out).any():
            print(f"⚠️ pitch_out has NaN! Input range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
            pitch_out = torch.where(torch.isnan(pitch_out), torch.zeros_like(pitch_out), pitch_out)

        with torch.no_grad():
            pitch_safe = torch.clamp(pitch_out.float(), min=-100.0, max=100.0)
            pitch_std = pitch_safe.std(dim=1, unbiased=False, keepdim=False)
            pitch_continuity = torch.clamp(pitch_std + 1e-6, min=1e-6, max=100.0).to(pitch_out.dtype)  # [B, 16] 音调连续性

        # 层2: Ribbon长语义带 - 谐波的语言学意义(音节/词汇级别的F0模式)
        ribbon_harmonic = self.encoded_harmonic_semantic(ribbon_conv)  # [B, 8, T]
        ribbon_harmonic_pooled = ribbon_harmonic.mean(dim=-1)  # [B, 8] 长期谐波模式

        # 层3: Thread微音段带 - 细粒度F0调制和谐波微结构
        # 使用Thread流分析短时F0变化和谐波细节
        thread_f0_micro = thread_stream.mean(dim=1)[:, :8]  # [B, 8] 简化提取谐波微结构

        # 三层融合: 原始F0(16) + Ribbon语义F0(8) + Thread微F0(8) = 32 → 16
        # 检查所有组件是否有NaN
        components = [f0_stability, pitch_continuity, ribbon_harmonic_pooled, thread_f0_micro]
        for i, comp in enumerate(components):
            if torch.isnan(comp).any():
                print(f"⚠️ Component {i} has NaN: {comp.shape}, range=[{comp.min():.3f}, {comp.max():.3f}]")
                components[i] = torch.where(torch.isnan(comp), torch.zeros_like(comp), comp)

        harmonic_triple = torch.cat(components, dim=-1)  # [B, 32]
        # 安全截断 + 去非数
        triple_safe = torch.nan_to_num(harmonic_triple[:, :24], nan=0.0, posinf=1e4, neginf=-1e4)
        if torch.isnan(triple_safe).any():
            print(f"⚠️ triple_safe has NaN before fusion!")
            triple_safe = torch.where(torch.isnan(triple_safe), torch.zeros_like(triple_safe), triple_safe)

        harmonic_fused = self.expert_fusion[0](triple_safe)  # [B, 16]

        # 检查融合结果
        if torch.isnan(harmonic_fused).any():
            print(f"⚠️ harmonic_fused has NaN after fusion!")
            harmonic_fused = torch.where(torch.isnan(harmonic_fused), torch.zeros_like(harmonic_fused), harmonic_fused)

        expert_features.append(harmonic_fused)

        # E2: Transient Expert - 瞬态分析
        # 层1: 原始声学瞬态检测
        raw_transient = self.raw_transient_detector(x_raw_conv)  # [B, 16, T]
        with torch.no_grad():
            transient_safe = torch.clamp(raw_transient.float(), min=-100.0, max=100.0)
            transient_std = transient_safe.std(dim=-1, unbiased=False, keepdim=False)
            transient_intensity = torch.clamp(transient_std + 1e-6, min=1e-6, max=100.0).to(raw_transient.dtype)  # [B, 16] 瞬态强度

        raw_energy = self.raw_energy_analyzer(x_raw_conv)  # [B, 16, T]
        with torch.no_grad():
            energy_safe = torch.clamp(raw_energy.float(), min=-100.0, max=100.0)
            energy_var = energy_safe.var(dim=-1, unbiased=False, keepdim=False)
            energy_variance = torch.clamp(energy_var + 1e-6, min=1e-6, max=100.0).to(raw_energy.dtype)  # [B, 16] 能量变化

        # 层2: Ribbon长语义带 - 音素级瞬态模式(爆破音vs摩擦音的语言学分类)
        ribbon_transient = self.encoded_transient_semantic(ribbon_conv)  # [B, 8, T]
        with torch.no_grad():
            ribbon_trans_safe = torch.clamp(ribbon_transient.float(), min=-100.0, max=100.0)
            ribbon_trans_std = ribbon_trans_safe.std(dim=-1, unbiased=False, keepdim=False)
            ribbon_transient_pooled = torch.clamp(ribbon_trans_std + 1e-6, min=1e-6, max=100.0).to(ribbon_transient.dtype)  # [B, 8] 语言学瞬态模式

        # 层3: Thread微音段带 - 最适合瞬态！保持原分辨率捕获短时冲击
        with torch.no_grad():
            thread_safe = torch.clamp(thread_stream.float(), min=-100.0, max=100.0)
            thread_std = thread_safe.std(dim=1, unbiased=False, keepdim=False)[:, :8]
            thread_transient_micro = torch.clamp(thread_std + 1e-6, min=1e-6, max=100.0).to(thread_stream.dtype)  # [B, 8] Thread天然适合瞬态

        # 三层融合: 原始瞬态(16) + Ribbon语义瞬态(8) + Thread微瞬态(8) = 32 → 16
        transient_components = [transient_intensity, energy_variance, ribbon_transient_pooled, thread_transient_micro]
        for i, comp in enumerate(transient_components):
            if torch.isnan(comp).any():
                print(f"⚠️ Transient component {i} has NaN")
                transient_components[i] = torch.where(torch.isnan(comp), torch.zeros_like(comp), comp)

        transient_triple = torch.cat(transient_components, dim=-1)
        transient_safe = transient_triple[:, :24]
        transient_safe = torch.nan_to_num(transient_safe, nan=0.0, posinf=1e4, neginf=-1e4)
        if torch.isnan(transient_safe).any():
            transient_safe = torch.where(torch.isnan(transient_safe), torch.zeros_like(transient_safe), transient_safe)

        transient_fused = self.expert_fusion[1](transient_safe)  # [B, 16]
        if torch.isnan(transient_fused).any():
            print(f"⚠️ transient_fused has NaN after fusion!")
            transient_fused = torch.where(torch.isnan(transient_fused), torch.zeros_like(transient_fused), transient_fused)

        expert_features.append(transient_fused)

        # E3: Burst-Inpaint Expert - 连续性和修复分析
        # 层1: 原始声学连续性检查
        raw_continuity = self.raw_continuity_check(x_raw_conv)  # [B, 16, T]
        if t > 1:
            continuity_breaks = torch.diff(raw_continuity, dim=-1).abs().mean(dim=-1)  # [B, 16]
        else:
            continuity_breaks = torch.zeros(b, 16, device=x_raw.device)

        gap_out, _ = self.raw_gap_detector(x_raw)  # [B, T, 16]
        gap_pattern = torch.clamp(gap_out.float().std(dim=1, unbiased=False) + 1e-6, min=1e-6, max=100.0).to(gap_out.dtype)  # [B, 16] 缺失模式

        # 层2: Ribbon长语义带 - 长程上下文依赖(句子级修复语义)
        ribbon_context = self.encoded_context_semantic(ribbon_conv)  # [B, 8, T]
        ribbon_context_pooled = ribbon_context.mean(dim=-1)  # [B, 8] 长程语义上下文

        # 层3: Thread微音段带 - 局部邻近修复(音素间过渡)
        thread_local_context = torch.clamp(thread_stream.float().var(dim=1, unbiased=False)[:, :8] + 1e-6, min=1e-6, max=100.0).to(thread_stream.dtype)  # [B, 8] 局部变异用于检测缺失

        # 三层融合: 原始连续性(16) + Ribbon长程(8) + Thread局部(8) = 32 → 16
        inpaint_triple = torch.cat([continuity_breaks, gap_pattern, ribbon_context_pooled, thread_local_context], dim=-1)
        inpaint_safe   = torch.nan_to_num(inpaint_triple[:, :24], nan=0.0, posinf=1e4, neginf=-1e4)
        inpaint_fused  = self.expert_fusion[2](inpaint_safe)  # [B, 16]
        expert_features.append(inpaint_fused)

        # E4: Low-SNR Expert - 信号质量和稳定性分析 (仅当n_experts=4时)
        if self.n_experts >= 4:
            # 层1: 原始声学SNR估计
            x_raw_mean = x_raw.mean(dim=1)  # [B, 36]
            raw_snr = self.raw_snr_estimator(x_raw_mean)  # [B, 16]

            raw_noise = self.raw_noise_profiler(x_raw_conv)  # [B, 16, T]
            noise_profile = torch.clamp(raw_noise.float().std(dim=-1, unbiased=False) + 1e-6, min=1e-6, max=100.0).to(raw_noise.dtype)  # [B, 16] 噪声特征

            # 层2: Ribbon长语义带 - 语义稳定性(词汇/句子级一致性)
            ribbon_stability = self.encoded_stability_semantic(ribbon_conv)  # [B, 8, T]
            ribbon_stability_pooled = torch.clamp(ribbon_stability.float().var(dim=-1, unbiased=False) + 1e-6, min=1e-6, max=100.0).to(ribbon_stability.dtype)  # [B, 8] 长期语义稳定性

            # 层3: Thread微音段带 - 信号质量微分析(音素级SNR)
            thread_signal_quality = thread_stream.mean(dim=1)[:, 8:16]  # [B, 8] Thread微信号质量

            # 三层融合: 原始SNR(16) + Ribbon语义稳定(8) + Thread质量(8) = 32 → 16
            lowsnr_triple = torch.cat([raw_snr, noise_profile, ribbon_stability_pooled, thread_signal_quality], dim=-1)
            lowsnr_fused = self.expert_fusion[3](lowsnr_triple[:, :24])  # [B, 16]
            expert_features.append(lowsnr_fused)

        # 组合所有专家特征 (动态长度)
        all_expert_features = torch.cat(expert_features, dim=-1)  # [B, n_experts*16]
        all_expert_features = torch.nan_to_num(all_expert_features, nan=0.0, posinf=1e4, neginf=-1e4)
        # 最终特征融合
        acoustic_features = self.feature_fusion(all_expert_features)  # [B, feature_dim]

        return acoustic_features


class HarmonicExpert(nn.Module):
    """E1: 谐波专家 - 简化但功能明确的谐波处理"""
    def __init__(self, d_model: int = 128):
        super().__init__()

        # 谐波建模：长卷积捕获低频谐波结构
        self.harmonic_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)

        # F0平滑：避免基频跳跃
        self.f0_smooth = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # 限制范围，平滑F0
            nn.Linear(d_model, d_model)
        )

        # 浊音连续性：GRU建模时序依赖
        self.voicing_gru = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        # 1. 谐波卷积
        x_conv = x.transpose(1, 2)  # [B, D, T]
        harmonic_feat = self.harmonic_conv(x_conv).transpose(1, 2)  # [B, T, D]

        # 2. F0平滑
        f0_smooth = self.f0_smooth(harmonic_feat)

        # 3. 浊音连续性
        voiced_feat, _ = self.voicing_gru(f0_smooth)  # [B, T, D]

        # 数值稳定性检查和裁剪
        voiced_feat = torch.clamp(voiced_feat, min=-10.0, max=10.0)

        return voiced_feat + x


class TransientExpert(nn.Module):
    """E2: 瞬态专家 - 简化但有效的瞬态处理"""
    def __init__(self, d_model: int = 128):
        super().__init__()

        # 瞬态检测：短卷积（深度可分离 + 轻度空洞）捕获快速变化且算力友好
        self.transient_dw = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model
        )
        self.transient_pw = nn.Conv1d(d_model, d_model, kernel_size=1)

        # 突变增强：GLU门控提取瞬态脉冲
        self.burst_enhance = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1),
            nn.Linear(d_model, d_model)
        )

        # 边缘保护：防止瞬态被过度平滑
        self.edge_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # 门控保护
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        # 1. 瞬态检测
        x_conv = x.transpose(1, 2)  # [B, D, T]
        transient_feat = self.transient_pw(self.transient_dw(x_conv)).transpose(1, 2)  # [B, T, D]

        # 2. 突变增强
        burst_enhanced = self.burst_enhance(transient_feat)

        # 3. 边缘保护
        gate_weights = self.edge_gate(burst_enhanced)
        protected = burst_enhanced * gate_weights

        return protected + x


class BurstInpaintExpert(nn.Module):
    """E3: 缺失修复专家 - 简化但有效的修复处理"""
    def __init__(self, d_model: int = 128):
        super().__init__()

        # 上下文捕获：扩张卷积获取长程依赖
        self.context_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=4, dilation=2)

        # 双向建模：LSTM获取前后上下文
        self.context_lstm = RobustLSTM(d_model, d_model//2, batch_first=True, bidirectional=True)

        # 修复网络：GLU适合生成任务
        self.inpaint_net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GLU(dim=-1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        # 1. 上下文捕获
        x_conv = x.transpose(1, 2)  # [B, D, T]
        context_feat = self.context_conv(x_conv).transpose(1, 2)  # [B, T, D]

        # 2. 双向建模
        bidirectional_feat, _ = self.context_lstm(context_feat)  # [B, T, D]

        # 数值稳定性检查
        bidirectional_feat = torch.clamp(bidirectional_feat, min=-10.0, max=10.0)

        # 3. 修复处理
        inpainted = self.inpaint_net(bidirectional_feat)

        return inpainted + x


class LowSNRExpert(nn.Module):
    """E4: 低SNR专家 - 简化但有效的降噪和语义保护"""
    def __init__(self, d_model: int = 128):
        super().__init__()

        # 降噪：平滑卷积
        self.denoise_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # 语义保护：LayerNorm + GLU
        self.semantic_protect = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*2),
            nn.GLU(dim=-1),
            nn.Dropout(0.05),  # 轻微dropout增强稳定性
            nn.Linear(d_model, d_model)
        )

        # 稳定性增强：LSTM平滑时序
        self.stability_lstm = RobustLSTM(d_model, d_model//2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        # 1. 降噪处理
        x_conv = x.transpose(1, 2)  # [B, D, T]
        denoised = self.denoise_conv(x_conv).transpose(1, 2)  # [B, T, D]

        # 2. 语义保护
        semantic_protected = self.semantic_protect(denoised)

        # 3. 稳定性增强
        stable_feat, _ = self.stability_lstm(semantic_protected)  # [B, T, D]

        # 数值稳定性检查
        stable_feat = torch.clamp(stable_feat, min=-10.0, max=10.0)

        return stable_feat + x


class SpecializedMicroMoE(nn.Module):
    """专业化MicroMoE - 基于音频内容的token-level路由

    Features:
    - 4个专业化expert: Harmonic/Transient/BurstInpaint/LowSNR
    - Acoustic-aware router替代global_pool
    - Token-level routing支持时序内的动态expert选择
    - CSI integration for Stage3 ablation compatibility
    """
    def __init__(
        self,
        D: int = 128,
        d_csi: int = 10,
        n_experts: int = 4,
        topk: int = 2,
        expert_dropout: float = 0.0,
        balance_weight: float = 0.5,
        router_use_csi: bool = True,
        use_token_level: bool = True,
    ):
        super().__init__()
        self.d_model = D
        self.d_csi = d_csi
        self.n_experts = n_experts
        self.topk = topk
        self.expert_dropout = expert_dropout
        self.balance_weight = balance_weight
        self.router_use_csi = router_use_csi
        self.use_token_level = use_token_level
        self.router_jitter = 0.0  # 训练态的小抖动强度，外部可注入
        # 双层acoustic feature extractor: 原始声学特征 + 编码特征 - 传递专家数量
        self.acoustic_extractor = AcousticFeatureExtractor(d_raw=36, d_model=D, feature_dim=64, n_experts=n_experts)

        # Enhanced router: acoustic_features(64) + [csi(10)] -> experts(4)
        acoustic_dim = 64
        router_input_dim = acoustic_dim + (d_csi if router_use_csi else 0)

        if use_token_level:
            # Token-level router: 处理每个时间步的路由
            self.token_router = nn.Sequential(
                nn.Linear(D, 64),  # 每个token的特征
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, n_experts)
            )

        # Sample-level router: 基于acoustic features
        hidden_dim = max(32, router_input_dim // 2)
        self.sample_router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )

        # Router初始化：确保均匀的专家选择
        for router in [self.sample_router, self.token_router if use_token_level else None]:
            if router is not None:
                with torch.no_grad():
                    # 最后一层（路由层）的偏置设为小的负值，避免极端softmax
                    router[-1].bias.fill_(-1.0)  # 小负值确保初期均匀分布
                    # 路由层权重使用更小的方差
                    nn.init.normal_(router[-1].weight, mean=0.0, std=0.01)

        # 专业化Expert networks - 根据n_experts动态创建
        expert_classes = [
            HarmonicExpert,      # E1: 谐波专家 (基础，总是需要)
            TransientExpert,     # E2: 瞬态专家 (基础，总是需要)
            BurstInpaintExpert,  # E3: 突发修复专家 (基础，总是需要)
            LowSNRExpert,        # E4: 低SNR专家 (需要CSI，Stage3跳过)
        ]

        self.experts = nn.ModuleList([
            expert_classes[i](D) for i in range(min(n_experts, len(expert_classes)))
        ])

        # Expert utilization tracking
        self.register_buffer('expert_counts', torch.zeros(n_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

        # 🔧 参数初始化稳定化 - 防止梯度爆炸
        self._init_parameters()

    def _init_parameters(self):
        """平衡的参数初始化，避免梯度消失和爆炸"""
        # 对所有专家网络应用平衡的初始化
        for expert in self.experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    # 使用标准的Xavier初始化，适度的gain
                    nn.init.xavier_uniform_(module.weight, gain=0.5)  # 平衡的gain
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                    # 卷积层使用He初始化（适合ReLU）
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.LSTM, nn.GRU)):
                    # RNN权重使用标准正交初始化
                    for name, param in module.named_parameters():
                        if 'weight' in name:
                            nn.init.orthogonal_(param, gain=1.0)  # 标准gain
                        elif 'bias' in name:
                            # 遗忘门偏置设为1，其他设为0
                            nn.init.zeros_(param)
                            if 'bias_hh' in name:  # 遗忘门偏置
                                hidden_size = param.size(0) // 3  # GRU有3个门
                                param.data[hidden_size:2*hidden_size].fill_(1.0)

        # 对acoustic feature extractor应用标准初始化
        for module in self.acoustic_extractor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        if 'bias_hh' in name and 'GRU' in str(type(module)):
                            hidden_size = param.size(0) // 3
                            param.data[hidden_size:2*hidden_size].fill_(1.0)

    def update_expert_usage(self, assignments: torch.Tensor):
        """Update expert usage statistics with numerical safety."""
        with torch.no_grad():  # 确保统计更新不参与梯度图
            # assignments: [B, E]
            counts = assignments.sum(dim=0)  # [E]
            total = assignments.sum()

            # EMA update with clamping
            momentum = 0.99
            self.expert_counts = momentum * self.expert_counts + (1 - momentum) * counts.detach()
            self.total_samples = momentum * self.total_samples + (1 - momentum) * total.detach()

    def load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """Compute load balance loss to encourage uniform expert usage with numerical stability."""
        # 强制fp32计算避免bf16下的数值问题
        gate_logits = gate_logits.float().clamp_(-20, 20)
        probs = F.softmax(gate_logits, dim=-1)  # [B, E]
        mean_probs = probs.mean(dim=0)  # [E]

        # 使用更稳定的MSE损失替代KL散度
        uniform = torch.full_like(mean_probs, 1.0 / self.n_experts)
        mse_loss = F.mse_loss(mean_probs, uniform)

        # 确保损失为正且有意义的梯度
        return mse_loss.clamp(min=1e-8)

    def forward(self, h: torch.Tensor, csi_vec: torch.Tensor = None, x_raw: torch.Tensor = None, dual_streams: dict = None) -> torch.Tensor:
        """
        专业化MoE前向传播 - 基于三层特征的智能路由

        Args:
            h: [B, T, D] 融合后特征 (DualStream输出)
            csi_vec: [B, d_csi] optional CSI vector for Stage3 compatibility
            x_raw: [B, T, 36] 原始输入特征
            dual_streams: dict containing {'ribbon_stream', 'thread_stream', 'fused_stream'}

        Returns:
            output: [B, T, D] expert-processed features
        """
        b, t, d = h.shape

        # 🚨 紧急调试模式：完全绕过MoE逻辑，但仍记录统计
        if hasattr(self, '_emergency_bypass') and self._emergency_bypass:
            # 在绕过模式下，模拟均匀的专家使用统计
            if self.training:
                with torch.no_grad():
                    # 创建均匀分布的假专家分配用于统计
                    uniform_assignments = torch.ones(b, self.n_experts, device=h.device) / self.n_experts
                    self.update_expert_usage(uniform_assignments)
            return h  # 直接返回输入，完全跳过MoE

        # 检查是否提供完整特征
        if x_raw is None or dual_streams is None:
            # 兼容模式：直接返回输入特征，避免acoustic_extractor的潜在问题
            return h
        else:
            # 尝试三层特征提取，但添加异常捕获
            try:
                # 1. 三层特征提取：原始声学 + Ribbon语义 + Thread微音段
                ribbon_stream = dual_streams.get('ribbon_stream', h)  # [B,T,128]
                thread_stream = dual_streams.get('thread_stream', h)  # [B,T,128]
                acoustic_features = self.acoustic_extractor(x_raw, ribbon_stream, thread_stream)  # [B, 64]
                acoustic_features = torch.nan_to_num(acoustic_features, nan=0.0, posinf=1e4, neginf=-1e4)
            except Exception as e:
                print(f"Warning: acoustic_extractor failed: {e}, falling back to bypass mode")
                return h

        # 2. Prepare router input
        if self.router_use_csi and csi_vec is not None:
            router_input = torch.cat([acoustic_features, csi_vec], dim=-1)  # [B, 64+d_csi]
        else:
            router_input = acoustic_features  # [B, 64] - Stage3 ablation

        # 3. Sample-level routing based on acoustic preferences
        sample_gate_logits = self.sample_router(router_input)  # [B, E]
        # 强制数值稳定性：fp32路由计算 + 安全归一化
        sample_gate_logits = sample_gate_logits.float().clamp_(-20, 20)  # fp32 + 截断
        # >>> ROUTER EXPLORATION JITTER <<<
        # >>> ROUTER EXPLORATION JITTER (sample-level) <<<
        if self.training and getattr(self, 'router_jitter', 0.0) > 0.0:
            j = float(self.router_jitter)
            sample_gate_logits = sample_gate_logits + j * torch.randn_like(sample_gate_logits)
        # <<< END JITTER <<<

        sample_probs = F.softmax(sample_gate_logits, dim=-1)  # fp32 softmax

        # 保存路由信息用于诊断（不参与梯度计算）
        if self.training:
            self._last_router_logits = sample_gate_logits.detach()

        if self.use_token_level:
            # 4. Token-level routing for fine-grained control
            token_gate_logits = self.token_router(h)  # [B, T, E]
            # >>> ROUTER EXPLORATION JITTER (token-level) <<<
            if self.training and getattr(self, 'router_jitter', 0.0) > 0.0:
                j = float(self.router_jitter)
                token_gate_logits = token_gate_logits + j * torch.randn_like(token_gate_logits)
            # <<< END JITTER <<<
            # 强制数值稳定性：fp32路由计算 + 安全归一化
            token_gate_logits = token_gate_logits.float().clamp_(-20, 20)  # fp32 + 截断
            token_probs = F.softmax(token_gate_logits, dim=-1)  # fp32 softmax

            # 融合sample-level和token-level routing
            # Sample偏好作为先验，token细化局部决策
            sample_probs_expanded = sample_probs.unsqueeze(1).expand(b, t, self.n_experts)  # [B, T, E]
            combined_probs = 0.6 * sample_probs_expanded + 0.4 * token_probs  # [B, T, E]

            # Token-level top-k selection
            weights, indices = torch.topk(combined_probs, k=self.topk, dim=-1)  # [B, T, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # 安全归一化
            weights = weights.to(h.dtype)  # 回到原 dtype（bf16/fp16）

            # Create token-level assignment matrix
            assignments = torch.zeros_like(combined_probs)  # [B, T, E]
            assignments.scatter_(-1, indices, weights)  # [B, T, E]

            # Expert dropout
            if self.training and self.expert_dropout > 0:
                dropout_mask = (torch.rand(b, t, self.n_experts, device=h.device) > self.expert_dropout).float()
                assignments = assignments * dropout_mask
                assignments = assignments / (assignments.sum(dim=-1, keepdim=True) + 1e-6)

            # Apply experts with token-level weighting
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_out = expert(h)  # [B, T, D]
                expert_outputs.append(expert_out)

            expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, T, D, E]

            # Token-level weighted combination
            output = torch.einsum('btde,bte->btd', expert_stack, assignments)  # [B, T, D]

            # Usage tracking (sample-level for consistency)
            if self.training:
                sample_assignments = assignments.mean(dim=1)  # [B, E] - average over time
                self.update_expert_usage(sample_assignments.detach())

        else:
            # Sample-level routing (fallback)
            weights, indices = torch.topk(sample_probs, k=self.topk, dim=-1)  # [B, k]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
            weights = weights.to(h.dtype)  # 回到原 dtype

            assignments = torch.zeros_like(sample_probs)  # [B, E]
            assignments.scatter_(1, indices, weights)

            if self.training and self.expert_dropout > 0:
                dropout_mask = (torch.rand(b, self.n_experts, device=h.device) > self.expert_dropout).float()
                assignments = assignments * dropout_mask
                assignments = assignments / (assignments.sum(dim=-1, keepdim=True) + 1e-6)

            # Apply experts
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                expert_out = expert(h)  # [B, T, D]
                expert_outputs.append(expert_out)

            expert_stack = torch.stack(expert_outputs, dim=-1)  # [B, T, D, E]
            output = torch.einsum('btde,be->btd', expert_stack, assignments)  # [B, T, D]

            if self.training:
                self.update_expert_usage(assignments.detach())

        return output

    @torch._dynamo.disable  # 不进入 torch.compile 图，安全使用 .item()
    def get_expert_utilization(self) -> torch.Tensor:
        """Expert utilization for logging/metrics only (kept outside compiled graph)."""
        if self.total_samples.item() > 0:
            util = self.expert_counts / (self.total_samples + 1e-8)
        else:
            util = torch.ones_like(self.expert_counts) / self.n_experts
        return util.detach()


    def get_aux_losses(
        self,
        h: torch.Tensor,
        csi_vec: torch.Tensor = None,
        x_raw: torch.Tensor = None,
        dual_streams: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        aux_losses: Dict[str, torch.Tensor] = {}
        b = h.size(0)

        # 1) 准备路由输入（与 forward 同步）
        if x_raw is not None and dual_streams is not None:
            ribbon_stream = dual_streams.get('ribbon_stream', h)
            thread_stream = dual_streams.get('thread_stream', h)
            acoustic_features = self.acoustic_extractor(x_raw, ribbon_stream, thread_stream)  # [B,64]
        else:
            acoustic_features = torch.zeros(b, 64, device=h.device, dtype=h.dtype)

        router_input = torch.cat([acoustic_features, csi_vec], dim=-1) if (self.router_use_csi and csi_vec is not None) else acoustic_features

        # 2) Sample-level 平衡损失
        sample_gate_logits = self.sample_router(router_input).float().clamp_(-20, 20)  # 数值稳定
        aux_losses['moe_balance_loss'] = self.load_balance_loss(sample_gate_logits)     # 维持为 tensor

        # 3) Token-level 平衡损失（若启用 token router）
        if self.use_token_level:
            token_gate_logits = self.token_router(h).float().clamp_(-20, 20)
            e = token_gate_logits.shape[-1]
            token_gate_flat = token_gate_logits.view(-1, e)                              # 关键：避免 UnboundLocalError
            aux_losses['moe_token_balance_loss'] = self.load_balance_loss(token_gate_flat)

        # 4) 可选的偏好/稀疏度指标（仅做监控，不直接入 loss）
        if x_raw is not None and dual_streams is not None:
            with torch.no_grad():
                aux_losses['moe_harmonic_pref'] = acoustic_features.mean()
        return aux_losses



class CompatibleMicroMoE(nn.Module):
    """向后兼容的MicroMoE接口 - 包装SpecializedMicroMoE"""
    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2, **kwargs):
        super().__init__()
        # 映射参数到SpecializedMicroMoE
        specialized_kwargs = {
            'D': d_model,
            'n_experts': n_experts,
            'topk': top_k,
        }
        # 传递其他kwargs
        for key, value in kwargs.items():
            if key in ['d_csi', 'expert_dropout', 'balance_weight', 'router_use_csi']:
                specialized_kwargs[key] = value

        self.specialized_moe = SpecializedMicroMoE(**specialized_kwargs)

    def forward(self, h: torch.Tensor, router_input: torch.Tensor, x_raw: torch.Tensor = None, dual_streams: dict = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """兼容接口 - 返回output和aux losses"""
        # router_input现在是acoustic_features(64)或acoustic_features(64)+csi(10)
        # 由于acoustic_extractor已经在上层调用过，这里只需要解析CSI

        expected_acoustic_dim = 64  # AcousticFeatureExtractor输出维度
        if router_input.shape[-1] > expected_acoustic_dim:
            # 包含CSI的情况: [acoustic_features(64) + csi(d_csi)]
            csi_dim = router_input.shape[-1] - expected_acoustic_dim
            if csi_dim == self.specialized_moe.d_csi:
                csi_vec = router_input[:, -csi_dim:]  # [B, d_csi]
            else:
                # CSI维度不匹配，使用截断或零填充
                if csi_dim > self.specialized_moe.d_csi:
                    csi_vec = router_input[:, -self.specialized_moe.d_csi:]  # 截断
                else:
                    # 零填充
                    padding = torch.zeros(router_input.shape[0], self.specialized_moe.d_csi - csi_dim,
                                        device=router_input.device, dtype=router_input.dtype)
                    csi_vec = torch.cat([router_input[:, expected_acoustic_dim:], padding], dim=-1)
        elif router_input.shape[-1] == expected_acoustic_dim:
            # 纯acoustic features，无CSI
            csi_vec = None
        else:
            # 维度不匹配，可能是legacy格式
            csi_vec = None

        output = self.specialized_moe(h, csi_vec, x_raw=x_raw, dual_streams=dual_streams)
        aux_losses = self.specialized_moe.get_aux_losses(h, csi_vec, x_raw=x_raw, dual_streams=dual_streams)
        if 'balance_loss' in aux_losses:               # 兼容旧键
            aux_losses['moe_balance_loss'] = aux_losses.pop('balance_loss')
        if 'token_balance_loss' in aux_losses:         # 兼容旧键
            aux_losses['moe_token_balance_loss'] = aux_losses.pop('token_balance_loss')
        return output, aux_losses

    def get_expert_utilization(self):
        """向后兼容方法"""
        return self.specialized_moe.get_expert_utilization()


# 保持旧的MicroMoE类以兼容可能的直接引用
class MicroMoE(nn.Module):
    """Legacy MicroMoE - redirects to SpecializedMicroMoE for consistency"""
    def __init__(self, D: int = 128, d_csi: int = 10, n_experts: int = 4, topk: int = 2, **kwargs):
        super().__init__()
        # 直接使用SpecializedMicroMoE，但禁用token-level routing以保持兼容性
        kwargs['use_token_level'] = kwargs.get('use_token_level', False)
        self.specialized_moe = SpecializedMicroMoE(
            D=D, d_csi=d_csi, n_experts=n_experts, topk=topk, **kwargs
        )

    def forward(self, h: torch.Tensor, router_input: torch.Tensor) -> torch.Tensor:
        """Legacy interface for basic MoE functionality"""
        # 解析router_input以提取CSI
        if router_input.shape[-1] > self.specialized_moe.d_model:
            csi_dim = router_input.shape[-1] - self.specialized_moe.d_model
            if csi_dim == self.specialized_moe.d_csi:
                csi_vec = router_input[:, -csi_dim:]
            else:
                csi_vec = None
        else:
            csi_vec = None

        return self.specialized_moe(h, csi_vec)

    def get_expert_utilization(self):
        return self.specialized_moe.get_expert_utilization()

    def get_aux_losses(self, h: torch.Tensor, csi_vec: torch.Tensor = None):
        return self.specialized_moe.get_aux_losses(h, csi_vec)


if __name__ == "__main__":
    pass
