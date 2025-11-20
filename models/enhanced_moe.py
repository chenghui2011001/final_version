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
    from .utils import global_pool, extract_acoustic_priors
except ImportError:  # pragma: no cover
    from utils import global_pool, extract_acoustic_priors


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
        # Add numerical stability for mixed precision
        ribbon_harmonic_safe = torch.clamp(ribbon_harmonic.float(), min=-100.0, max=100.0)
        ribbon_harmonic_pooled = torch.nan_to_num(ribbon_harmonic_safe.mean(dim=-1),
                                                 nan=0.0, posinf=1.0, neginf=-1.0).to(ribbon_harmonic.dtype)

        # 层3: Thread微音段带 - 细粒度F0调制和谐波微结构
        # 使用Thread流分析短时F0变化和谐波细节
        thread_safe = torch.clamp(thread_stream.float(), min=-100.0, max=100.0)
        thread_f0_micro = torch.nan_to_num(thread_safe.mean(dim=1)[:, :8],
                                          nan=0.0, posinf=1.0, neginf=-1.0).to(thread_stream.dtype)

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
            # Add nan_to_num for mixed precision stability
            ribbon_trans_std = torch.nan_to_num(ribbon_trans_std, nan=0.0, posinf=1.0, neginf=0.0)
            ribbon_transient_pooled = torch.clamp(ribbon_trans_std + 1e-6, min=1e-6, max=100.0).to(ribbon_transient.dtype)  # [B, 8] 语言学瞬态模式

        # 层3: Thread微音段带 - 最适合瞬态！保持原分辨率捕获短时冲击
        with torch.no_grad():
            thread_safe = torch.clamp(thread_stream.float(), min=-100.0, max=100.0)
            thread_std = thread_safe.std(dim=1, unbiased=False, keepdim=False)[:, :8]
            # Add nan_to_num for mixed precision stability
            thread_std = torch.nan_to_num(thread_std, nan=0.0, posinf=1.0, neginf=0.0)
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


class UnifiedAudioExpert(nn.Module):
    """专业化引导的统一音频专家 - 保持FFN架构但添加专业化偏好

    Optional F0-conditioning:
    - Keeps the public forward signature unchanged.
    - Parent can inject a per-sample conditioning vector via `set_f0_condition`.
    - When enabled, applies a lightweight FiLM-style modulation before the expert FFN.
    """
    def __init__(self, d_model: int = 128, d_ff: int = None, expert_id: int = 0,
                 use_f0_condition: bool = False, f0_cond_dim: int = 6):
        super().__init__()
        self.expert_id = expert_id
        self.d_model = d_model
        self.use_f0_condition = use_f0_condition
        self.f0_cond_dim = f0_cond_dim

        # 自适应FFN维度
        if d_ff is None:
            d_ff = d_model * 4
        self.d_ff = d_ff

        # 专家特定的FFN架构 - 结构性差异化
        if self.expert_id == 0:
            # Harmonic专家：深层网络，关注长期依赖
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.LayerNorm(d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_ff // 2),
                nn.GELU(),
                nn.Linear(d_ff // 2, d_model),
                nn.Dropout(0.05)
            )
        elif self.expert_id == 1:
            # Transient专家：宽层网络，关注瞬态特征
            wide_ff = int(d_ff * 1.5)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, wide_ff),
                nn.ReLU(),  # 使用ReLU增强非线性
                nn.Dropout(0.1),
                nn.Linear(wide_ff, d_model),
                nn.Dropout(0.05)
            )
        elif self.expert_id == 2:
            # BurstInpaint专家：残差网络，关注局部修复
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_ff, d_model),
                nn.Dropout(0.05)
            )
            # 额外的残差路径
            self.inpaint_residual = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, d_model)
            )
        else:
            # LowSNR专家：保守网络，关注噪声抑制
            conservative_ff = int(d_ff * 0.75)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, conservative_ff),
                nn.Tanh(),  # 使用Tanh限制输出范围
                nn.Dropout(0.05),  # 更小的dropout
                nn.Linear(conservative_ff, d_model),
                nn.Dropout(0.02)
            )

        self.layer_norm = nn.LayerNorm(d_model)

        # 专家特定的差异化机制
        self.expert_bias = nn.Parameter(torch.zeros(d_model))

        # 专业化引导：每个专家有特定的特征关注权重
        self.specialization_weights = nn.Parameter(torch.ones(d_model) * 0.1)  # 小初始权重

        # 可选：F0 条件化（FiLM 风格调制）
        # 由父模块通过 set_f0_condition(cond:[B,f0_cond_dim] or [B,T,f0_cond_dim]) 设置。
        # 仅当 use_f0_condition=True 时启用；默认关闭以保持行为不变。
        if self.use_f0_condition:
            self.f0_gate = nn.Sequential(
                nn.Linear(self.f0_cond_dim, 2 * d_model)
            )
        else:
            self.f0_gate = None

        # 运行时缓存的条件（不进入state_dict）
        self._f0_cond = None

        # 根据expert_id设置专业化偏好
        self._init_specialization_bias()

    def _init_specialization_bias(self):
        """基于音频场景的专家专业化初始化"""
        with torch.no_grad():
            # 基于音频处理需求的差异化初始化

            # 修复: 使用平衡的差异化初始化策略
            base_spec_std = 0.05  # 适中的基础特化权重标准差，避免过度差异
            base_bias_std = 0.01  # 适中的基础偏置标准差，平衡个性和稳定性

            if self.expert_id == 0:
                # Expert 0: Harmonic专家 - 专注音调稳定性和周期结构
                # 轻微偏向低频、稳定特征
                self.specialization_weights.normal_(-0.02, base_spec_std * 0.8)  # 轻微负偏向，关注稳定性
                self.expert_bias.normal_(-0.005, base_bias_std)  # 轻微负偏置，保守处理

            elif self.expert_id == 1:
                # Expert 1: Transient专家 - 专注动态变化和瞬态检测
                # 轻微偏向高频、动态特征
                self.specialization_weights.normal_(0.03, base_spec_std * 1.2)  # 轻微正偏向，敏感检测
                self.expert_bias.normal_(0.008, base_bias_std * 1.5)  # 轻微正偏置，激活动态

            elif self.expert_id == 2:
                # Expert 2: BurstInpaint专家 - 专注上下文连续性和修复
                # 中性偏向，关注连续性
                self.specialization_weights.normal_(0.01, base_spec_std)  # 轻微正偏向，上下文关注
                self.expert_bias.normal_(0.003, base_bias_std * 1.2)  # 轻微正偏置

            elif self.expert_id == 3:
                # Expert 3: LowSNR专家 - 专注原始特征含噪性分析和质量自适应
                # 轻微偏向噪声抑制
                self.specialization_weights.normal_(-0.01, base_spec_std * 0.6)  # 轻微负偏向，避免放大噪声
                self.expert_bias.normal_(-0.003, base_bias_std * 0.8)  # 轻微负偏置，降噪处理

            # 修复: 使用更保守的专家功能偏置模式
            specialty_patterns = {
                0: [0.005, -0.003, 0.002, -0.004],  # Harmonic: 稳定模式
                1: [0.008, 0.005, -0.007, 0.006],   # Transient: 动态模式
                2: [0.003, 0.004, 0.005, -0.004],   # BurstInpaint: 上下文模式
                3: [-0.002, 0.001, -0.001, 0.002]   # LowSNR: 特征质量评估模式
            }
            if self.expert_id in specialty_patterns:
                pattern = specialty_patterns[self.expert_id]
                for i, val in enumerate(pattern):
                    if i * 32 < len(self.expert_bias):
                        self.expert_bias[i*32:(i+1)*32] += val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x = self.layer_norm(x)

        # F0 条件化（若启用且提供了 cond）
        if self.use_f0_condition and self.f0_gate is not None and (self._f0_cond is not None):
            cond = self._f0_cond
            # 支持 [B, C] 或 [B, T, C] 两种形状
            if cond.dim() == 3:
                # 时间平均到样本级提示（数值更稳）
                cond_vec = cond.mean(dim=1)
            else:
                cond_vec = cond  # [B, C]

            # 数值安全与类型对齐
            cond_vec = torch.nan_to_num(cond_vec, nan=0.0, posinf=1e4, neginf=-1e4).to(x.dtype)
            ab = self.f0_gate(cond_vec)  # [B, 2D]
            a, b = ab.chunk(2, dim=-1)
            # 轻量调制，限制幅度
            a = 0.25 * torch.tanh(a)
            b = 0.10 * torch.tanh(b)
            # 应用到每个时间步
            x = x * (1.0 + a.unsqueeze(1)) + b.unsqueeze(1)

        # 修复: 增强专家差异化机制
        # 1. 特征选择性关注 (更强的差异化)
        attention_mask = torch.sigmoid(self.specialization_weights)  # [D] -> [0, 1]
        x_specialized = x * attention_mask.unsqueeze(0).unsqueeze(0)  # [B, T, D]

        # 2. 专家特定的特征变换
        expert_transform = torch.tanh(self.expert_bias.unsqueeze(0).unsqueeze(0))  # [1, 1, D]
        x_transformed = x + 0.1 * expert_transform  # 小幅特征调制

        # 3. 结合原始特征、专业化特征和变换特征
        # 让每个专家有不同的混合比例
        mix_weight = 0.3 + 0.4 * torch.sigmoid(self.specialization_weights.mean())  # [0.3, 0.7]
        x = mix_weight * x_specialized + (1 - mix_weight) * x_transformed

        # FFN处理 (专业化的主要计算)
        x = self.ffn(x)

        # BurstInpaint专家的额外残差路径
        if self.expert_id == 2 and hasattr(self, 'inpaint_residual'):
            inpaint_contrib = self.inpaint_residual(residual)
            x = x + 0.1 * inpaint_contrib  # 小权重添加修复贡献

        # 添加专家特定的输出偏置 (增强差异化)
        output_bias = self.expert_bias * 0.1  # 控制输出偏置强度
        x = x + output_bias.unsqueeze(0).unsqueeze(0)

        # 残差连接
        return x + residual

    # ---- 条件注入 API（保持 forward 接口不变）----
    def set_f0_condition(self, cond: torch.Tensor):
        """Set per-sample F0-related conditioning vector.

        Accepts [B, C] (sample-level) or [B, T, C] (token-level). Stored transiently.
        """
        self._f0_cond = cond

    def clear_condition(self):
        self._f0_cond = None

    def get_specialization_info(self):
        """获取专家专业化信息 - 基于音频场景功能"""
        specializations = {
            0: "Harmonic Expert (tonal stability, periodic structure)",
            1: "Transient Expert (dynamic changes, burst detection)",
            2: "BurstInpaint Expert (context continuity, repair)",
            3: "LowSNR Expert (feature noise analysis, quality assessment)"
        }
        audio_scenarios = {
            0: "Voiced speech, musical tones, stable pitch",
            1: "Consonants, percussive sounds, rapid changes",
            2: "Packet loss, gaps, missing segments",
            3: "Noisy raw features, poor feature quality"
        }
        return {
            'expert_id': self.expert_id,
            'specialization': specializations.get(self.expert_id, "Unknown"),
            'audio_scenario': audio_scenarios.get(self.expert_id, "Unknown"),
            'bias_norm': self.expert_bias.norm().item(),
            'spec_weight_norm': self.specialization_weights.norm().item()
        }


# 注意：TransientExpert 现在由 UnifiedAudioExpert 代替


# 注意：BurstInpaintExpert 现在由 UnifiedAudioExpert 代替


# 注意：LowSNRExpert 现在由 UnifiedAudioExpert 代替


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

        # Enhanced router: 纯音频特征路由 - acoustic_features(64) + global_stats(16) -> experts(4)
        # 不再使用CSI，改为基于音频内容本身的特征进行专家路由
        acoustic_dim = 64
        global_stats_dim = 16  # 频谱质心变化 + 谱滚降
        router_input_dim = acoustic_dim + global_stats_dim  # 移除CSI依赖

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

        # Router初始化：确保均匀的专家选择和数值稳定性
        for router in [self.sample_router, self.token_router if use_token_level else None]:
            if router is not None:
                with torch.no_grad():
                    # 修复1: 使用更温和的偏置初始化，避免极端值
                    router[-1].bias.fill_(-0.1)  # 温和的负值，避免过度集中
                    # 修复2: 增大路由层权重方差，确保足够的学习能力
                    nn.init.normal_(router[-1].weight, mean=0.0, std=0.05)

                    # 修复3: 为隐藏层也添加合理的初始化
                    for layer in router[:-1]:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.5)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)

        # 使用统一的简化专家架构 - 让路由器决定专业化
        # 不再预设专家功能，而是通过路由器学习自然分工：
        # Expert 0 -> 可能特化为谐波处理
        # Expert 1 -> 可能特化为瞬态处理
        # Expert 2 -> 可能特化为修复处理
        self.experts = nn.ModuleList([
            UnifiedAudioExpert(
                d_model=D,
                expert_id=i,
                use_f0_condition=(i == 0),  # 仅对谐波专家启用F0条件化
                f0_cond_dim=6
            ) for i in range(n_experts)
        ])

        # Expert utilization tracking
        self.register_buffer('expert_counts', torch.zeros(n_experts))
        self.register_buffer('total_samples', torch.tensor(0.0))

        # 专家专业化引导机制
        self.register_buffer('expert_signal_types', torch.zeros(n_experts, 3))  # [harmonic, transient, noise]

        # 训练步数跟踪（用于温度退火）
        self._current_training_step = 0
        self.register_buffer('expert_update_counts', torch.zeros(n_experts))

    def set_training_step(self, step: int):
        """设置当前训练步数，用于温度退火等机制"""
        self._current_training_step = step

        # 由于使用统一架构，移除专家特定的学习率倍数
        # 让所有专家在相同条件下公平竞争和学习
        self.register_buffer('expert_lr_multipliers', torch.ones(n_experts))

        # 🔧 参数初始化稳定化 - 防止梯度爆炸
        self._init_parameters()

    def _init_parameters(self):
        """统一的简洁初始化策略 - 所有专家使用相同的稳定初始化"""
        for expert in self.experts:
            # 统一的标准初始化，让训练过程自然分化
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    # 使用标准Xavier初始化
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    # LayerNorm标准初始化
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    # 旧的复杂初始化方法已移除，现在使用统一简洁的初始化

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
        """Compute load balance loss to encourage uniform expert usage with enhanced numerical stability."""
        # 强制fp32计算避免bf16下的数值问题
        gate_logits = gate_logits.float().clamp_(-15, 15)  # 减小范围，增强稳定性
        probs = F.softmax(gate_logits, dim=-1)  # [B, E]
        mean_probs = probs.mean(dim=0)  # [E]

        # 修复: 添加温度缩放，防止极端分布（与路由器温度一致）
        temperature = 3.0  # 与路由器温度一致
        smooth_probs = F.softmax(gate_logits / temperature, dim=-1).mean(dim=0)

        # 使用更稳定的KL散度损失，但添加温度正则
        uniform = torch.full_like(mean_probs, 1.0 / self.n_experts)

        # 结合MSE和KL散度，增强稳定性
        mse_loss = F.mse_loss(smooth_probs, uniform)
        kl_loss = F.kl_div(
            torch.log(mean_probs + 1e-8),
            uniform,
            reduction='sum'
        ) / self.n_experts

        # 加权组合，MSE更稳定，KL更有效
        combined_loss = 0.7 * mse_loss + 0.3 * kl_loss

        # 确保损失为正且有意义的梯度，但限制上界防止爆炸
        return combined_loss.clamp(min=1e-8, max=1.0)

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

        # 2. Prepare enhanced router input with additional context
        # 基础声学特征
        base_router_input = acoustic_features  # [B, 64]

        # 添加全局统计特征增强路由决策
        with torch.no_grad():
            # 计算输入特征的全局统计
            h_mean = h.mean(dim=1)  # [B, D] - 序列均值
            h_std = h.std(dim=1)    # [B, D] - 序列方差
            h_max, _ = h.max(dim=1) # [B, D] - 序列最大值

            # 提取关键统计特征（降维到16维）
            global_stats = torch.cat([
                h_mean[:, :8],   # 前8维均值
                h_std[:, :4],    # 前4维标准差
                h_max[:, :4]     # 前4维最大值
            ], dim=-1)  # [B, 16]

        # 组合路由输入：纯音频特征路由，不使用CSI
        # 专家路由基于音频内容：Harmonic/Transient/BurstInpaint/LowSNR都从音频特征中分析
        router_input = torch.cat([base_router_input, global_stats], dim=-1)  # [B, 64+16] 纯音频特征

        # 3. Sample-level routing based on acoustic preferences
        sample_gate_logits = self.sample_router(router_input)  # [B, E]
        # 强制数值稳定性：fp32路由计算 + 安全归一化
        sample_gate_logits = sample_gate_logits.float().clamp_(-15, 15)  # fp32 + 更保守的截断

        # 修复：渐进式温度退火 - 从探索到专业化
        # 获取训练步数（如果可用）
        training_step = getattr(self, '_current_training_step', 0)

        # 温度退火策略：开始高温探索，逐步降温专业化
        initial_temp = 2.5    # 初始温度：适度探索
        final_temp = 1.0      # 最终温度：明确专业化
        annealing_steps = 5000  # 退火步数

        if training_step < annealing_steps:
            progress = training_step / annealing_steps
            routing_temperature = initial_temp - (initial_temp - final_temp) * progress
        else:
            routing_temperature = final_temp

        sample_gate_logits = sample_gate_logits / routing_temperature

        # >>> ROUTER EXPLORATION JITTER <<<
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
            # 强制数值稳定性：fp32路由计算 + 安全归一化
            token_gate_logits = token_gate_logits.float().clamp_(-15, 15)  # fp32 + 更保守的截断

            # 修复：添加温度缩放防止极端路由决策
            token_gate_logits = token_gate_logits / routing_temperature

            # >>> ROUTER EXPLORATION JITTER (token-level) <<<
            if self.training and getattr(self, 'router_jitter', 0.0) > 0.0:
                j = float(self.router_jitter)
                token_gate_logits = token_gate_logits + j * torch.randn_like(token_gate_logits)
            # <<< END JITTER <<<

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
            # Optional F0 conditioning vector from raw features
            f0_cond = None
            if x_raw is not None:
                try:
                    f0_cond = extract_acoustic_priors(x_raw)  # [B,6]
                except Exception:
                    f0_cond = None

            expert_outputs = []
            for i, expert in enumerate(self.experts):
                if (f0_cond is not None) and getattr(expert, 'use_f0_condition', False) and hasattr(expert, 'set_f0_condition'):
                    expert.set_f0_condition(f0_cond)
                expert_out = expert(h)  # [B, T, D]
                if hasattr(expert, 'clear_condition'):
                    expert.clear_condition()
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
            # Optional F0 conditioning vector from raw features
            f0_cond = None
            if x_raw is not None:
                try:
                    f0_cond = extract_acoustic_priors(x_raw)  # [B,6]
                except Exception:
                    f0_cond = None

            expert_outputs = []
            for i, expert in enumerate(self.experts):
                if (f0_cond is not None) and getattr(expert, 'use_f0_condition', False) and hasattr(expert, 'set_f0_condition'):
                    expert.set_f0_condition(f0_cond)
                expert_out = expert(h)  # [B, T, D]
                if hasattr(expert, 'clear_condition'):
                    expert.clear_condition()
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

    def get_expert_param_groups(self, base_lr: float = 1e-4):
        """获取专家参数组，统一学习率配置"""
        param_groups = []

        # 所有专家使用相同的学习率配置
        for i, expert in enumerate(self.experts):
            param_groups.append({
                'params': list(expert.parameters()),
                'lr': base_lr,
                'name': f'unified_expert_{i}',
                'weight_decay': 1e-6  # 统一的权重衰减
            })

        # 其他模块使用基础学习率
        other_params = []
        for name, module in self.named_children():
            if name != 'experts':
                other_params.extend(list(module.parameters()))

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other_modules',
                'weight_decay': 1e-6
            })

        return param_groups

    def get_routing_analysis(self) -> dict:
        """分析路由器的学习成果，揭示专家特化方向"""
        analysis = {
            'expert_utilization': self.get_expert_utilization().tolist(),
            'routing_entropy': None,
            'specialization_info': {}
        }

        # 计算路由熵（多样性指标）
        util = self.get_expert_utilization()
        if util.sum() > 0:
            util_norm = util / (util.sum() + 1e-8)
            routing_entropy = -(util_norm * torch.log(util_norm + 1e-8)).sum().item()
            analysis['routing_entropy'] = routing_entropy

        # 分析每个专家的特化情况
        for i in range(len(self.experts)):
            expert_util = util[i].item()
            if expert_util > 0.1:  # 只分析使用率超过10%的专家
                specialization = "unknown"
                if i == 0 and expert_util > 0.4:
                    specialization = "likely_harmonic"  # 谐波特化
                elif i == 1 and expert_util > 0.4:
                    specialization = "likely_transient"  # 瞬态特化
                elif i == 2 and expert_util > 0.4:
                    specialization = "likely_inpaint"  # 修复特化

                analysis['specialization_info'][f'expert_{i}'] = {
                    'utilization': expert_util,
                    'likely_specialization': specialization,
                    'parameters': sum(p.numel() for p in self.experts[i].parameters())
                }

        return analysis

    def print_routing_summary(self):
        """打印路由器学习成果的简洁摘要"""
        analysis = self.get_routing_analysis()

        print("[MoE Router Learning Summary]")
        print(f"   Expert Utilization: {[f'{u:.3f}' for u in analysis['expert_utilization']]}")
        print(f"   Routing Entropy: {analysis.get('routing_entropy', 0.0):.3f} (higher = more diverse)")

        for expert_id, info in analysis['specialization_info'].items():
            specialization = info['likely_specialization']
            util = info['utilization']
            print(f"   {expert_id}: {util:.1%} usage -> {specialization}")


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

        # 纯音频特征路由：与forward方法保持一致，添加global_stats
        # 计算global_stats - 与forward方法完全一致
        h_mean = h.mean(dim=1)  # [B, D]
        h_std = h.std(dim=1, unbiased=False)  # [B, D]
        h_max = h.max(dim=1)[0]  # [B, D]

        # 确保取足够的维度
        n_dims = min(4, h_mean.size(-1))  # 避免超出h的实际维度
        global_stats = torch.cat([
            h_mean[:, :n_dims],   # 前n_dims维均值
            h_std[:, :n_dims],    # 前n_dims维标准差
            h_max[:, :n_dims]     # 前n_dims维最大值
        ], dim=-1)  # [B, n_dims*3]

        # 如果global_stats不足16维，补零到16维
        if global_stats.size(-1) < 16:
            padding = torch.zeros(global_stats.size(0), 16 - global_stats.size(-1), device=global_stats.device, dtype=global_stats.dtype)
            global_stats = torch.cat([global_stats, padding], dim=-1)

        router_input = torch.cat([acoustic_features, global_stats], dim=-1)  # [B, acoustic_dim+16] 与forward一致

        # 2) Sample-level 平衡损失
        sample_gate_logits = self.sample_router(router_input).float().clamp_(-20, 20)  # 数值稳定
        aux_losses['moe_balance_loss'] = self.load_balance_loss(sample_gate_logits)     # 维持为 tensor

        # 3) Token-level 平衡损失（若启用 token router）
        if self.use_token_level:
            token_gate_logits = self.token_router(h).float().clamp_(-20, 20)
            e = token_gate_logits.shape[-1]
            token_gate_flat = token_gate_logits.view(-1, e)                              # 关键：避免 UnboundLocalError
            aux_losses['moe_token_balance_loss'] = self.load_balance_loss(token_gate_flat)

        # 4) 专家差异化损失 - 临时禁用以减少不必要约束
        # diversification_loss = self.compute_expert_diversification_loss()
        # aux_losses['expert_diversification_loss'] = diversification_loss
        # 注释：当前差异化损失过高(moe_d=0.0112)，可能阻碍专家系统性能提升

        # 5) 可选的偏好/稀疏度指标（仅做监控，不直接入 loss）
        if x_raw is not None and dual_streams is not None:
            with torch.no_grad():
                aux_losses['moe_harmonic_pref'] = acoustic_features.mean()
        return aux_losses

    def compute_expert_diversification_loss(self) -> torch.Tensor:
        """计算专家差异化损失，鼓励专家学会不同的特征表示"""
        # 收集所有专家的关键参数
        expert_biases = []
        expert_spec_weights = []

        for expert in self.experts:
            expert_biases.append(expert.expert_bias.flatten())  # [D]
            expert_spec_weights.append(expert.specialization_weights.flatten())  # [D]

        # 将专家参数堆叠 [n_experts, D]
        biases_stack = torch.stack(expert_biases, dim=0)  # [E, D]
        spec_weights_stack = torch.stack(expert_spec_weights, dim=0)  # [E, D]

        # 计算专家间的相似度矩阵
        bias_similarity = torch.mm(biases_stack, biases_stack.t())  # [E, E]
        spec_similarity = torch.mm(spec_weights_stack, spec_weights_stack.t())  # [E, E]

        # 去除对角线（专家与自己的相似度）
        mask = ~torch.eye(self.n_experts, device=biases_stack.device, dtype=torch.bool)
        bias_off_diag = bias_similarity[mask]
        spec_off_diag = spec_similarity[mask]

        # 鼓励专家间差异化：相似度越小越好
        bias_div_loss = torch.relu(bias_off_diag).mean()  # 惩罚正相似度
        spec_div_loss = torch.relu(spec_off_diag).mean()  # 惩罚正相似度

        # 组合差异化损失
        total_div_loss = 0.5 * bias_div_loss + 0.5 * spec_div_loss

        return total_div_loss.clamp(min=0.0, max=1.0)  # 限制范围，防止爆炸



class CompatibleMicroMoE(nn.Module):
    """向后兼容的MicroMoE接口 - 包装SpecializedMicroMoE或EnhancedMicroMoEWithBypass"""
    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2, **kwargs):
        super().__init__()

        # 检查是否启用直流通路
        enable_direct_pathway = kwargs.pop('enable_direct_pathway', False)

        if enable_direct_pathway:
            # 使用增强版本EnhancedMicroMoEWithBypass
            enhanced_kwargs = {
                'D': d_model,
                'n_experts': n_experts,
                'topk': top_k,
            }
            # 传递增强版本的kwargs
            for key, value in kwargs.items():
                if key in ['d_csi', 'expert_dropout', 'balance_weight', 'router_use_csi', 'use_token_level',
                          'initial_bypass_weight', 'adaptive_threshold', 'pathway_warmup_steps']:
                    enhanced_kwargs[key] = value

            # 添加默认的直流通路参数
            enhanced_kwargs['enable_direct_pathway'] = True
            self.specialized_moe = EnhancedMicroMoEWithBypass(**enhanced_kwargs)
            self._is_enhanced = True
        else:
            # 使用标准版本SpecializedMicroMoE
            specialized_kwargs = {
                'D': d_model,
                'n_experts': n_experts,
                'topk': top_k,
            }
            # 传递其他kwargs
            for key, value in kwargs.items():
                if key in ['d_csi', 'expert_dropout', 'balance_weight', 'router_use_csi', 'use_token_level']:
                    specialized_kwargs[key] = value

            self.specialized_moe = SpecializedMicroMoE(**specialized_kwargs)
            self._is_enhanced = False

    def forward(self, h: torch.Tensor, router_input: torch.Tensor, x_raw: torch.Tensor = None, dual_streams: dict = None, training_step: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        if self._is_enhanced:
            # 调用增强版本，传递training_step
            output = self.specialized_moe(h, csi_vec, x_raw=x_raw, dual_streams=dual_streams, training_step=training_step)
        else:
            # 调用标准版本
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

    def get_performance_stats(self):
        """获取性能统计信息（仅增强版本）"""
        if self._is_enhanced and hasattr(self.specialized_moe, 'get_performance_stats'):
            return self.specialized_moe.get_performance_stats()
        return {}

    def get_separated_outputs(self):
        """获取分离的输出用于损失计算（仅增强版本）"""
        if self._is_enhanced and hasattr(self.specialized_moe, 'get_separated_outputs'):
            return self.specialized_moe.get_separated_outputs()
        return None, None

    def update_performance_ema(self, direct_loss: float, expert_loss: float):
        """更新性能EMA（仅增强版本）"""
        if self._is_enhanced and hasattr(self.specialized_moe, 'update_performance_ema'):
            self.specialized_moe.update_performance_ema(direct_loss, expert_loss)


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


class EnhancedMicroMoEWithBypass(nn.Module):
    """带直流通路的增强MicroMoE - 用于性能对比验证

    核心特性:
    - 可配置的直流通路，支持绕过专家系统
    - 动态权重调度，基于性能自适应调整专家vs直流权重
    - 分离损失计算，独立监控专家和直流通路性能
    - 渐进式启用，避免训练初期专家路由次优解
    """

    def __init__(
        self,
        D: int = 128,
        d_csi: int = 10,
        n_experts: int = 3,
        topk: int = 2,
        expert_dropout: float = 0.0,
        balance_weight: float = 0.5,
        router_use_csi: bool = True,
        use_token_level: bool = True,
        # 直流通路相关参数
        enable_direct_pathway: bool = True,
        initial_bypass_weight: float = 0.8,
        adaptive_threshold: float = 0.05,
        pathway_warmup_steps: int = 1500,
    ):
        super().__init__()

        # 基础MoE配置
        self.d_model = D
        self.enable_direct_pathway = enable_direct_pathway
        self.adaptive_threshold = adaptive_threshold
        self.pathway_warmup_steps = pathway_warmup_steps

        # 原始专家系统
        self.moe_system = SpecializedMicroMoE(
            D=D, d_csi=d_csi, n_experts=n_experts, topk=topk,
            expert_dropout=expert_dropout, balance_weight=balance_weight,
            router_use_csi=router_use_csi, use_token_level=use_token_level
        )

        # 为兼容性添加acoustic_extractor别名
        self.acoustic_extractor = self.moe_system.acoustic_extractor

        # 直流通路模块
        if enable_direct_pathway:
            # Stage1等效的真正直通路径：简单而高效
            self.simple_direct_pathway = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D),
                nn.GELU(),
                nn.Linear(D, D)
            )

            # 传统直通路径（向后兼容）
            self.direct_pathway = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D),
                nn.GELU(),
                nn.Linear(D, D)
            )

            # 可学习的权重参数
            self.bypass_weight = nn.Parameter(torch.tensor(initial_bypass_weight))

            # 性能监控缓冲区
            self.register_buffer('expert_loss_ema', torch.tensor(float('inf')))
            self.register_buffer('direct_loss_ema', torch.tensor(float('inf')))
            self.register_buffer('training_step_count', torch.tensor(0))

            # 输出缓存用于分离损失计算
            self._last_direct_output = None
            self._last_expert_output = None

    def compute_pathway_weights(self, training_step: int):
        """基于训练步数和性能动态调整直流与专家权重，支持架构级绕过

        Args:
            training_step: 当前训练步数

        Returns:
            tuple: (bypass_weight, expert_weight)
        """
        if not self.enable_direct_pathway:
            return 0.0, 1.0

        # Stage3专门策略：快速激活expert系统
        if training_step < 500:
            # 前500步最小启动期：使用较低bypass权重
            return 0.4, 0.6  # 优先expert模式

        # 阶段2: 快速过渡到expert主导
        elif training_step < self.pathway_warmup_steps:
            # 快速衰减到专家主导模式
            progress = (training_step - 500) / max(1, self.pathway_warmup_steps - 500)
            bypass_weight = 0.4 - 0.1 * progress  # 从0.4降到0.3
            expert_weight = 1.0 - bypass_weight
            return bypass_weight, expert_weight

        # Stage3训练策略：大幅延长expert训练期
        if training_step < 15000:  # 从5000延长到15000步
            # 前15000步强制使用expert，几乎禁用性能比较
            self.bypass_weight.data = torch.clamp(self.bypass_weight.data * 0.998, 0.1, 0.3)
        elif training_step < 25000:  # 25000步内保守调整
            # 缓慢过渡期：放宽性能比较阈值
            if (torch.isfinite(self.expert_loss_ema) and torch.isfinite(self.direct_loss_ema) and
                self.expert_loss_ema < float('inf') and self.direct_loss_ema < float('inf')):

                # 计算性能比率 (专家 / 直流)
                performance_ratio = self.expert_loss_ema / (self.direct_loss_ema + 1e-8)

                # 放宽阈值：只有在专家真正显著更好时才降低bypass
                if performance_ratio < 0.8:  # 专家必须比direct好20%以上
                    self.bypass_weight.data *= 0.995  # 非常保守的衰减
                # 只有在性能比率超过2.0时才增加bypass（比之前的1.1宽松很多）
                elif performance_ratio > 2.0:
                    self.bypass_weight.data *= 1.001  # 极保守的增加
        else:
            # 25000步后才开始正常的性能比较
            if (torch.isfinite(self.expert_loss_ema) and torch.isfinite(self.direct_loss_ema) and
                self.expert_loss_ema < float('inf') and self.direct_loss_ema < float('inf')):

                performance_ratio = self.expert_loss_ema / (self.direct_loss_ema + 1e-8)

                if performance_ratio < (1.0 - self.adaptive_threshold):
                    self.bypass_weight.data *= 0.99
                elif performance_ratio > (1.0 + self.adaptive_threshold):
                    self.bypass_weight.data *= 1.002

        # 权重限制：强制应用clamp
        self.bypass_weight.data = torch.clamp(self.bypass_weight.data, 0.1, 0.85)
        bypass_weight = torch.clamp(self.bypass_weight, 0.1, 0.85)
        expert_weight = 1.0 - bypass_weight

        return float(bypass_weight), float(expert_weight)

    def forward(
        self,
        h: torch.Tensor,
        csi_vec: torch.Tensor = None,
        x_raw: torch.Tensor = None,
        dual_streams: dict = None,
        training_step: int = 0
    ) -> torch.Tensor:
        """增强前向传播，支持架构级直通路径与专家系统混合

        Args:
            h: [B, T, D] 输入特征
            csi_vec: [B, d_csi] 信道状态信息
            x_raw: [B, T, 36] 原始声学特征
            dual_streams: 双流特征字典
            training_step: 当前训练步数

        Returns:
            torch.Tensor: [B, T, D] 处理后的特征
        """
        if not self.enable_direct_pathway:
            # 标准MoE模式
            return self.moe_system(h, csi_vec, x_raw, dual_streams)

        # 更新步数计数
        if self.training:
            self.training_step_count += 1

        # 1. 计算动态权重
        bypass_weight, expert_weight = self.compute_pathway_weights(training_step)

        # 2. 架构级绕过：当直通权重 >= 0.9时，完全跳过MoE系统
        if bypass_weight >= 0.9:
            # Stage1等效模式：只使用简单直通路径，避免MoE复杂度
            output = self.simple_direct_pathway(h)
            if self.training:
                # 在架构级绕过模式下，模拟专家输出用于统计
                self._last_direct_output = output.detach().clone()
                self._last_expert_output = output.detach().clone()  # 模拟相同输出
                self._last_bypass_weight = 1.0
                self._last_expert_weight = 0.0
            return output

        # 3. 混合模式：计算两个路径（当bypass_weight在0.1-0.9之间）
        if bypass_weight > 0.1:
            # 直通路径：使用简单版本确保与Stage1等效
            direct_output = self.simple_direct_pathway(h)  # [B, T, D]

            # 专家系统路径：完整MoE计算
            expert_output = self.moe_system(h, csi_vec, x_raw, dual_streams)  # [B, T, D]

            # 加权融合
            mixed_output = bypass_weight * direct_output + expert_weight * expert_output

            # 缓存输出用于分离损失计算
            if self.training:
                self._last_direct_output = direct_output.detach().clone()
                self._last_expert_output = expert_output.detach().clone()
                self._last_bypass_weight = bypass_weight
                self._last_expert_weight = expert_weight

            return mixed_output

        # 4. 纯专家模式：完全使用MoE系统
        expert_output = self.moe_system(h, csi_vec, x_raw, dual_streams)
        if self.training:
            self._last_direct_output = None  # 不使用直通路径
            self._last_expert_output = expert_output.detach().clone()
            self._last_bypass_weight = 0.0
            self._last_expert_weight = 1.0

        return expert_output

    def update_performance_ema(self, direct_loss: float, expert_loss: float, alpha: float = 0.99):
        """更新性能EMA用于权重调整

        Args:
            direct_loss: 直流通路损失
            expert_loss: 专家系统损失
            alpha: EMA衰减系数
        """
        if self.enable_direct_pathway and self.training:
            # 安全的EMA更新
            if torch.isfinite(self.direct_loss_ema) and self.direct_loss_ema < float('inf'):
                self.direct_loss_ema = alpha * self.direct_loss_ema + (1 - alpha) * direct_loss
            else:
                self.direct_loss_ema = torch.tensor(direct_loss)

            if torch.isfinite(self.expert_loss_ema) and self.expert_loss_ema < float('inf'):
                self.expert_loss_ema = alpha * self.expert_loss_ema + (1 - alpha) * expert_loss
            else:
                self.expert_loss_ema = torch.tensor(expert_loss)

    def get_performance_stats(self) -> dict:
        """获取性能统计信息用于监控

        Returns:
            dict: 包含权重、损失EMA、性能比率等统计信息
        """
        stats = {}

        if self.enable_direct_pathway:
            current_bypass = float(self.bypass_weight.data)

            stats.update({
                'bypass_weight': current_bypass,
                'expert_weight': 1.0 - current_bypass,
                'direct_loss_ema': float(self.direct_loss_ema) if torch.isfinite(self.direct_loss_ema) else float('inf'),
                'expert_loss_ema': float(self.expert_loss_ema) if torch.isfinite(self.expert_loss_ema) else float('inf'),
                'training_steps': int(self.training_step_count),
            })

            # 架构级绕过状态
            if current_bypass >= 0.9:
                stats['pathway_mode'] = 'architectural_bypass'
                stats['stage1_equivalent'] = True
            elif current_bypass > 0.1:
                stats['pathway_mode'] = 'mixed'
                stats['stage1_equivalent'] = False
            else:
                stats['pathway_mode'] = 'pure_expert'
                stats['stage1_equivalent'] = False

            # 性能比率
            if (torch.isfinite(self.expert_loss_ema) and torch.isfinite(self.direct_loss_ema) and
                self.direct_loss_ema > 0):
                stats['performance_ratio'] = float(self.expert_loss_ema / self.direct_loss_ema)
            else:
                stats['performance_ratio'] = 1.0

            # 计算复杂度估算（相对于Stage1）
            if stats['pathway_mode'] == 'architectural_bypass':
                stats['complexity_ratio'] = 1.0  # 与Stage1相同
            elif stats['pathway_mode'] == 'mixed':
                stats['complexity_ratio'] = 1.0 + current_bypass * 2.0  # 估算MoE开销
            else:
                stats['complexity_ratio'] = 3.0  # 纯专家模式开销

        # 添加专家系统统计
        try:
            expert_util = self.moe_system.get_expert_utilization()
            stats['expert_utilization'] = [float(u) for u in expert_util]
        except:
            stats['expert_utilization'] = [0.0] * getattr(self.moe_system, 'n_experts', 3)

        return stats

    def get_separated_outputs(self):
        """获取分离的输出用于损失计算

        Returns:
            tuple: (direct_output, expert_output) 或 (None, None)
        """
        if (self.enable_direct_pathway and hasattr(self, '_last_direct_output') and
            hasattr(self, '_last_expert_output')):
            return self._last_direct_output, self._last_expert_output
        return None, None

    def get_expert_utilization(self):
        """代理专家利用率获取"""
        return self.moe_system.get_expert_utilization()

    def get_aux_losses(self, h: torch.Tensor, csi_vec: torch.Tensor = None,
                       x_raw: torch.Tensor = None, dual_streams: dict = None) -> dict:
        """获取辅助损失，包括MoE平衡损失等"""
        return self.moe_system.get_aux_losses(h, csi_vec, x_raw, dual_streams)


if __name__ == "__main__":
    pass
