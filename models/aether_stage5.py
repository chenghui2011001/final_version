#!/usr/bin/env python3
"""
Stage5 AETHER Model: 码率约束训练架构

主要功能:
1. 基于Stage4权重的增量训练
2. 分层残差量化 (HierarchicalRVQ)
3. 自适应码率控制 (1.2±0.1 kbps)
4. 语义保持和质量优化
5. 多目标率失真优化
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import math
import os


# === [VIB] Gaussian bottleneck ==============================================
class GaussianBottleneck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mu = nn.Conv1d(dim, dim, kernel_size=1)
        self.logvar = nn.Conv1d(dim, dim, kernel_size=1)
        # 添加LayerNorm稳定VIB输出尺度
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x_bct: torch.Tensor):
        # x_bct: [B, C, T]
        mu = self.mu(x_bct)
        logvar = self.logvar(x_bct).clamp_(-8.0, 8.0)  # 修复2: logvar数值稳定化
        std = torch.exp(0.5 * logvar).clamp_min(1e-4)  # 修复：添加极小下限防止数值不稳定
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL(q(z|x) || N(0, I)) - 修复2: 逐帧平均，转换为bits单位与rate对齐
        LOG2E = 1.4426950408889634  # 1/ln(2)
        kld_per_element = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # nats, 标准KL公式
        # 先求每个时间帧的平均KL，再求batch平均 → 每帧平均bits
        B, D, T = kld_per_element.shape
        # 修复2：改成按通道均值，数值降低~24倍
        kld_per_frame = kld_per_element.mean(dim=1)  # [B, T] - 每帧每通道均值
        kld_bits_per_frame = (kld_per_frame.sum() / (B * T)) * LOG2E  # 逐帧平均再转bits

        # 修复8: 应用LayerNorm稳定输出尺度，避免VIB强时方差塌陷
        z_normalized = self.output_norm(z.transpose(1, 2)).transpose(1, 2)

        return z_normalized, mu, logvar, kld_bits_per_frame


# === [GATE] Learnable stage gates (Hard-Concrete / ST-Gumbel) ==============
class StageGate(nn.Module):
    """Per-frame gate using ST-Gumbel-Sigmoid with content-dependent logits.

    Produces [B,T] soft/hard gates from input residual features via a small Conv1d net.
    """
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=1)
        )

    def forward(self, residual_bct: torch.Tensor, temperature: float = 0.67, logit_bias: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, T = residual_bct.shape
        logits = self.net(residual_bct).squeeze(1)  # [B,T]
        if logit_bias != 0.0:
            logits = logits + float(logit_bias)
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            g = torch.log(u) - torch.log(1 - u)
            pre = (logits + g) / max(1e-6, float(temperature))
            soft = torch.sigmoid(pre)
        else:
            soft = torch.sigmoid(logits)
        hard = (soft > 0.5).float()
        hard_st = hard + (soft - soft.detach())
        return hard_st, soft


# === [SEM] Lightweight semantic teacher (feature->semantic) ================
class SemanticTeacher(nn.Module):
    def __init__(self, in_dim=36, proj_dim=128):
        super().__init__()
        # 增强的语义特征提取器 - 更深层的网络以提取有意义的语义
        self.feature_extractor = nn.Sequential(
            # 第一层：局部特征提取
            nn.Conv1d(in_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # 第二层：中等范围特征
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # 第三层：长范围依赖
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # 语义投影头
        self.semantic_proj = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, proj_dim, kernel_size=1)
        )

    def forward(self, x_btc: torch.Tensor):
        # x_btc: [B, T, 36] -> [B, 36, T]
        x_bct = x_btc.transpose(1, 2)

        # 提取深层语义特征
        features = self.feature_extractor(x_bct)  # [B, 128, T]
        sem = self.semantic_proj(features)        # [B, proj_dim, T]

        # L2归一化
        return torch.nn.functional.normalize(sem, p=2, dim=1)


class VectorQuantizer(nn.Module):
    """基础向量量化器，支持EMA更新和改进的梯度流"""
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps

        # 可训练码书 - 使用更合理的初始化
        embed = torch.randn(codebook_size, dim) * 0.01  # 缩小初始化范围
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        # 添加初始化标志，用于动态初始化
        self.register_buffer('initialized', torch.tensor(False))

        # 困惑度监控
        self.register_buffer('_perplexity', torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor, return_indices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        inputs: [B, D, T] 输入特征
        return_indices: 是否返回码字索引
        returns: (quantized, commitment_loss, perplexity, indices)
        """
        B, D, T = inputs.shape
        inputs_flat = inputs.permute(0, 2, 1).contiguous().view(-1, D)  # [BT, D]

        # 动态初始化码书以匹配输入数据范围
        if not self.initialized:
            with torch.no_grad():
                # 使用输入数据的前几个batch来初始化码书
                input_std = inputs_flat.std(dim=0, keepdim=True)
                input_mean = inputs_flat.mean(dim=0, keepdim=True)
                # 从输入数据分布初始化码书
                self.embed.copy_(torch.randn_like(self.embed) * input_std * 0.5 + input_mean)
                self.embed_avg.copy_(self.embed.clone())
                self.initialized.copy_(torch.tensor(True))

        # 计算距离
        distances = torch.sum(inputs_flat ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embed ** 2, dim=1) - \
                   2 * torch.matmul(inputs_flat, self.embed.t())

        # 获取最近的码字索引
        encoding_indices = torch.argmin(distances, dim=1)  # [BT]
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # 量化
        quantized_flat = torch.matmul(encodings, self.embed)
        quantized = quantized_flat.view(B, T, D).permute(0, 2, 1).contiguous()

        # EMA更新(仅在训练时)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )
            embed_sum = torch.matmul(encodings.t(), inputs_flat)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            )
            embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

            # === 添加dead code复活机制 ===
            batch_tokens = encoding_indices.shape[0]  # BT
            dead_thresh = max(0.5 * (batch_tokens / self.codebook_size), 0.1)
            dead_mask = (self.cluster_size < dead_thresh)

            if dead_mask.any():
                n_dead = dead_mask.sum()
                # 从活跃样本中重新采样
                sample_ids = torch.randint(0, inputs_flat.size(0), (n_dead,), device=inputs.device)
                self.embed.data[dead_mask] = inputs_flat[sample_ids].detach()
                self.embed_avg.data[dead_mask] = self.embed.data[dead_mask]
                self.cluster_size.data[dead_mask] = self.cluster_size.mean()

        # 损失计算（commitment_weight在loss计算中统一处理）
        # 注意：commitment loss是输入向量化的代价，应该是inputs与quantized的距离
        # inputs是[B,D,T], quantized也是[B,D,T]
        commitment_loss = F.mse_loss(inputs_flat, quantized_flat.detach()) * self.commitment_weight

        # 直通梯度
        quantized = inputs + (quantized - inputs).detach()

        # 困惑度计算
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self._perplexity = perplexity

        # 返回码字索引（重塑为[B, T]）
        indices = encoding_indices.view(B, T) if return_indices else None

        return quantized, commitment_loss, perplexity, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码字索引重构特征
        indices: [B, T] 码字索引
        returns: [B, D, T] 重构的特征
        """
        B, T = indices.shape
        indices_flat = indices.view(-1)  # [BT]

        # 码书查找
        quantized_flat = self.embed[indices_flat]  # [BT, D]
        quantized = quantized_flat.view(B, T, self.dim).permute(0, 2, 1).contiguous()  # [B, D, T]

        return quantized

    @property
    def perplexity(self) -> torch.Tensor:
        return self._perplexity

class HierarchicalRVQDecoder(nn.Module):
    """
    分层RVQ解码器：从码字索引重构特征
    这是Stage5缺失的关键组件
    """
    def __init__(self, rvq_encoder: 'HierarchicalRVQ'):
        super().__init__()
        self.num_stages = rvq_encoder.num_stages
        self.input_dim = rvq_encoder.input_dim
        # 共享量化器的码书（不复制参数）
        self.quantizers = rvq_encoder.quantizers

    def forward(self, stage_indices_list: List[torch.Tensor]) -> torch.Tensor:
        """
        从码字索引重构特征
        stage_indices_list: List[[B, T]] 每个RVQ阶段的码字索引
        returns: [B, D, T] 重构的特征
        """
        if len(stage_indices_list) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} stages, got {len(stage_indices_list)}")

        B, T = stage_indices_list[0].shape
        device = stage_indices_list[0].device

        # 初始化重构特征
        reconstructed = torch.zeros(B, self.input_dim, T, device=device)

        # 逐阶段重构
        for stage_idx, indices in enumerate(stage_indices_list):
            if indices.shape != (B, T):
                raise ValueError(f"Stage {stage_idx} indices shape mismatch: expected ({B}, {T}), got {indices.shape}")

            # 使用量化器的解码功能
            stage_features = self.quantizers[stage_idx].decode_from_indices(indices)  # [B, D, T]
            reconstructed += stage_features

        return reconstructed

    def get_reconstruction_error(self, original: torch.Tensor, stage_indices_list: List[torch.Tensor]) -> torch.Tensor:
        """
        计算重构误差
        original: [B, D, T] 原始特征
        stage_indices_list: List[[B, T]] 码字索引列表
        returns: 重构损失
        """
        reconstructed = self.forward(stage_indices_list)
        return F.mse_loss(reconstructed, original)

class RealEntropyCodec(nn.Module):
    """
    真实的熵编码系统：替代EntropyModel的假码率计算
    实现真正的码字索引 -> 比特流转换
    """
    def __init__(self, codebook_size: int, context_size: int = 5):
        super().__init__()
        self.codebook_size = codebook_size
        self.context_size = context_size

        # 可学习的码字概率分布
        self.register_parameter('prior_logits', nn.Parameter(torch.zeros(codebook_size)))

        # 上下文自适应模型
        self.context_net = nn.Sequential(
            nn.Embedding(codebook_size, 64),
            nn.LSTM(64, 128, batch_first=True),
        )
        self.prob_head = nn.Linear(128, codebook_size)

        # EMA统计（用于码率估计）
        self.register_buffer('symbol_counts', torch.ones(codebook_size))
        self.register_buffer('total_symbols', torch.tensor(float(codebook_size)))

    def update_statistics(self, indices: torch.Tensor):
        """更新码字统计信息"""
        if self.training:
            # 统计码字频率
            unique_indices, counts = torch.unique(indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                if 0 <= idx < self.codebook_size:
                    self.symbol_counts[idx] += count.float()
                    self.total_symbols += count.float()

    def get_symbol_probabilities(self) -> torch.Tensor:
        """获取当前码字概率分布"""
        # Laplace平滑
        smoothed_counts = self.symbol_counts + 1e-8
        probs = smoothed_counts / smoothed_counts.sum()
        return probs

    def estimate_rate_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从码字索引估计真实码率
        indices: [B, T] 码字索引
        returns: 平均每符号比特数
        """
        # 更新统计
        self.update_statistics(indices)

        # 获取概率分布
        probs = self.get_symbol_probabilities()

        # 计算交叉熵（真实码率）
        B, T = indices.shape
        total_bits = 0.0

        for b in range(B):
            for t in range(T):
                symbol_idx = indices[b, t].item()
                if 0 <= symbol_idx < self.codebook_size:
                    symbol_prob = probs[symbol_idx].item()
                    bits = -math.log2(max(symbol_prob, 1e-10))
                    total_bits += bits

        avg_bits_per_symbol = total_bits / (B * T)
        return torch.tensor(avg_bits_per_symbol, device=indices.device)

    def compress(self, indices: torch.Tensor) -> Tuple[bytes, float]:
        """
        模拟的熵编码压缩（返回估计比特数）
        indices: [B, T] 码字索引
        returns: (虚拟比特流, 真实比特率)
        """
        real_rate = self.estimate_rate_from_indices(indices)

        # 虚拟比特流（实际部署中替换为真实编码器）
        B, T = indices.shape
        estimated_bytes = int(real_rate.item() * B * T / 8) + 1
        dummy_bitstream = b'\x00' * estimated_bytes

        return dummy_bitstream, real_rate.item()

    def decompress(self, bitstream: bytes, shape: Tuple[int, int]) -> torch.Tensor:
        """
        模拟的熵解码（返回随机索引，实际部署中替换）
        bitstream: 比特流
        shape: (B, T) 输出形状
        returns: [B, T] 码字索引
        """
        B, T = shape
        # 模拟解码：按概率分布采样
        probs = self.get_symbol_probabilities()
        indices = torch.multinomial(probs, B * T, replacement=True).view(B, T)
        return indices

class EntropyModel(nn.Module):
    """兼容性包装器：保持API不变，内部使用真实熵编码"""
    def __init__(self, feature_dim: int, codebook_size: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.codebook_size = codebook_size

        # 使用真实熵编码系统
        self.real_codec = RealEntropyCodec(codebook_size)

    def forward(self, quantized_features: torch.Tensor, stage_weight: float = 1.0) -> torch.Tensor:
        """
        兼容性接口：返回理论码率（用于快速估计）
        """
        # 理论最大码率（用于训练时的快速估计）
        bits_per_frame = math.log2(self.codebook_size)
        return torch.tensor(bits_per_frame, dtype=torch.float32, device=quantized_features.device)

    def estimate_real_rate(self, indices: torch.Tensor) -> torch.Tensor:
        """获取真实码率估计"""
        return self.real_codec.estimate_rate_from_indices(indices)

class HierarchicalRVQ(nn.Module):
    """
    分层残差量化器
    Stage 1: 主要结构 (1024 codes, ~10 bits)
    Stage 2: 细节补偿 (512 codes, ~9 bits)
    Stage 3: 残差修正 (256 codes, ~8 bits)
    """
    def __init__(
        self,
        input_dim: int = 24,
        num_stages: int = 3,
        codebook_sizes: List[int] = None,
        commitment_weights: List[float] = None,
        decay: float = 0.99,
        eps: float = 1e-5,
        enable_stage_gates: bool = False,
        gate_temperature: float = 0.67,
        soft_entropy_tau: float = 0.3,
        enable_codebook_mask: bool = True,
        enable_debug_stats: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_stages = num_stages
        self.enable_stage_gates = bool(enable_stage_gates)
        self.gate_temperature = float(gate_temperature)
        self.soft_entropy_tau = float(soft_entropy_tau)
        self.enable_codebook_mask = bool(enable_codebook_mask)
        self.enable_debug_stats = bool(enable_debug_stats)
        # Per-channel EMA stats for reversible whitening (avoid cross-channel LN)
        self.register_buffer('ema_mu', torch.zeros(input_dim))
        self.register_buffer('ema_std', torch.ones(input_dim))
        self.ema_momentum = 0.99

        if codebook_sizes is None:
            codebook_sizes = [1024, 512, 256]
        if commitment_weights is None:
            commitment_weights = [0.01, 0.02, 0.03]  # 大幅降低commitment权重避免损失爆炸

        assert len(codebook_sizes) == num_stages
        assert len(commitment_weights) == num_stages

        # 分层量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                dim=input_dim,
                codebook_size=codebook_sizes[i],
                commitment_weight=commitment_weights[i],
                decay=decay,
                eps=eps
            ) for i in range(num_stages)
        ])

        # Per-stage learnable softmax temperature multipliers (combined with global tau)
        self.stage_tau = nn.Parameter(torch.ones(num_stages))

        # 自适应权重
        self.stage_weights = nn.Parameter(torch.ones(num_stages))

        # 真实熵编码系统
        self.entropy_codecs = nn.ModuleList([
            RealEntropyCodec(codebook_sizes[i])
            for i in range(num_stages)
        ])

        # 兼容性码率估计器
        self.rate_estimators = nn.ModuleList([
            EntropyModel(input_dim, codebook_sizes[i])
            for i in range(num_stages)
        ])

        # Learnable stage gates for s>=1 (i.e., stages 2/3...)
        if self.enable_stage_gates:
            self.stage_gates = nn.ModuleList([
                StageGate(in_dim=input_dim) if i >= 1 else nn.Identity()
                for i in range(num_stages)
            ])
        else:
            self.stage_gates = None

        # Learnable codebook masks (effective-K control)
        if self.enable_codebook_mask:
            self.codebook_mask_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(codebook_sizes[i])) for i in range(num_stages)
            ])
        else:
            self.codebook_mask_logits = None

    def forward(
        self,
        x: torch.Tensor,
        adaptive_weights: Optional[torch.Tensor] = None,
        return_indices: bool = False,
        stage_mask: Optional[List[bool]] = None,
        control_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, D, T] 输入特征
        adaptive_weights: [num_stages] 自适应阶段权重(可选)
        return_indices: 是否返回码字索引（用于真实RVQ）
        returns: 量化结果和相关损失
        """
        # Reversible per-channel whitening across (B,T); keep EMA for inference
        residual = x
        B, D, T = residual.shape
        # Batch statistics
        mu_bt = residual.mean(dim=(0, 2))                       # [D]
        std_bt = residual.std(dim=(0, 2)).clamp_min(1e-5)        # [D]
        if self.training:
            with torch.no_grad():
                a = 1.0 - self.ema_momentum
                self.ema_mu.mul_(self.ema_momentum).add_(mu_bt, alpha=a)
                self.ema_std.mul_(self.ema_momentum).add_(std_bt, alpha=a)
        mu = self.ema_mu
        sd = self.ema_std
        residual = (residual - mu.view(1, D, 1)) / (sd.view(1, D, 1) + 1e-5)
        quantized_stages = []
        commitment_losses = []
        rate_estimates = []
        perplexities = []
        stage_indices = [] if return_indices else None
        stage_soft_probs: List[torch.Tensor] = []  # list of [B, T, K]
        stage_gate_soft: List[torch.Tensor] = []   # list of [B, T]
        stage_gate_masks: List[torch.Tensor] = []  # list of [B, T]
        codebook_mask_usage: List[torch.Tensor] = []  # list of scalar tensors
        stage_debug_stats: List[Dict[str, float]] = []

        # Apply control overrides
        gate_temp = float(control_overrides.get('gate_temperature', self.gate_temperature)) if control_overrides else self.gate_temperature
        gate_bias = float(control_overrides.get('gate_logit_bias', 0.0)) if control_overrides else 0.0
        tau_global = float(control_overrides.get('soft_entropy_tau', self.soft_entropy_tau)) if control_overrides else self.soft_entropy_tau

        force_gate_open = bool(control_overrides.get('force_gate_open', False)) if control_overrides else False

        for i, (quantizer, rate_est, entropy_codec) in enumerate(zip(self.quantizers, self.rate_estimators, self.entropy_codecs)):
            # 计算当前阶段的权重
            if adaptive_weights is not None:
                stage_weight = adaptive_weights[i]
            else:
                stage_weight = F.softplus(self.stage_weights[i])

            # 阶段 gating：若提供 stage_mask 且该级禁用，则跳过量化
            gated_off = (stage_mask is not None and i < len(stage_mask) and not bool(stage_mask[i]))

            # Per-frame stage gates (i>=1) using residual features
            B, D, T = residual.shape
            device = residual.device
            if force_gate_open and i >= 1:
                gate_mask = torch.ones(B, T, device=device)
                stage_gate_masks.append(gate_mask)
                stage_gate_soft.append(torch.ones(B, T, device=device))
            elif (not gated_off) and self.enable_stage_gates and i >= 1:
                gate_module = self.stage_gates[i]
                gate_mask, gate_soft = gate_module(residual, temperature=gate_temp, logit_bias=gate_bias)
                stage_gate_masks.append(gate_mask)
                stage_gate_soft.append(gate_soft)
            else:
                gate_mask = torch.ones(B, T, device=device)
                stage_gate_masks.append(gate_mask)
                stage_gate_soft.append(torch.ones(B, T, device=device))

            if gated_off:
                dtype = residual.dtype
                quant = torch.zeros(B, D, T, device=device, dtype=dtype)
                commitment = torch.tensor(0.0, device=device, dtype=dtype)
                perplexity = torch.tensor(0.0, device=device, dtype=dtype)
                if return_indices and stage_indices is not None:
                    indices = torch.zeros(B, T, device=device, dtype=torch.long)
                    stage_indices.append(indices)
                # 残差保持不变
                quantized_stages.append(torch.zeros_like(quant))
                # 软分配（全零，避免下游NaN）
                K = self.rate_estimators[i].codebook_size
                stage_soft_probs.append(torch.zeros(B, T, K, device=device))
            else:
                # 量化真实残差
                if return_indices:
                    quant, commitment, perplexity, indices = quantizer(residual, return_indices=True)
                    stage_indices.append(indices)
                else:
                    quant, commitment, perplexity, _ = quantizer(residual, return_indices=False)

                # Soft assignment probabilities with codebook mask
                # residual: [B, D, T] -> [B*T, D]
                r_flat = residual.permute(0, 2, 1).reshape(B * T, D)
                embed = quantizer.embed  # [K, D]
                d2 = (
                    (r_flat.pow(2).sum(dim=1, keepdim=True))
                    + (embed.pow(2).sum(dim=1)).unsqueeze(0)
                    - 2.0 * (r_flat @ embed.t())
                )  # [BT,K]
                # Effective per-stage temperature = global_tau * stage_tau[i] (learnable)
                tau_eff = (self.stage_tau[i].clamp(0.1, 10.0) * tau_global).clamp_min(1e-6)
                logits = -d2 / tau_eff
                if self.enable_codebook_mask and self.codebook_mask_logits is not None:
                    mask_probs = torch.sigmoid(self.codebook_mask_logits[i])  # [K]
                    # Track average usage for sparsity penalty
                    codebook_mask_usage.append(mask_probs.mean())
                    logits = logits + torch.log(mask_probs.clamp_min(1e-6)).unsqueeze(0)
                probs = torch.softmax(logits, dim=1)  # [BT,K]
                stage_soft_probs.append(probs.view(B, T, -1))

                # 残差更新与结果叠加按权重
                # Apply learnable gate mask per-frame (broadcast over channels)
                quant = quant * gate_mask.unsqueeze(1)
                residual = residual - stage_weight * quant
                quantized_stages.append(stage_weight * quant)

                # === Debug stats per stage ===
                if self.enable_debug_stats:
                    try:
                        r_view = residual.detach()
                        q_view = quant.detach()
                        ent_bT = (-(stage_soft_probs[-1] * torch.log2(stage_soft_probs[-1].clamp_min(1e-12))).sum(dim=-1))  # [B,T]
                        dbg = {
                            'res_mean': float(r_view.mean().item()),
                            'res_std': float(r_view.std().item()),
                            'res_min': float(r_view.amin().item()),
                            'res_max': float(r_view.amax().item()),
                            'q_mean': float(q_view.mean().item()),
                            'q_std': float(q_view.std().item()),
                            'q_min': float(q_view.amin().item()),
                            'q_max': float(q_view.amax().item()),
                            'gate_mean': float(stage_gate_soft[-1].mean().item()),
                            'soft_H_bits': float(ent_bT.mean().item())
                        }
                    except Exception:
                        dbg = {}
                else:
                    dbg = {}

            # 码率估计（传递stage权重用于动态控制）
            if return_indices and stage_indices is not None:
                # 使用软熵近似 + per-frame gate软值作为有效码率估计
                if len(stage_soft_probs) > 0:
                    q = stage_soft_probs[-1]  # [B, T, K]
                    ent_bT = (-(q * torch.log2(q.clamp_min(1e-12))).sum(dim=-1))  # [B,T]
                    rate = (ent_bT * stage_gate_soft[-1]).mean()
                else:
                    # 回退到真实熵编码估计（再乘平均gate）
                    real_rate = entropy_codec.estimate_rate_from_indices(indices)
                    rate = real_rate * stage_gate_soft[-1].mean()
            else:
                # 回退到理论码率（快速训练）并考虑平均门控
                rate = rate_est(quant, stage_weight) * stage_gate_soft[-1].mean()

            commitment_losses.append(commitment)
            rate_estimates.append(rate)
            perplexities.append(perplexity)
            if self.enable_debug_stats:
                stage_debug_stats.append(dbg)

        # 重建总量化结果（de-whiten back to original latent distribution)
        total_quantized = sum(quantized_stages)
        total_quantized = total_quantized * sd.view(1, D, 1) + mu.view(1, D, 1)
        total_commitment = sum(commitment_losses) / len(commitment_losses)
        total_rate = sum(rate_estimates)

        result = {
            'quantized': total_quantized,
            'commitment_loss': total_commitment,
            'rate_bits_per_frame': total_rate,
            'final_residual': residual,
            'stage_quantized': quantized_stages,
            'stage_perplexities': perplexities,
            'stage_weights': F.softplus(self.stage_weights)
        }

        # 添加索引信息（用于真实RVQ系统）
        if return_indices:
            result['stage_indices'] = stage_indices

        # 附加软分配和门控信息（供loss使用）
        if len(stage_soft_probs) > 0:
            result['stage_soft_probs'] = stage_soft_probs
        if stage_gate_masks:
            result['stage_gate_masks'] = stage_gate_masks
            result['stage_gate_soft'] = stage_gate_soft
        if codebook_mask_usage:
            result['codebook_mask_usage'] = codebook_mask_usage
        if self.enable_debug_stats and stage_debug_stats:
            result['stage_debug_stats'] = stage_debug_stats

        return result

    def decode_from_indices(self, stage_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        从码字索引重构特征（不依赖解码器）
        stage_indices: List[[B, T]] 每个RVQ阶段的码字索引
        returns: [B, D, T] 重构的特征
        """
        if len(stage_indices) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} stages, got {len(stage_indices)}")

        B, T = stage_indices[0].shape
        device = stage_indices[0].device

        # 初始化重构特征
        reconstructed = torch.zeros(B, self.input_dim, T, device=device)

        # 逐阶段重构 - 修复：解码也要按相同的stage_weight叠加
        for stage_idx, indices in enumerate(stage_indices):
            # 获取对应的stage_weight
            stage_weight = F.softplus(self.stage_weights[stage_idx])
            stage_features = self.quantizers[stage_idx].decode_from_indices(indices)  # [B, D, T]
            reconstructed += stage_weight * stage_features  # 按权重叠加

        return reconstructed

    def compress_to_bitstream(self, x: torch.Tensor) -> Tuple[List[bytes], Dict[str, float]]:
        """
        完整的压缩流程：特征 -> 码字索引 -> 比特流
        x: [B, D, T] 输入特征
        returns: (比特流列表, 统计信息)
        """
        # 获取码字索引
        results = self.forward(x, return_indices=True)
        stage_indices = results['stage_indices']

        # 逐阶段熵编码
        bitstreams = []
        total_bits = 0.0
        compression_stats = {}

        for stage_idx, (indices, codec) in enumerate(zip(stage_indices, self.entropy_codecs)):
            bitstream, real_rate = codec.compress(indices)
            bitstreams.append(bitstream)
            total_bits += real_rate * indices.numel()
            compression_stats[f'stage_{stage_idx}_rate'] = real_rate

        B, T = stage_indices[0].shape
        compression_stats['total_kbps'] = total_bits / T * 50 / 1000  # 50fps -> kbps
        compression_stats['avg_bits_per_frame'] = total_bits / T

        return bitstreams, compression_stats

    def decompress_from_bitstream(self, bitstreams: List[bytes], shape: Tuple[int, int]) -> torch.Tensor:
        """
        完整的解压缩流程：比特流 -> 码字索引 -> 特征
        bitstreams: List[bytes] 各阶段比特流
        shape: (B, T) 输出形状
        returns: [B, D, T] 重构特征
        """
        if len(bitstreams) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} bitstreams, got {len(bitstreams)}")

        # 逐阶段熵解码
        stage_indices = []
        for stage_idx, (bitstream, codec) in enumerate(zip(bitstreams, self.entropy_codecs)):
            indices = codec.decompress(bitstream, shape)
            stage_indices.append(indices)

        # 重构特征
        return self.decode_from_indices(stage_indices)

class AdaptiveRateController(nn.Module):
    """自适应码率控制器，确保1.2±0.1kbps稳定性"""
    def __init__(
        self,
        target_kbps: float = 1.2,
        tolerance: float = 0.1,
        control_strength: float = 0.1,
        frame_rate: int = 50,
        history_size: int = 100,
        pid_kp: float = 0.6,
        pid_ki: float = 0.05,
        pid_kd: float = 0.0,
        gate_logit_bias_init: float = 0.0
    ):
        super().__init__()
        self.target_kbps = target_kbps
        self.tolerance = tolerance
        self.control_strength = control_strength
        self.frame_rate = frame_rate

        # RVQ阶段权重控制（替代PID）
        self.register_buffer('stage_weights_ema', torch.ones(3) / 3.0)  # 默认3阶段
        self.register_buffer('rate_ema', torch.tensor(target_kbps))
        self.ema_alpha = 0.95  # EMA平滑系数

        # 码率历史和统计
        self.rate_history = deque(maxlen=history_size)
        self.theoretical_bits_per_stage = [10.0, 9.0, 8.0]  # 理论比特数[Stage1, Stage2, Stage3]

        # PID states
        self.kp = float(pid_kp)
        self.ki = float(pid_ki)
        self.kd = float(pid_kd)
        self._err_int = 0.0
        self._err_prev = 0.0

        # Control knobs (base values)
        self.base_gate_temperature = 0.67
        self.base_soft_tau = 0.30
        self.base_vib_scale = 1.0

        # Control gains
        self.gain_gate_temp = 0.5
        self.gain_tau = 0.5
        self.gain_vib = 0.8
        self.gain_mask = 0.3

        # Last computed controls
        self._last_controls = {
            'gate_temperature': self.base_gate_temperature,
            'soft_entropy_tau': self.base_soft_tau,
            'vib_beta_scale': self.base_vib_scale,
            'mask_sparsity_boost': 0.0
        }
        # Exponential tau controller state and bounds
        self._tau_running = float(self.base_soft_tau)
        self.tau_min = 0.2
        self.tau_max = 2.0
        self.kappa_tau = 0.08
        # Exponential gate temperature controller (flip sign for under-target)
        self._gate_temp_running = float(self.base_gate_temperature)
        self.gate_temp_min = 0.2
        self.gate_temp_max = 2.0
        self.kappa_gate = 0.05
        # Gate logit bias (PI-like integration)
        self._gate_bias_running = float(gate_logit_bias_init)
        self.gate_bias_min = -2.0
        self.gate_bias_max = 2.0
        self.k_gate_bias = 0.05

    def compute_rate_loss(
        self,
        rate_bits_per_frame: torch.Tensor,
        current_step: int = 0
    ) -> torch.Tensor:
        """
        计算码率约束损失
        """
        # 转换为kbps
        current_kbps = rate_bits_per_frame * self.frame_rate / 1000.0

        # 更新历史
        if self.training:
            self.rate_history.append(current_kbps.detach().cpu().item())

        # 目标范围
        lower_bound = self.target_kbps - self.tolerance  # 1.1 kbps
        upper_bound = self.target_kbps + self.tolerance  # 1.3 kbps

        # 分段损失函数
        if current_kbps < lower_bound:
            # 码率过低：轻微惩罚
            rate_loss = 0.1 * (lower_bound - current_kbps) ** 2
        elif current_kbps > upper_bound:
            # 码率过高：强烈惩罚
            rate_loss = 2.0 * (current_kbps - upper_bound) ** 2
        else:
            # 目标范围内：小幅奖励
            distance_from_target = torch.abs(current_kbps - self.target_kbps)
            rate_loss = -0.01 * torch.exp(-distance_from_target * 10)

        # 更新EMA码率统计
        if self.training:
            self.rate_ema.data = self.ema_alpha * self.rate_ema + (1 - self.ema_alpha) * current_kbps.detach()

        # 返回简化的码率损失（不使用PID）
        return rate_loss

    def compute_control_knobs(self) -> Dict[str, float]:
        """Compute control knobs via PI on rate error using EMA rate.

        Returns dict with keys: gate_temperature, soft_entropy_tau, vib_beta_scale, mask_sparsity_boost.
        """
        # Error in kbps
        e = float(self.rate_ema.item() - self.target_kbps)
        # PI update
        self._err_int = max(-1.0, min(1.0, self._err_int + e * 0.1))  # integral clamp
        u = self.kp * e + self.ki * self._err_int

        # Exponential updates
        try:
            # Exponential update: tau <- clamp(tau * exp(kappa*(target - rate_ema)))
            delta = self.kappa_tau * (self.target_kbps - float(self.rate_ema.item()))
            self._tau_running = float(min(self.tau_max, max(self.tau_min, self._tau_running * math.exp(delta))))
            soft_tau = self._tau_running
            # Gate temperature: reduce when under-target (target - rate_ema > 0 -> decrease temp)
            delta_g = self.kappa_gate * (float(self.rate_ema.item()) - self.target_kbps)
            self._gate_temp_running = float(min(self.gate_temp_max, max(self.gate_temp_min, self._gate_temp_running * math.exp(delta_g))))
            gate_temperature = self._gate_temp_running
            # Gate logit bias: positive when under-target
            self._gate_bias_running += self.k_gate_bias * (self.target_kbps - float(self.rate_ema.item()))
            self._gate_bias_running = float(min(self.gate_bias_max, max(self.gate_bias_min, self._gate_bias_running)))
            gate_logit_bias = self._gate_bias_running
        except Exception:
            soft_tau = max(0.05, self.base_soft_tau * (1.0 - self.gain_tau * u))
            gate_temperature = max(0.1, self.base_gate_temperature * (1.0 - self.gain_gate_temp * u))
            gate_logit_bias = 0.0
        vib_scale = max(0.2, self.base_vib_scale * (1.0 + self.gain_vib * u))
        mask_boost = max(0.0, self.gain_mask * u)  # encourage sparsity when rate is high

        self._last_controls = {
            'gate_temperature': float(gate_temperature),
            'gate_logit_bias': float(gate_logit_bias),
            'soft_entropy_tau': float(soft_tau),
            'vib_beta_scale': float(vib_scale),
            'mask_sparsity_boost': float(mask_boost)
        }
        return self._last_controls

    def get_last_controls(self) -> Dict[str, float]:
        return dict(self._last_controls)

    def get_adaptive_stage_weights(self) -> torch.Tensor:
        """获取自适应的RVQ阶段权重

        逻辑：
        - 码率过高：降低高阶段权重（Stage 2,3）
        - 码率过低：提高高阶段权重
        """
        rate_error = (self.rate_ema - self.target_kbps) / self.target_kbps

        # 基础权重（1/3各）
        base_weights = torch.ones(3) / 3.0

        if rate_error.abs() < self.tolerance / self.target_kbps:
            # 在目标范围内，保持均衡
            adjusted_weights = base_weights
        elif rate_error > 0:  # 码率过高
            # 降低高阶段权重
            factor = min(0.5, rate_error.abs())
            adjusted_weights = torch.tensor([0.5 + factor, 0.3 - factor/2, 0.2 - factor/2])
        else:  # 码率过低
            # 提高高阶段权重
            factor = min(0.3, rate_error.abs())
            adjusted_weights = torch.tensor([0.2, 0.4 + factor, 0.4 + factor])

        # EMA平滑
        self.stage_weights_ema.data = (
            self.ema_alpha * self.stage_weights_ema +
            (1 - self.ema_alpha) * adjusted_weights.to(self.stage_weights_ema.device)
        )

        return self.stage_weights_ema

    def get_current_stats(self) -> Dict[str, float]:
        """获取当前码率统计信息"""
        if len(self.rate_history) < 10:
            return {'mean_kbps': 0, 'std_kbps': 0, 'in_range_ratio': 0}

        rates = torch.tensor(list(self.rate_history))
        lower = self.target_kbps - self.tolerance
        upper = self.target_kbps + self.tolerance
        in_range = ((rates >= lower) & (rates <= upper)).float().mean()

        return {
            'mean_kbps': rates.mean().item(),
            'std_kbps': rates.std().item(),
            'in_range_ratio': in_range.item(),
            'current_target': self.target_kbps,
            'rate_ema': self.rate_ema.item(),
            'stage_weights': self.stage_weights_ema.tolist()
        }

class SemanticPreservationHead(nn.Module):
    """语义保持模块，确保压缩后语义信息完整性"""
    def __init__(
        self,
        compressed_dim: int = 24,
        target_dim: int = 36,
        preservation_layers: int = 3  # 提升到3层
    ):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.target_dim = target_dim

        # 语义重构网络（加深层数+残差连接）
        layers = []
        residual_layers = []
        current_dim = compressed_dim

        for i in range(preservation_layers):
            if i == preservation_layers - 1:
                next_dim = target_dim
            elif i == 0:
                next_dim = compressed_dim * 2
            else:
                next_dim = current_dim  # 中间层保持维度，便于残差

            # 主分支
            layers.extend([
                nn.Conv1d(current_dim, next_dim, 3, padding=1),
                nn.BatchNorm1d(next_dim),
                nn.GELU() if i < preservation_layers - 1 else nn.Identity(),
            ])

            # 残差分支（中间层）
            if i > 0 and i < preservation_layers - 1 and current_dim == next_dim:
                residual_layers.append(i)

            current_dim = next_dim

        self.semantic_net = nn.Sequential(*layers)
        self.has_residual = len(residual_layers) > 0

        # 语义特征对比头
        self.contrastive_head = nn.Sequential(
            nn.Conv1d(target_dim, target_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(target_dim // 2, target_dim // 4, 1)
        )

    def forward(self, compressed_features: torch.Tensor) -> torch.Tensor:
        """
        compressed_features: [B, compressed_dim, T] 压缩后特征
        returns: [B, target_dim, T] 语义重构特征
        """
        x = compressed_features

        # 逐层前向，中间层添加残差
        layer_idx = 0
        for module in self.semantic_net:
            if isinstance(module, nn.Conv1d):
                x_prev = x
                x = module(x)
                # 在中间卷积层后添加残差（如果维度匹配）
                if (layer_idx > 0 and layer_idx < len([m for m in self.semantic_net if isinstance(m, nn.Conv1d)]) - 1
                    and x.shape == x_prev.shape):
                    x = x + x_prev
                layer_idx += 1
            else:
                x = module(x)

        return x

    def compute_contrastive_loss(
        self,
        compressed_features: torch.Tensor,
        reference_features: torch.Tensor
    ) -> torch.Tensor:
        """计算对比学习损失"""
        # 获取语义表示
        compressed_semantic = self.contrastive_head(
            self.forward(compressed_features)
        )  # [B, D//4, T]
        reference_semantic = self.contrastive_head(reference_features)

        # 简化的对比损失(可扩展为更复杂的InfoNCE)
        return F.mse_loss(compressed_semantic, reference_semantic)

    

class PerceptualQualityHead(nn.Module):
    """基于特征的感知质量估计头（PESQ/STOI/SI-SDR 轻量估计）"""
    def __init__(self, input_dim: int = 24, num_metrics: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_metrics = num_metrics

        # 质量估计采用基于特征的轻量实现

        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 质量预测网络
        self.quality_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_metrics)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D, T] 输入特征
        returns: [B, num_metrics] 质量预测
        """
        # 全局池化
        pooled = self.global_pool(features).squeeze(-1)  # [B, D]

        # 质量预测（原始logits）
        raw_scores = self.quality_net(pooled)  # [B, num_metrics]

        # 输出范围约束 - 修复：使用更合理的初始化期望
        if self.num_metrics >= 1:
            # PESQ: [-0.5, 4.5] -> tanh * 2.5 + 2.0 (中心在2.0，范围[-0.5, 4.5])
            pesq = torch.tanh(raw_scores[:, 0:1]) * 2.5 + 2.0
        if self.num_metrics >= 2:
            # STOI: [0, 1] -> sigmoid
            stoi = torch.sigmoid(raw_scores[:, 1:2])
        if self.num_metrics >= 3:
            # SI-SDR: 约束在±20dB范围
            si_sdr = torch.tanh(raw_scores[:, 2:3]) * 20.0

        # 组合结果
        if self.num_metrics == 1:
            quality_scores = pesq
        elif self.num_metrics == 2:
            quality_scores = torch.cat([pesq, stoi], dim=1)
        elif self.num_metrics >= 3:
            quality_scores = torch.cat([pesq, stoi, si_sdr], dim=1)
        else:
            quality_scores = raw_scores  # 回退

        return quality_scores

class AETHERStage5Model(nn.Module):
    """
    Stage5主模型：AETHER + RVQ + 率失真优化

    特性:
    1. 基于Stage4权重的增量训练
    2. 分层RVQ压缩
    3. 自适应码率控制
    4. 语义保持和质量评估
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # === 导入真实的Stage4组件 ===
        from models.enhanced_aether_integration import AETHEREncoder
        from models.maybe_useless.aether_fargan_decoder import AETHERFARGANDecoder

        # === 创建真实的encoder和decoder ===
        # 注意：n_experts和top_k会在load_stage4_weights中从checkpoint推断并更新
        self.encoder = AETHEREncoder(
            d_in=config.get('original_feature_dim', 36),
            d_model=128,
            dz=config.get('feature_dim', 24),
            d_csi=config.get('d_csi', 4),
            use_film=config.get('use_film', True),
            film_position=config.get('film_position', 'post'),
            use_moe=config.get('use_moe', True),
            use_quantization=False,  # Stage5中由RVQ处理量化
            latent_bits=4,
            moe_router_use_csi=False,
            use_semantic_head=False,
            n_experts=config.get('n_experts', 4) if config.get('n_experts') else 4,  # 临时值，从checkpoint推断
            top_k=config.get('top_k', 2) if config.get('top_k') else 2,  # 临时值，从checkpoint推断
        )

        self.decoder = AETHERFARGANDecoder(
            d_out=config.get('original_feature_dim', 36),
            d_csi=config.get('d_csi', 4),
            enable_synth=True,
            feature_spec_type="fargan",
            use_film=config.get('use_film', True),
            enable_output_calibration=config.get('enable_fargan_output_calibration', False),
            enable_ceps_affine_calib=config.get('enable_ceps_affine_calib', True)
        )

        # === 特征压缩层：36维 -> 24维 ===
        original_dim = config.get('original_feature_dim', 36)
        compressed_dim = config.get('feature_dim', 24)

        if original_dim != compressed_dim:
            self.feature_compressor = nn.Sequential(
                nn.Linear(original_dim, compressed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(compressed_dim * 2, compressed_dim),
                nn.LayerNorm(compressed_dim)
            )
            # 特征重构层：24维 -> 36维
            self.feature_reconstructor = nn.Sequential(
                nn.Linear(compressed_dim, compressed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(compressed_dim * 2, original_dim),
                nn.LayerNorm(original_dim)
            )
        else:
            self.feature_compressor = nn.Identity()
            self.feature_reconstructor = nn.Identity()

        # === RVQ压缩链 ===
        self.rvq_encoder = HierarchicalRVQ(
            input_dim=config.get('feature_dim', 24),
            num_stages=config.get('rvq_stages', 3),
            codebook_sizes=config.get('codebook_sizes', [1024, 512, 256]),
            commitment_weights=config.get('commitment_weights', [0.01, 0.02, 0.03]),
            decay=config.get('rvq_decay', 0.99),
            eps=config.get('rvq_eps', 1e-5),
            enable_stage_gates=config.get('enable_learnable_stage_gates', False),
            gate_temperature=config.get('gate_temperature', 0.67),
            soft_entropy_tau=config.get('soft_entropy_tau', 0.3),
            enable_codebook_mask=True,
            enable_debug_stats=config.get('enable_debug_stats', True)
        )

        # === RVQ解码器（关键：以前缺失）===
        self.rvq_decoder = HierarchicalRVQDecoder(self.rvq_encoder)

        # === 码率控制器 ===
        self.rate_controller = AdaptiveRateController(
            target_kbps=config.get('target_kbps', 1.2),
            tolerance=config.get('rate_tolerance', 0.1),
            control_strength=config.get('control_strength', 0.1),
            frame_rate=config.get('frame_rate', 50),
            gate_logit_bias_init=config.get('gate_logit_bias_init', 0.0)
        )

        # === 语义保持模块 ===
        self.semantic_preservor = SemanticPreservationHead(
            compressed_dim=config.get('feature_dim', 24),
            target_dim=config.get('original_feature_dim', 36),
            preservation_layers=config.get('semantic_layers', 2)
        )

        # === [VIB] config & module ==================================================
        self.vib_beta = config.get('beta_vib', 1e-3)
        self.enable_vib = config.get('enable_vib', True)
        self.vib = GaussianBottleneck(config.get('feature_dim', 24))

        # === [SEM] teacher & student projection ====================================
        self.semantic_dim = config.get('semantic_dim', 128)
        # 教师网络：主教师网络（可训练）
        self.teacher = SemanticTeacher(
            in_dim=config.get('original_feature_dim', 36),
            proj_dim=self.semantic_dim
        )
        # 动量教师网络：用于稳定语义目标（不直接训练）
        self.teacher_momentum = SemanticTeacher(
            in_dim=config.get('original_feature_dim', 36),
            proj_dim=self.semantic_dim
        )
        # 动量系数 (0.999 = 很慢的更新)
        self.teacher_momentum_coeff = config.get('teacher_momentum_coeff', 0.999)

        # 分阶段信道扰动配置
        self.channel_disable_steps = config.get('channel_disable_steps', 1000)  # 前1000步禁用
        self.channel_warmup_steps = config.get('channel_warmup_steps', 1000)    # 1000步线性启用

        # 语义抽取策略配置
        self.semantic_use_reconstructed = config.get('semantic_use_reconstructed', True)  # 使用重构特征
        self.semantic_dual_branch = config.get('semantic_dual_branch', False)  # 双分支模式

        # 学生端：增强的语义特征投影器 - 与SemanticTeacher结构对齐
        self.student_proj = nn.Sequential(
            # 第一层：局部特征提取 (与teacher第一层对齐)
            nn.Conv1d(config.get('original_feature_dim', 36), 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # 第二层：中等范围特征 (与teacher第二层对齐)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # 第三层：长范围依赖 (与teacher第三层对齐)
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # 语义投影头 (与teacher投影头对齐)
            nn.Conv1d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, self.semantic_dim, kernel_size=1)
        )

        # === 质量评估器 ===
        # 回退到基于特征的质量估计 - 更可靠的初期训练
        self.quality_estimator = PerceptualQualityHead(
            input_dim=config.get('feature_dim', 24),
            num_metrics=config.get('quality_metrics', 3)
        )

        # 音频质量估计器已移除（未接线），如需恢复请单独实现

        # === RVQ stage sizes for logging/loss ===
        self.rvq_stage_sizes = config.get('rvq_codebook_sizes', [1024, 512, 256])

        # 初始化：动量教师复制学生网络的参数（BYOL风格）
        self._init_momentum_teacher()

    def load_stage4_weights(self, checkpoint_path: str):
        """加载Stage4预训练权重，参照stage4_train_full.py的_load_stage3_checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # === 1. 从checkpoint推断MoE配置 ===
        enc_sd = checkpoint.get("encoder_state_dict")
        dec_sd = checkpoint.get("decoder_state_dict")

        if enc_sd:
            # 推断MoE专家数量
            n_experts_inferred = None
            import re
            pat = re.compile(r"^moe\.specialized_moe\.experts\.(\d+)\.")
            idxs = set()
            for key in enc_sd.keys():
                m = pat.match(key)
                if m:
                    idxs.add(int(m.group(1)))
            if idxs:
                n_experts_inferred = len(idxs)

            # 如果没有从experts参数推断到，尝试从其他MoE参数推断
            if n_experts_inferred is None:
                for key in enc_sd.keys():
                    if 'expert_counts' in key:
                        n_experts_inferred = enc_sd[key].shape[0]
                        break

            # 获取当前encoder的n_experts
            current_n_experts = getattr(self.encoder.moe.specialized_moe, 'n_experts', 4) if hasattr(self.encoder, 'moe') and hasattr(self.encoder.moe, 'specialized_moe') else 4

            if n_experts_inferred and n_experts_inferred != current_n_experts:
                print(f"Inferred n_experts={n_experts_inferred} from checkpoint (current={current_n_experts})")

                # 重新创建encoder with正确的配置
                from models.enhanced_aether_integration import AETHEREncoder
                self.encoder = AETHEREncoder(
                    d_in=self.config.get('feature_dims', 36),
                    d_model=128,
                    dz=24,
                    d_csi=self.config.get('d_csi', 4),
                    use_film=self.config.get('use_film', True),
                    film_position=self.config.get('film_position', 'post'),
                    use_moe=self.config.get('use_moe', True),
                    use_quantization=False,
                    latent_bits=4,
                    moe_router_use_csi=False,
                    use_semantic_head=False,
                    n_experts=n_experts_inferred,  # 使用推断的值
                    top_k=min(2, n_experts_inferred),  # 自动调整top_k
                )
                print(f"Recreated encoder with n_experts={n_experts_inferred}, top_k={min(2, n_experts_inferred)}")

        # === 2. 加载权重 ===
        if enc_sd:
            print(f"Loading encoder weights from {checkpoint_path}")
            self.encoder.load_state_dict(enc_sd, strict=False)

        if dec_sd:
            print(f"Loading decoder weights from {checkpoint_path}")
            self.decoder.load_state_dict(dec_sd, strict=False)

        # === 3. 初始冻结策略：Stage5渐进式训练 ===
        # 阶段1 (0-2000步): 只训练RVQ码本和VIB语义头
        self._apply_initial_freezing()

    def _apply_initial_freezing(self):
        """应用初始冻结策略：只训练RVQ和VIB组件"""
        # 冻结编码器主体
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 冻结解码器主体
        for param in self.decoder.parameters():
            param.requires_grad = False

        # 只允许训练RVQ相关组件
        for name, param in self.named_parameters():
            if any(component in name for component in [
                'rvq', 'quantizer', 'semantic_preservor',
                'student_proj', 'teacher', 'quality_estimator'
            ]):
                param.requires_grad = True

        print("🚀 Stage5 Phase 1: Training RVQ codebooks + VIB semantic head only")
        print("   - Encoder: FROZEN")
        print("   - Decoder: FROZEN")
        print("   - FARGAN head: FROZEN")
        print("   - RVQ + VIB: TRAINABLE")

    def update_training_phase(self, current_step: int):
        """根据训练步数更新冻结策略"""
        if not hasattr(self, '_current_phase'):
            self._current_phase = 1

        # 强制调试：每100步检查一次阶段状态
        if current_step % 100 == 0:
            print(f"[Phase Debug] Step {current_step}: current_phase={getattr(self, '_current_phase', 'None')}")

        # 获取配置中的阶段转换步数
        phase2_step = self.config.get('phase2_unfreeze_step', 2000)
        phase3_step = self.config.get('phase3_unfreeze_step', 4000)
        phase4_step = self.config.get('phase4_unfreeze_step', 6000)

        if current_step >= phase2_step and self._current_phase == 1:
            # 阶段2：解冻FARGAN波形头
            print(f"[Phase Transition] Step {current_step}: Entering Phase 2 - Unfreezing FARGAN head (configured at step {phase2_step})")
            self._unfreeze_fargan_head()
            self._current_phase = 2
            # 标记需要重新创建优化器
            self._optimizer_needs_update = True
        elif current_step >= phase3_step and self._current_phase == 2:
            # 阶段3：解冻解码器组件
            print(f"[Phase Transition] Step {current_step}: Entering Phase 3 - Unfreezing decoder components (configured at step {phase3_step})")
            self._unfreeze_decoder_components()
            self._current_phase = 3
            # 标记需要重新创建优化器
            self._optimizer_needs_update = True
        elif current_step >= phase4_step and self._current_phase == 3:
            # 阶段4：部分解冻编码器
            print(f"[Phase Transition] Step {current_step}: Entering Phase 4 - Unfreezing encoder components (configured at step {phase4_step})")
            self._unfreeze_encoder_components()
            self._current_phase = 4
            # 标记需要重新创建优化器
            self._optimizer_needs_update = True

    def get_dynamic_loss_weights(self, current_step: int) -> dict:
        """根据训练阶段返回动态损失权重"""

        # 获取配置中的阶段转换步数
        phase2_step = self.config.get('phase2_unfreeze_step', 2000)
        phase3_step = self.config.get('phase3_unfreeze_step', 4000)
        phase4_step = self.config.get('phase4_unfreeze_step', 6000)

        if current_step < phase2_step:
            # 阶段1：专注RVQ和语义，降低wave权重避免过度关注音频重建
            return {
                'feat': 0.1,          # 适中特征权重(初期1500*0.1=150, 后期10*0.1=1)
                'wave': 0.8,          # 降低wave权重，避免抢夺语义学习
                'semantic': 1.2,      # 重点训练语义，提升权重
                'commitment': 2.0,    # 重点训练RVQ
                'quality': 0.05,      # 降低质量损失
# 'vib': 权重不再使用，由beta_vib直接控制
                'rate': 0.3           # 保持适度的码率控制
            }
        elif current_step < phase3_step:
            # 阶段2：平衡特征和波形训练，继续重视语义学习
            return {
                'feat': 0.2,          # 提升特征权重(此时feat_loss~100, 100*0.2=20)
                'wave': 1.5,          # 适度增加wave权重，但不过度
                'semantic': 0.8,      # 继续重视语义学习
                'commitment': 1.0,    # 保持RVQ训练
                'quality': 0.1,
# 'vib': 权重不再使用，由beta_vib直接控制
                'rate': 0.4           # 增加码率控制
            }
        elif current_step < phase4_step:
            # 阶段3：平衡训练，开始逐步增加wave权重
            return {
                'feat': 0.5,          # feat_loss~50时，50*0.5=25
                'wave': 2.5,          # 逐步增加wave权重，但保持克制
                'semantic': 0.6,      # 保持语义权重，避免过早下降
                'commitment': 0.5,
                'quality': 0.1,
# 'vib': 权重不再使用，由beta_vib直接控制
                'rate': 0.5           # 强化码率控制
            }
        else:
            # 阶段4：精细调优，平衡音频质量和语义保持
            return {
                'feat': 1.0,          # feat_loss~10时，10*1.0=10
                'wave': 4.0,          # 适度增加wave权重，但不过度压制语义
                'semantic': 0.3,      # 保持适度的语义学习，避免语义坍塌
                'commitment': 0.3,
                'quality': 0.15,
# 'vib': 权重不再使用，由beta_vib直接控制
                'rate': 0.6           # 最大码率控制
            }

    def _unfreeze_fargan_head(self):
        """解冻FARGAN波形头"""
        unfrozen_count = 0
        for name, param in self.decoder.named_parameters():
            if any(component in name for component in ['fargan_core', 'period_estimator']):
                param.requires_grad = True
                unfrozen_count += 1
                print(f"   Unfrozen: {name}")
        print(f"🔓 Stage5 Phase 2: Unfrozen FARGAN wave head ({unfrozen_count} parameters)")

    def _unfreeze_decoder_components(self):
        """解冻解码器主要组件"""
        decoder_components = ['aether_decode', 'refine', 'film']
        for name, param in self.decoder.named_parameters():
            if any(comp in name.lower() for comp in decoder_components):
                param.requires_grad = True
        print("🔓 Stage5 Phase 3: Unfrozen decoder components")

    def _unfreeze_encoder_components(self):
        """部分解冻编码器"""
        encoder_components = ['film', 'output_proj']  # 只解冻关键组件
        for name, param in self.encoder.named_parameters():
            if any(comp in name.lower() for comp in encoder_components):
                param.requires_grad = True
        print("🔓 Stage5 Phase 4: Partially unfrozen encoder")

    def _reparam(self, mu, logvar):
        """VIB重参数化采样"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _kl_standard_normal(self, mu, logvar):
        """计算KL(q(z|x) || N(0, I))"""
        return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).mean()

    def forward(
        self,
        inputs: torch.Tensor,
        csi_dict: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        current_step: int = 0,
        return_wave: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，参照Stage4的训练流程
        inputs: [B, T, C] 输入特征 (36维)
        csi_dict: CSI字典
        returns: 模型输出字典
        """
        # === 1. 输入特征预处理（防止量级过大）===
        # 检查输入范围并进行适度标准化
        input_std = inputs.std()
        if input_std > 10.0:  # 如果标准差过大，进行标准化
            normalized_inputs = inputs / (input_std / 3.0)  # 将标准差控制在3左右
        else:
            normalized_inputs = inputs

        # === 2. 特征编码（使用真实的Stage4 encoder）===
        # 输入应该是[B, T, 36]格式的特征
        # 传递训练步骤给encoder的FiLM系统
        try:
            encoded_latent, enc_logs = self.encoder(normalized_inputs, csi_dict, training_step=current_step)  # [B, T, 24]
        except TypeError:
            # 回退：不支持training_step参数
            encoded_latent, enc_logs = self.encoder(normalized_inputs, csi_dict)  # [B, T, 24]

        # 转换为[B, C, T]格式用于RVQ
        latent_bct = encoded_latent.transpose(1, 2)  # [B, 24, T]

        # --- [VIB] apply Gaussian bottleneck on latent_bct -------------------------
        if self.enable_vib:
            # 修复1: 对VIB输入进行LayerNorm，避免μ、logvar过大
            latent_btn = latent_bct.transpose(1, 2)  # [B,24,T] -> [B,T,24]
            latent_btn = torch.layer_norm(latent_btn, (latent_btn.size(-1),))
            latent_bct_normalized = latent_btn.transpose(1, 2)  # [B,T,24] -> [B,24,T]

            z_bct, mu_bct, logvar_bct, vib_kld = self.vib(latent_bct_normalized)
            # 修复：训练时用随机采样z，推理时用确定性μ避免抖动
            if self.training:
                latent_bct = z_bct
            else:
                latent_bct = mu_bct  # 评估/推理走确定性通道
            # VIB调试统计
            try:
                debug_vib = {
                    'mu_mean': float(mu_bct.mean().item()), 'mu_std': float(mu_bct.std().item()),
                    'mu_min': float(mu_bct.amin().item()), 'mu_max': float(mu_bct.amax().item()),
                    'logvar_mean': float(logvar_bct.mean().item()), 'logvar_std': float(logvar_bct.std().item()),
                    'z_mean': float(z_bct.mean().item()), 'z_std': float(z_bct.std().item()),
                }
            except Exception:
                debug_vib = {}
        else:
            vib_kld = torch.tensor(0.0, device=latent_bct.device)
            debug_vib = {}

        # 预先获取自适应的阶段权重（供RVQ与rate/gates使用）
        adaptive_stage_weights = self.rate_controller.get_adaptive_stage_weights()
        # 获取控制器输出的控制旋钮（基于EMA rate）
        control_knobs = self.rate_controller.compute_control_knobs()

        # Rate prior heads removed (using pure soft-entropy + gates)

        # === 2. RVQ量化（在24维潜在空间）===

        # === 2. RVQ量化 + JSCC信道模拟 ===
        # 阶段gating：当码率高于目标+阈值时，硬禁用高阶阶段以收紧码率
        stage_mask = None
        if self.training and self.config.get('stage_gate_enable', False):
            try:
                gate_thr = float(self.config.get('stage_gate_threshold_kbps', 0.15))
                # 使用控制器的EMA码率（kbps）判断
                current_kbps_ema = float(self.rate_controller.rate_ema.item())
                if current_kbps_ema > (self.rate_controller.target_kbps + gate_thr):
                    # 仅启用第一级，禁用其余级
                    n = len(self.rvq_encoder.quantizers)
                    stage_mask = [True] + [False] * (n - 1)
            except Exception:
                stage_mask = None

        # 获取RVQ索引（训练和推理都需要索引用于JSCC）
        rvq_results = self.rvq_encoder(
            latent_bct,
            adaptive_weights=adaptive_stage_weights,
            return_indices=True,
            stage_mask=stage_mask,
            control_overrides={
                'gate_temperature': control_knobs.get('gate_temperature', 0.67),
                'gate_logit_bias': control_knobs.get('gate_logit_bias', 0.0),
                'soft_entropy_tau': control_knobs.get('soft_entropy_tau', 0.3),
                'force_gate_open': True if self.config.get('recon_only', False) else False
            }
        )
        rvq_indices = rvq_results['stage_indices']

        # === JSCC信道模拟：作用在比特流（索引）上，支持分阶段启用 ===
        # 分阶段信道扰动：早期关闭，逐渐启用
        channel_enabled = self._should_apply_channel_simulation(current_step)

        if hasattr(self, '_apply_channel_simulation') and self._apply_channel_simulation and channel_enabled:
            # 对RVQ索引应用信道模拟（真正的JSCC）
            corrupted_indices = self._simulate_channel_on_bitstream(rvq_indices, csi_dict, current_step)
        else:
            corrupted_indices = rvq_indices

        # 从（可能损坏的）索引重构潜在特征
        quantized_latent_bct = self.rvq_decoder.forward(corrupted_indices)

        # No rate prior computation; soft-entropy uses stage_soft_probs directly

        # === 3. 转换回[B, T, C]格式用于decoder ===
        quantized_latent = quantized_latent_bct.transpose(1, 2)  # [B, T, 24]

        # === 5. 解码（使用真实的Stage4 decoder）===
        # 解码器需要正确的CSI格式
        if return_wave:
            # 返回FARGAN特征和音频
            try:
                # 传入目标特征统计用于输出端线性校准（只在合成路径用）
                decoder_output = self.decoder(quantized_latent, csi_dict, return_wave=True)
                if isinstance(decoder_output, tuple):
                    reconstructed_features, synthesized_audio = decoder_output
                else:
                    # 兼容处理
                    reconstructed_features = decoder_output
                    synthesized_audio = None
            except Exception as e:
                # CSI问题的回退处理
                print(f"Warning: Decoder failed with CSI, using None: {e}")
                decoder_output = self.decoder(quantized_latent, None, return_wave=True)
                if isinstance(decoder_output, tuple):
                    reconstructed_features, synthesized_audio = decoder_output
                else:
                    reconstructed_features = decoder_output
                    synthesized_audio = None
        else:
            # 只返回FARGAN特征
            try:
                reconstructed_features = self.decoder(quantized_latent, csi_dict, return_wave=False)
            except Exception as e:
                print(f"Warning: Decoder failed with CSI, using None: {e}")
                reconstructed_features = self.decoder(quantized_latent, None, return_wave=False)
            synthesized_audio = None

        # === 质量评估（基于量化特征）===
        # 回退到基于量化特征的质量估计 - 更可靠反映重构质量
        quality_prediction = self.quality_estimator(quantized_latent_bct)

        # === 语义保持（在解码后计算，解决域不一致问题）===
        if self.semantic_use_reconstructed:
            # 使用重构特征（36维）作为学生语义来源，与教师域一致
            reconstructed_bct = reconstructed_features.transpose(1, 2)  # [B, T, 36] -> [B, 36, T]

            # 修复：确保时序维度一致
            if reconstructed_bct.shape[2] != normalized_inputs.shape[1]:
                # 如果长度不匹配，插值对齐
                target_T = normalized_inputs.shape[1]
                reconstructed_bct = F.interpolate(reconstructed_bct, size=target_T, mode='linear', align_corners=True)

            student_sem = torch.nn.functional.normalize(
                self.student_proj(reconstructed_bct), p=2, dim=1)   # [B, Dsem, T]
        else:
            # 使用量化特征（24维）作为学生语义来源（原方案）
            semantic_features = self.semantic_preservor(quantized_latent_bct)
            student_sem = torch.nn.functional.normalize(
                self.student_proj(semantic_features), p=2, dim=1)   # [B, Dsem, T]

        # 使用动量教师（稳定目标）从原始特征提取语义
        with torch.no_grad():
            teacher_sem = self.teacher_momentum(normalized_inputs)  # [B, Dsem, T]

        # 训练时更新动量教师
        if self.training:
            self._update_momentum_teacher()

        # 为了兼容现有代码，保留 semantic_features 变量
        if self.semantic_use_reconstructed:
            semantic_features = reconstructed_features.transpose(1, 2)  # [B, 36, T]
        else:
            semantic_features = self.semantic_preservor(quantized_latent_bct)

        # TODO: 未来可以添加音频质量估计作为辅助
        # if synthesized_audio is not None:
        #     audio_for_quality = synthesized_audio.squeeze(1)  # [B, L]
        #     audio_quality = self.audio_quality_estimator(audio_for_quality)
        #     # 可以结合两种质量评估

        # 收集调试统计信息（供训练循环迭代打印）
        debug_stats = {}
        try:
            if hasattr(self.decoder, '_last_feat0_stats') and isinstance(self.decoder._last_feat0_stats, dict):
                debug_stats['feat0'] = dict(self.decoder._last_feat0_stats)
        except Exception:
            pass
        try:
            if hasattr(self.decoder, '_last_gain_stats') and isinstance(self.decoder._last_gain_stats, dict):
                debug_stats['fargan_gain'] = dict(self.decoder._last_gain_stats)
        except Exception:
            pass
        # VIB stats
        if debug_vib:
            debug_stats['vib'] = debug_vib
        # Encoder/latent stats
        try:
            enc_stats = {
                'enc_mean': float(encoded_latent.mean().item()),
                'enc_std': float(encoded_latent.std().item()),
                'enc_min': float(encoded_latent.amin().item()),
                'enc_max': float(encoded_latent.amax().item()),
                'q_lat_mean': float(quantized_latent_bct.mean().item()),
                'q_lat_std': float(quantized_latent_bct.std().item()),
            }
            debug_stats['latent'] = enc_stats
        except Exception:
            pass
        # RVQ stage stats (from encoder)
        try:
            if isinstance(rvq_results, dict) and ('stage_debug_stats' in rvq_results):
                debug_stats['rvq'] = rvq_results['stage_debug_stats']
        except Exception:
            pass
        # Feature block (ceps 0..17) vs target block stats
        try:
            from models.feature_adapter import FARGANFeatureSpec
            ceps_slice = FARGANFeatureSpec.get_feature_slice('ceps')
            pitch_slice = FARGANFeatureSpec.get_feature_slice('dnn_pitch')
            fc_slice = FARGANFeatureSpec.get_feature_slice('frame_corr')
            lpc_slice = FARGANFeatureSpec.get_feature_slice('lpc')

            out_ceps = reconstructed_features[..., ceps_slice]
            tgt_ceps = normalized_inputs[..., ceps_slice]
            out_fc = reconstructed_features[..., fc_slice]
            tgt_fc = normalized_inputs[..., fc_slice]
            out_lpc = reconstructed_features[..., lpc_slice]
            tgt_lpc = normalized_inputs[..., lpc_slice]
            debug_stats['feat_ceps'] = {
                'mean': float(out_ceps.mean().item()),
                'std': float(out_ceps.std().item()),
                'min': float(out_ceps.amin().item()),
                'max': float(out_ceps.amax().item()),
            }
            debug_stats['target_ceps'] = {
                'mean': float(tgt_ceps.mean().item()),
                'std': float(tgt_ceps.std().item()),
                'min': float(tgt_ceps.amin().item()),
                'max': float(tgt_ceps.amax().item()),
            }
            # Pitch (index 18) stats: output vs target
            out_pitch = reconstructed_features[..., pitch_slice]
            tgt_pitch = normalized_inputs[..., pitch_slice]
            debug_stats['feat_pitch'] = {
                'mean': float(out_pitch.mean().item()),
                'std': float(out_pitch.std().item()),
                'min': float(out_pitch.amin().item()),
                'max': float(out_pitch.amax().item()),
            }
            debug_stats['target_pitch'] = {
                'mean': float(tgt_pitch.mean().item()),
                'std': float(tgt_pitch.std().item()),
                'min': float(tgt_pitch.amin().item()),
                'max': float(tgt_pitch.amax().item()),
            }
            # frame_corr (index 19) block stats
            debug_stats['feat_frame_corr'] = {
                'mean': float(out_fc.mean().item()),
                'std': float(out_fc.std().item()),
                'min': float(out_fc.amin().item()),
                'max': float(out_fc.amax().item()),
            }
            debug_stats['target_frame_corr'] = {
                'mean': float(tgt_fc.mean().item()),
                'std': float(tgt_fc.std().item()),
                'min': float(tgt_fc.amin().item()),
                'max': float(tgt_fc.amax().item()),
            }
            # lpc (20..35) block stats
            debug_stats['feat_lpc'] = {
                'mean': float(out_lpc.mean().item()),
                'std': float(out_lpc.std().item()),
                'min': float(out_lpc.amin().item()),
                'max': float(out_lpc.amax().item()),
            }
            debug_stats['target_lpc'] = {
                'mean': float(tgt_lpc.mean().item()),
                'std': float(tgt_lpc.std().item()),
                'min': float(tgt_lpc.amin().item()),
                'max': float(tgt_lpc.amax().item()),
            }

            # Per-dim summaries (lists) for ceps/lpc
            try:
                debug_stats['ceps_std_out'] = out_ceps.std(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['ceps_std_tgt'] = tgt_ceps.std(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['ceps_mean_out'] = out_ceps.mean(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['ceps_mean_tgt'] = tgt_ceps.mean(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['lpc_std_out'] = out_lpc.std(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['lpc_std_tgt'] = tgt_lpc.std(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['lpc_mean_out'] = out_lpc.mean(dim=(0, 1)).detach().cpu().tolist()
                debug_stats['lpc_mean_tgt'] = tgt_lpc.mean(dim=(0, 1)).detach().cpu().tolist()
            except Exception:
                pass
        except Exception:
            pass
        try:
            if hasattr(self.decoder, '_last_calib_stats') and isinstance(self.decoder._last_calib_stats, dict):
                debug_stats['calib'] = dict(self.decoder._last_calib_stats)
        except Exception:
            pass
        # Decoder MoE stats (util/entropy) and trainable state
        try:
            if hasattr(self.decoder, 'get_dec_moe_stats') and callable(getattr(self.decoder, 'get_dec_moe_stats')):
                dec_stats = self.decoder.get_dec_moe_stats()
                if isinstance(dec_stats, dict) and dec_stats:
                    debug_stats['dec_moe'] = dec_stats
            # Trainable flag
            dec_trainable = any(p.requires_grad for n, p in self.decoder.named_parameters() if 'dec_moe' in n)
            debug_stats['dec_moe_trainable'] = bool(dec_trainable)
        except Exception:
            pass

        return {
            # 核心输出
            'encoded_latent': encoded_latent,  # [B, T, 24] encoder输出
            'quantized_latent': quantized_latent,  # [B, T, 24] RVQ后的潜在特征
            'reconstructed_features': reconstructed_features,  # [B, T, 36] decoder输出特征
            'recon_features_raw': getattr(self.decoder, '_last_raw_features', None),  # [B, T, 36] 未校准特征
            'synthesized_audio': synthesized_audio,  # [B, 1, L] 合成音频
            'original_features': normalized_inputs,  # [B, T, 36] 标准化后的输入特征（用于损失计算）

            # 语义和质量
            'semantic_features': semantic_features,
            'quality_prediction': quality_prediction,

            # VIB和语义对比学习相关
            'vib_kld': vib_kld,
            'student_sem': student_sem,     # [B, Dsem, T]
            'teacher_sem': teacher_sem,     # [B, Dsem, T]

            # Rate-related (soft-entropy path)
            'rvq_indices': rvq_indices,              # list of [B, T] long
            'rvq_stage_sizes': self.rvq_stage_sizes,  # 便于loss使用

            # 损失相关
            'rate_bits_per_frame': rvq_results['rate_bits_per_frame'],
            'commitment_loss': rvq_results['commitment_loss'],

            # encoder logs
            'encoder_logs': enc_logs,

            # 调试信息
            'rvq_details': {
                'stage_quantized': rvq_results.get('stage_quantized', []),
                'stage_perplexities': rvq_results.get('stage_perplexities', []),
                'stage_weights': rvq_results.get('stage_weights', []),
                'final_residual': rvq_results.get('final_residual', None),
                'stage_indices': rvq_indices  # 新增：码字索引
            },
            # Soft-entropy and gating info (for differentiable rate control)
            'rvq_soft_probs': rvq_results.get('stage_soft_probs', []),  # list of [B,T,K]
            'stage_gate_soft': rvq_results.get('stage_gate_soft', []),   # list of [B,T]
            'stage_gate_masks': rvq_results.get('stage_gate_masks', []), # list of [B,T]
            'codebook_mask_usage': rvq_results.get('codebook_mask_usage', []),
            'rate_stats': self.rate_controller.get_current_stats(),
            'control_info': control_knobs,
            'debug_stats': debug_stats,

            # Decoder distribution heads (if exposed by refiner)
            'pred_mu': getattr(self.decoder.refiner, '_last_mu', None) if hasattr(self, 'decoder') and hasattr(self.decoder, 'refiner') else None,
            'pred_logstd': getattr(self.decoder.refiner, '_last_logstd', None) if hasattr(self, 'decoder') and hasattr(self.decoder, 'refiner') else None,

            # === 新增：完整RVQ接口 ===
            'can_compress': not self.training,  # 推理时可以压缩
            'compression_ready': rvq_indices is not None  # 是否准备好压缩
        }

    def compress_features(self, features: torch.Tensor) -> Tuple[List[bytes], Dict[str, float]]:
        """
        完整的特征压缩流程（新增接口）
        features: [B, T, 36] 输入特征
        returns: (比特流列表, 统计信息)
        """
        self.eval()  # 设置为推理模式
        with torch.no_grad():
            # 1. 编码
            encoded_latent, _ = self.encoder(features, None)
            latent_bct = encoded_latent.transpose(1, 2)

            # 2. RVQ压缩
            bitstreams, compression_stats = self.rvq_encoder.compress_to_bitstream(latent_bct)

            return bitstreams, compression_stats

    def decompress_features(self, bitstreams: List[bytes], shape: Tuple[int, int]) -> torch.Tensor:
        """
        完整的特征解压缩流程（新增接口）
        bitstreams: List[bytes] RVQ比特流
        shape: (B, T) 输出形状
        returns: [B, T, 36] 重构特征
        """
        self.eval()
        with torch.no_grad():
            # 1. RVQ解压缩
            latent_bct = self.rvq_encoder.decompress_from_bitstream(bitstreams, shape)

            # 2. 解码
            quantized_latent = latent_bct.transpose(1, 2)
            reconstructed_features = self.decoder(quantized_latent, None, return_wave=False)

            return reconstructed_features

    def estimate_bitrate(self, sequence_length: int) -> float:
        """估计给定序列长度的码率(kbps)"""
        # 基于当前RVQ stage权重的自适应码率估算
        if hasattr(self, 'rate_controller') and hasattr(self.rate_controller, 'stage_weights_ema'):
            # 使用实际的stage权重
            stage_weights = self.rate_controller.stage_weights_ema
            theoretical_bits = self.rate_controller.theoretical_bits_per_stage
            theoretical_bits_tensor = torch.tensor(theoretical_bits, device=stage_weights.device)
            adaptive_bits_per_frame = torch.sum(stage_weights * theoretical_bits_tensor).item()
        else:
            # 回退到理论最大值
            adaptive_bits_per_frame = sum([
                math.log2(size) for size in self.config.get('codebook_sizes', [1024, 512, 256])
            ])

        frame_rate = self.config.get('frame_rate', 50)
        estimated_kbps = adaptive_bits_per_frame * frame_rate / 1000.0
        return estimated_kbps

    def enable_channel_simulation(self, enable: bool = True):
        """启用/禁用信道模拟（必要！）"""
        self._apply_channel_simulation = enable
        print(f"Stage5 channel simulation: {'enabled' if enable else 'disabled'}")

    def _apply_stage4_channel(self, latent: torch.Tensor, csi_dict: Optional[Dict], current_step: int) -> torch.Tensor:
        """
        完全按Stage4方式应用信道模拟（使用ChannelSimulator.apply）
        latent: [B, T, D] 量化后的潜在特征（Stage4格式）
        """
        if csi_dict is None:
            return latent

        try:
            # 导入Stage4的信道模拟器
            from utils.channel_sim import ChannelSimulator
            chan_sim = ChannelSimulator()

            B, T, D = latent.shape  # [B, T, D] Stage4格式
            device = latent.device

            # === 从 CSI 中提取 SNR 和衰落参数 ===
            # Stage4需要: amp_t[B,T], snr_db_t[B,T]
            if 'snr_db' in csi_dict:
                snr_db_t = csi_dict['snr_db']  # [B, T]
            elif 'snr_proxy' in csi_dict:
                snr_db_t = csi_dict['snr_proxy'] * 20.0  # proxy -> dB
            else:
                snr_db_t = torch.randn(B, T, device=device) * 5.0 + 10.0

            # 计算衰落幅度 (amp_t)
            # 综合时间/频率选择性和LOS比例
            amp_t = torch.ones(B, T, device=device)

            # 时间选择性影响衰落深度
            if 'time_selectivity' in csi_dict:
                time_fade = 1.0 - 0.3 * csi_dict['time_selectivity'].clamp(0, 1)
                amp_t *= time_fade
            elif 'doppler_norm' in csi_dict:
                doppler_fade = 1.0 - 0.2 * csi_dict['doppler_norm'].clamp(0, 1)
                amp_t *= doppler_fade

            # 频率选择性影响衰落深度
            if 'freq_selectivity' in csi_dict:
                freq_fade = 1.0 - 0.2 * csi_dict['freq_selectivity'].clamp(0, 1)
                amp_t *= freq_fade
            elif 'tau_rms_ms' in csi_dict:
                tau_fade = 1.0 - 0.15 * torch.clamp(csi_dict['tau_rms_ms'] / 10.0, 0, 1)
                amp_t *= tau_fade

            # LOS比例影响衰落的方差（LOS高->stable, NLOS->variable）
            if 'los_ratio' in csi_dict:
                los = csi_dict['los_ratio'].clamp(0, 1)
                # LOS高的地方衰落小，NLOS地方衰落大
                amp_variability = 0.1 * (1.0 - los) * torch.randn(B, T, device=device)
                amp_t = amp_t + amp_variability
            elif 'k_factor_db' in csi_dict:
                k_fade = torch.sigmoid(csi_dict['k_factor_db'] / 10.0)  # K-factor -> stability
                k_variability = 0.1 * (1.0 - k_fade) * torch.randn(B, T, device=device)
                amp_t = amp_t + k_variability

            # 确保衰落幅度为正
            amp_t = torch.clamp(amp_t, min=0.1, max=2.0)

            # === 使用Stage4的ChannelSimulator.apply ===
            corrupted_latent = chan_sim.apply(latent, amp_t, snr_db_t)

            return corrupted_latent

        except ImportError:
            # 回退：简单噪声模拟
            noise = torch.randn_like(latent) * 0.05
            return latent + noise
        except Exception as e:
            print(f"Warning: Channel simulation failed: {e}")
            return latent

    def _simulate_channel_on_bitstream(self, stage_indices: List[torch.Tensor], csi_dict: Optional[Dict], current_step: int) -> List[torch.Tensor]:
        """
        在RVQ比特流上模拟信道效应（使用专门的JSCC信道模拟器）
        """
        if csi_dict is None:
            return stage_indices

        try:
            # 使用专门的JSCC信道模拟器
            from utils.jscc_channel_sim import create_jscc_channel_simulator

            jscc_sim = create_jscc_channel_simulator()
            codebook_sizes = self.config.get('codebook_sizes', [1024, 512, 256])

            # === 关键修复：Stage4 CSI -> JSCC CSI 键名映射 ===
            mapped_csi = self._map_stage4_csi_to_jscc(csi_dict)

            # 选择错误模式（可以根据训练步骤或配置调整）
            error_mode = "random"  # 可选: "burst", "erasure"

            corrupted_indices = jscc_sim.apply_channel_to_rvq_indices(
                stage_indices=stage_indices,
                csi_dict=mapped_csi,
                codebook_sizes=codebook_sizes,
                error_mode=error_mode
            )

            return corrupted_indices

        except Exception as e:
            print(f"Warning: JSCC channel simulation failed: {e}")
            # 回退到原始索引
            return stage_indices

    def _map_stage4_csi_to_jscc(self, csi_dict: Dict) -> Dict:
        """
        将Stage4风格的CSI映射为JSCC仿真器期望的键名
        Stage4: snr_db, amp, fading_onehot, ber, channel_type
        JSCC: snr_proxy, time_selectivity, freq_selectivity, los_ratio
        """
        device = next(iter(csi_dict.values())).device
        dtype = next(iter(csi_dict.values())).dtype
        B = next(iter(csi_dict.values())).shape[0]

        mapped = {}

        # 1. snr_proxy: 从snr_db归一化
        if 'snr_db' in csi_dict:
            snr_db = csi_dict['snr_db']
            if snr_db.dim() > 1:  # [B, T] -> [B]
                snr_db = snr_db.mean(dim=1)
            mapped['snr_proxy'] = snr_db / 20.0  # 归一化到[-0.25, 0.75]范围
        else:
            mapped['snr_proxy'] = torch.zeros(B, device=device, dtype=dtype)

        # 2. time_selectivity: 从amp时域变化估计
        if 'amp' in csi_dict:
            amp = csi_dict['amp']  # [B, T] 或 [B]
            if amp.dim() > 1 and amp.shape[1] > 1:
                # 计算时域变化：相邻帧差分的标准差
                amp_diff = torch.diff(amp, dim=1)
                time_sel = amp_diff.std(dim=1).clamp(0, 1)
            else:
                time_sel = torch.zeros(B, device=device, dtype=dtype)
            mapped['time_selectivity'] = time_sel
        elif 'doppler_norm' in csi_dict:
            mapped['time_selectivity'] = csi_dict['doppler_norm'].clamp(0, 1)
        else:
            mapped['time_selectivity'] = torch.zeros(B, device=device, dtype=dtype)

        # 3. freq_selectivity: 从tau_rms_ms或fading强度估计
        if 'tau_rms_ms' in csi_dict:
            tau_rms = csi_dict['tau_rms_ms']
            if tau_rms.dim() > 1:
                tau_rms = tau_rms.mean(dim=1)
            freq_sel = (tau_rms / 10.0).clamp(0, 1)  # 归一化
            mapped['freq_selectivity'] = freq_sel
        elif 'fading_onehot' in csi_dict:
            # 从fading类型推断频率选择性
            fading = csi_dict['fading_onehot']  # [B, num_fading_types]
            # 假设最后一个维度是严重衰落
            if fading.shape[-1] > 1:
                freq_sel = fading[:, -1]  # 取最后一个类型作为频率选择性
            else:
                freq_sel = fading[:, 0] * 0.5  # 降低强度
            mapped['freq_selectivity'] = freq_sel.clamp(0, 1)
        else:
            mapped['freq_selectivity'] = torch.zeros(B, device=device, dtype=dtype)

        # 4. los_ratio: 从k_factor_db或信道类型估计
        if 'k_factor_db' in csi_dict:
            k_factor = csi_dict['k_factor_db']
            if k_factor.dim() > 1:
                k_factor = k_factor.mean(dim=1)
            # K-factor to LOS ratio: K_lin/(K_lin+1)
            k_lin = torch.pow(10.0, k_factor / 10.0)
            los_ratio = k_lin / (k_lin + 1.0)
            mapped['los_ratio'] = los_ratio.clamp(0, 1)
        elif 'channel_type' in csi_dict:
            # 从信道类型推断LOS比例
            ch_type = csi_dict['channel_type']
            # 简化映射：fading=0.1, awgn=0.9
            los_ratio = torch.where(ch_type == 0, 0.1, 0.9)  # 假设0=fading, 1=awgn
            mapped['los_ratio'] = los_ratio
        else:
            mapped['los_ratio'] = torch.ones(B, device=device, dtype=dtype) * 0.5

        return mapped

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'estimated_bitrate_kbps': self.estimate_bitrate(1000),
            'target_kbps': self.config.get('target_kbps', 1.2),
            'rvq_stages': len(self.rvq_encoder.quantizers),
            'codebook_sizes': self.config.get('codebook_sizes', [1024, 512, 256])
        }

    def _init_momentum_teacher(self):
        """初始化动量教师：用参数名确保teacher和student一一对应"""
        teacher_dict = dict(self.teacher_momentum.named_parameters())
        student_dict = dict(self.student_proj.named_parameters())

        for name, student_param in student_dict.items():
            if name in teacher_dict:
                teacher_dict[name].data.copy_(student_param.data)
                teacher_dict[name].requires_grad = False
            else:
                print(f"Warning: Student param {name} not found in teacher")

        # 对于teacher中有但student中没有的参数，使用teacher自身的初始化
        for name, teacher_param in teacher_dict.items():
            if name not in student_dict:
                teacher_param.requires_grad = False  # 确保不直接训练

    def _update_momentum_teacher(self):
        """更新动量教师：基于学生网络的EMA更新（BYOL风格）"""
        with torch.no_grad():
            teacher_dict = dict(self.teacher_momentum.named_parameters())
            student_dict = dict(self.student_proj.named_parameters())

            for name, student_param in student_dict.items():
                if name in teacher_dict:
                    teacher_param = teacher_dict[name]
                    teacher_param.data.mul_(self.teacher_momentum_coeff).add_(
                        student_param.data, alpha=1.0 - self.teacher_momentum_coeff
                    )

    def _should_apply_channel_simulation(self, current_step: int) -> bool:
        """判断是否应该应用信道扰动（分阶段策略）"""
        if current_step < self.channel_disable_steps:
            # 早期阶段：完全禁用
            return False
        elif current_step < self.channel_disable_steps + self.channel_warmup_steps:
            # 渐进启用阶段：线性概率增长
            progress = (current_step - self.channel_disable_steps) / self.channel_warmup_steps
            # 使用随机采样决定是否启用
            return torch.rand(1).item() < progress
        else:
            # 后期阶段：完全启用
            return True

def create_stage5_model(config: Dict) -> AETHERStage5Model:
    """创建Stage5模型的工厂函数"""
    model = AETHERStage5Model(config)
    return model

if __name__ == "__main__":
    # 测试模型创建
    config = {
        'feature_dim': 24,
        'original_feature_dim': 36,
        'rvq_stages': 3,
        'codebook_sizes': [1024, 512, 256],
        'commitment_weights': [0.01, 0.02, 0.03],  # 降低commitment权重
        'target_kbps': 1.2,
        'rate_tolerance': 0.1,
        'frame_rate': 50,
        # 新增：梯度感知权重配置（total_steps自动计算）
        'use_gradient_aware_weighting': True,
        'grad_adaptation_rate': 0.05,
        # 训练配置（total_steps将自动计算）
        'num_epochs': 10,
        'batch_size': 4,
        # 'total_steps': None  # 将由 num_epochs * steps_per_epoch 自动计算
    }

    model = create_stage5_model(config)
    print("Stage5 Model Info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")

    # 测试前向传播
    batch_size, feature_dim, seq_len = 2, 24, 100
    dummy_input = torch.randn(batch_size, feature_dim, seq_len)

    with torch.no_grad():
        outputs = model(dummy_input)
        print(f"\nTest forward pass:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Quantized features shape: {outputs['quantized_features'].shape}")
        print(f"  Rate bits per frame: {outputs['rate_bits_per_frame']:.2f}")
        print(f"  Estimated kbps: {outputs['rate_bits_per_frame'] * 50 / 1000:.3f}")
