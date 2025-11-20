#!/usr/bin/env python3
"""
Stage5 损失函数模块

包含:
1. 多目标率失真损失
2. 语义保持损失
3. 时间一致性损失
4. 动态权重调度
5. 综合损失计算
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math

# Constants for bits conversion
LOG2E = 1.0 / math.log(2.0)

# 导入现有的波形损失 - 修复P3: 确保路径一致性，失败时显式报错
try:
    from .pipeline.wave_loss import fargan_wave_losses
    WAVE_LOSS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import fargan_wave_losses from pipeline.wave_loss: {e}")
    try:
        # 尝试从上级目录导入
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
        from pipeline.wave_loss import fargan_wave_losses
        WAVE_LOSS_AVAILABLE = True
        print("Successfully imported fargan_wave_losses from fallback path")
    except ImportError as e2:
        print(f"Critical: Wave loss import failed completely: {e2}")
        WAVE_LOSS_AVAILABLE = False
        # 抛出错误而不是悄然降级
        def fargan_wave_losses(pred_audio, target_audio, period, device):
            """错误：波形损失不可用"""
            raise RuntimeError(f"Wave loss not available. Import errors: {e}, {e2}")

def _ensure_wave_loss_available():
    """确保波形损失可用，否则抛出明确错误"""
    if not WAVE_LOSS_AVAILABLE:
        raise RuntimeError("Wave loss is required but not available. Check import paths.")


# === [SEM] InfoNCE (NT-Xent) ===============================================
def info_nce_global(z1_bct, z2_bct, temperature=0.2):
    """Improved InfoNCE loss with better temperature and pooling"""
    B, D, T = z1_bct.shape

    # 使用更智能的池化策略：加权平均而不是简单均值
    # 计算每个时间步的"重要性"权重
    z1_importance = torch.softmax(z1_bct.norm(dim=1), dim=-1)  # [B, T]
    z2_importance = torch.softmax(z2_bct.norm(dim=1), dim=-1)  # [B, T]

    # 加权池化
    z1 = (z1_bct * z1_importance.unsqueeze(1)).sum(dim=-1)  # [B, D]
    z2 = (z2_bct * z2_importance.unsqueeze(1)).sum(dim=-1)  # [B, D]

    # L2归一化
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    # 计算相似度矩阵
    logits = (z1 @ z2.t()) / temperature      # [B, B]
    labels = torch.arange(B, device=z1.device)

    return F.cross_entropy(logits, labels)

def nt_xent_loss(student_bct: torch.Tensor, teacher_bct: torch.Tensor, temperature: float = 0.1,
                 current_step: int = 0, adaptive_temp: bool = True) -> torch.Tensor:
    # student/teacher: [B, D, T] 已经 L2-normalized
    B, D, T = student_bct.shape

    # 修复3&9: 温度schedule + 减少伪负样本策略

    # 温度schedule: cosine annealing from 0.25 -> 0.07 over 6000 steps
    if adaptive_temp and current_step > 0:
        max_steps = 6000
        progress = min(current_step / max_steps, 1.0)
        temp_start, temp_end = 0.25, 0.07
        # Cosine annealing 提供平滑过渡
        temperature = temp_end + 0.5 * (temp_start - temp_end) * (1 + math.cos(math.pi * progress))

    # 1. 计算每帧的能量（L2 norm）
    s_energy = torch.norm(student_bct, dim=1)  # [B, T]
    t_energy = torch.norm(teacher_bct, dim=1)  # [B, T]

    # 2. 设定能量阈值（取25%分位数，过滤掉最低25%能量帧）
    energy_threshold = torch.quantile(s_energy.flatten(), 0.25)  # 25%分位数，保留中高能量帧

    # 3. 创建掩码，保留中高能量帧
    valid_mask = (s_energy > energy_threshold) & (t_energy > energy_threshold)  # [B, T]

    # 4. 温度适应k选择：早期用小k，避免过难；低温度后用大k增加负样本
    if temperature > 0.15:  # 早期高温阶段
        k = max(T // 5, 6)   # 修复3: 更小k，严格避免伪负样本
    else:  # 低温阶段
        k = max(T // 3, 12)  # 中等k，平衡负样本数量和伪负样本风险
    s_pooled_list = []
    t_pooled_list = []
    batch_indicators = []  # 记录每个样本来自哪个batch，用于同语句掩码

    for b in range(B):
        if valid_mask[b].sum() >= k:
            # 在有效帧中选Top-k，修复：取student和teacher能量的并集
            valid_indices = torch.where(valid_mask[b])[0]
            s_valid_energy = s_energy[b, valid_indices]
            t_valid_energy = t_energy[b, valid_indices]

            # 分别取student和teacher的Top-k/2
            half_k = max(k // 2, 4)  # 确保至少4个
            s_top_k = valid_indices[torch.topk(s_valid_energy, min(half_k, len(valid_indices)))[1]]
            t_top_k = valid_indices[torch.topk(t_valid_energy, min(half_k, len(valid_indices)))[1]]

            # 合并并去重，确保总数接近k
            combined_indices = torch.cat([s_top_k, t_top_k])
            top_k_indices = torch.unique(combined_indices)[:k]  # 去重并截断到k个
        else:
            # 回退到全局Top-k，同样使用并集策略
            half_k = max(k // 2, 4)
            s_top_k = torch.topk(s_energy[b], half_k)[1]
            t_top_k = torch.topk(t_energy[b], half_k)[1]
            combined_indices = torch.cat([s_top_k, t_top_k])
            top_k_indices = torch.unique(combined_indices)[:k]

        k_actual = len(top_k_indices)
        s_pooled_list.append(student_bct[b, :, top_k_indices].transpose(0, 1))  # [k_actual, D]
        t_pooled_list.append(teacher_bct[b, :, top_k_indices].transpose(0, 1))  # [k_actual, D]
        batch_indicators.extend([b] * k_actual)  # 记录batch索引

    # 5. 重新组织为[total_samples, D]格式
    s = torch.cat(s_pooled_list, dim=0)  # [total_samples, D]
    t = torch.cat(t_pooled_list, dim=0)  # [total_samples, D]
    batch_ids = torch.tensor(batch_indicators, device=s.device)  # [total_samples]

    # 6. 计算对比损失 - 由于去重，实际样本数可能 < B*k
    total_samples = s.size(0)
    if total_samples == 0:
        return torch.tensor(0.0, device=student_bct.device, requires_grad=True)

    # 修复9: 温度与有效负样本数量耦合调整
    effective_negatives = total_samples * (total_samples - 1)  # 去掉对角线后的有效负样本对数
    if effective_negatives > 1000:  # 大batch时，轻微降低温度增强对比
        batch_scale_factor = min(1.3, math.sqrt(effective_negatives / 1000.0))
        adjusted_temperature = temperature * batch_scale_factor
    else:
        adjusted_temperature = temperature

    logits = (s @ t.t()) / adjusted_temperature   # [total_samples, total_samples]

    # 7. 同语句掩码：屏蔽同一batch内的负样本对角线以外部分
    same_batch_mask = batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1)  # [total_samples, total_samples]
    eye_mask = torch.eye(total_samples, device=logits.device, dtype=torch.bool)
    # 对同一语句内的非对角线位置（伪负样本）施加大负值屏蔽
    logits = logits.masked_fill(same_batch_mask & ~eye_mask, -float('inf'))

    labels = torch.arange(total_samples, device=s.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def compute_temporal_consistency_loss(
    quantized_sequence: torch.Tensor,
    smoothness_weight: float = 0.1,   # 恢复正常权重
    max_jump_threshold: float = 2.0   # 恢复原始阈值
) -> torch.Tensor:
    """
    计算时间一致性损失，确保量化后序列的平滑性

    Args:
        quantized_sequence: [B, D, T] 量化后的特征序列
        smoothness_weight: 平滑性权重
        max_jump_threshold: 最大跳跃阈值
    """
    if quantized_sequence.size(-1) < 2:
        return torch.tensor(0.0, device=quantized_sequence.device)

    # 对输入进行L2归一化避免scale问题
    normalized_seq = F.normalize(quantized_sequence, p=2, dim=1)

    # 计算帧间差分
    diff = normalized_seq[:, :, 1:] - normalized_seq[:, :, :-1]  # [B, D, T-1]

    # L1平滑损失（已经归一化，数值会更合理）
    l1_smooth = F.l1_loss(diff, torch.zeros_like(diff), reduction='none').mean(dim=1)  # [B, T-1]

    # 大跳跃惩罚（基于归一化后的差值）
    jump_magnitude = torch.norm(diff, p=2, dim=1)  # [B, T-1]
    jump_penalty = F.relu(jump_magnitude - max_jump_threshold).pow(2)

    # 组合损失（正常权重）
    total_loss = smoothness_weight * l1_smooth.mean() + jump_penalty.mean()

    return total_loss

def compute_real_pesq_loss(
    model_outputs: Dict[str, Any],
    targets: Dict[str, Any],
    minimum_pesq_threshold: float = 2.5,
    current_step: int = 0,
    device: torch.device = None
) -> torch.Tensor:
    """
    使用真实PESQ计算质量损失

    Args:
        model_outputs: 包含synthesized_audio的模型输出
        targets: 包含target_audio的目标数据
        minimum_pesq_threshold: 最低PESQ阈值
        current_step: 当前训练步骤
        device: 设备
    """
    try:
        # 检查是否有合成音频
        synthesized_audio = model_outputs.get('synthesized_audio')  # [B, 1, L]
        target_audio = targets.get('target_audio')  # [B, L]

        if synthesized_audio is None or target_audio is None:
            # 如果没有音频，返回0损失
            if current_step % 100 == 0:
                print(f"[PESQ] Step {current_step}: No audio available, PESQ loss disabled")
            return torch.tensor(0.0, device=device)

        # 导入PESQ（延迟导入避免启动时的依赖问题）
        try:
            from pesq import pesq
        except ImportError:
            if current_step % 100 == 0:
                print(f"[PESQ] Step {current_step}: PESQ package not available, using fallback")
            return torch.tensor(0.0, device=device)

        # 转换音频格式
        if synthesized_audio.dim() == 3:  # [B, 1, L]
            synth_audio = synthesized_audio.squeeze(1)  # [B, L]
        else:
            synth_audio = synthesized_audio

        # 确保长度匹配（取较短的长度）
        min_length = min(synth_audio.shape[1], target_audio.shape[1])
        synth_audio = synth_audio[:, :min_length]
        target_audio = target_audio[:, :min_length]

        # 计算每个样本的PESQ
        pesq_scores = []
        sample_rate = 16000  # 假设16kHz采样率

        for i in range(synth_audio.shape[0]):
            # 转换为numpy并归一化到[-1, 1]
            ref_audio = target_audio[i].detach().cpu().numpy()
            deg_audio = synth_audio[i].detach().cpu().numpy()

            # 归一化音频
            ref_audio = np.clip(ref_audio / np.abs(ref_audio).max().clip(1e-7, None), -1, 1)
            deg_audio = np.clip(deg_audio / np.abs(deg_audio).max().clip(1e-7, None), -1, 1)

            try:
                # 计算PESQ (wb模式，支持16kHz)
                pesq_score = pesq(sample_rate, ref_audio, deg_audio, 'wb')
                pesq_scores.append(pesq_score)
            except Exception as e:
                # PESQ计算失败时使用低分
                pesq_scores.append(1.0)

        # 转换为tensor
        pesq_tensor = torch.tensor(pesq_scores, device=device, dtype=torch.float32)

        # 计算质量损失：低于阈值时惩罚
        quality_penalty = F.relu(minimum_pesq_threshold - pesq_tensor).pow(2)

        # 调试输出
        if current_step % 50 == 0:
            print(f"[Real PESQ] Step {current_step}: PESQ range [{pesq_tensor.min():.3f}, {pesq_tensor.max():.3f}], "
                  f"threshold={minimum_pesq_threshold}, penalty={quality_penalty.mean():.6f}")

        return quality_penalty.mean()

    except Exception as e:
        # 出错时返回0损失，避免训练中断
        if current_step % 100 == 0:
            print(f"[PESQ] Step {current_step}: PESQ computation failed: {e}")
        return torch.tensor(0.0, device=device)

    

class GradientAwareLossWeights:
    """基于梯度感知的动态损失权重调度器"""

    def __init__(self, total_steps: int = 8000, adaptation_rate: float = 0.1):
        self.total_steps = total_steps
        self.adaptation_rate = adaptation_rate
        # 梯度历史记录
        self.gradient_history = {
            'feat': [],
            'wave': [],
            'semantic': [],
            'quality': [],
            'commitment': []
        }
        # 当前权重（只保留有效的loss项）
        self.current_weights = {
            'feat': 0.5,        # 特征重构权重
            'wave': 0.6,        # 波形损失权重
            'semantic': 0.4,     # 语义保持权重
            'quality': 0.3,      # 质量估计权重
            'commitment': 0.05   # VQ commitment权重
        }

    def compute_gradient_magnitudes(self, model, individual_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        基于“共享干线”参数集合估计各损失的梯度范数（GradNorm风格）。
        共享集合默认包含：encoder.*, rvq_encoder.*, decoder.refiner.*；
        排除：fargan_core, period_estimator, teacher(_momentum), rate_controller。
        """
        grad_magnitudes: Dict[str, float] = {}

        # 收集共享干线参数
        shared_params = []
        for name, p in model.named_parameters():
            if not (p.requires_grad and p.is_leaf):
                continue
            n = name.lower()
            include = (('encoder.' in n) or ('rvq_encoder' in n) or ('decoder.refiner' in n))
            exclude = ('fargan_core' in n) or ('period_estimator' in n) or ('teacher' in n) or ('rate_controller' in n)
            if include and (not exclude):
                shared_params.append(p)
        # 退化处理：若没收集到，使用全部 requires_grad 作为共享集合
        if not shared_params:
            shared_params = [p for _, p in model.named_parameters() if p.requires_grad]

        for loss_name, loss_tensor in individual_losses.items():
            try:
                if (loss_tensor is None) or (not torch.is_tensor(loss_tensor)) or (not loss_tensor.requires_grad):
                    grad_magnitudes[loss_name] = 0.0
                    continue
                grads = torch.autograd.grad(loss_tensor, shared_params,
                                            retain_graph=True, create_graph=False, allow_unused=True)
                # L2范数聚合
                sq = 0.0
                for g in grads:
                    if g is not None:
                        sq = sq + float(g.pow(2).sum().item())
                grad_magnitudes[loss_name] = float((sq + 1e-12) ** 0.5)
            except Exception:
                grad_magnitudes[loss_name] = float(loss_tensor.item()) if torch.is_tensor(loss_tensor) else 0.0

        return grad_magnitudes

    def update_weights_based_on_gradients(self, model, individual_losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """基于梯度动态更新权重"""

        # 计算梯度幅度
        grad_mags = self.compute_gradient_magnitudes(model, individual_losses)

        # 调试：检查梯度计算结果
        if any(v > 0 for v in grad_mags.values()):
            grad_debug = ", ".join([f"{k}:{v:.2e}" for k, v in grad_mags.items() if v > 0])
        else:
            grad_debug = "all_zero"

        # 更新梯度历史
        for key, mag in grad_mags.items():
            if key in self.gradient_history:
                self.gradient_history[key].append(mag)
                # 保持历史长度
                if len(self.gradient_history[key]) > 20:
                    self.gradient_history[key] = self.gradient_history[key][-20:]

        # 计算相对梯度强度
        total_grad = sum(grad_mags.values()) + 1e-8
        relative_grads = {k: v / total_grad for k, v in grad_mags.items()}

        # 动态调整权重：梯度越大，权重越小（防止主导）
        for key in self.current_weights:
            if key in relative_grads and key in individual_losses:
                rel_grad = relative_grads[key]
                current_loss = individual_losses[key].item()

                # 综合考虑梯度大小和loss大小
                if rel_grad > 0.3:  # 梯度过大
                    target_weight = self.current_weights[key] * 0.8  # 降低权重
                elif rel_grad < 0.05:  # 梯度过小
                    target_weight = self.current_weights[key] * 1.2  # 增加权重
                else:
                    target_weight = self.current_weights[key]  # 保持权重

                # 权重范围限制
                target_weight = max(0.01, min(1.5, target_weight))

                # 平滑更新
                self.current_weights[key] = (
                    (1 - self.adaptation_rate) * self.current_weights[key] +
                    self.adaptation_rate * target_weight
                )

        return self.current_weights.copy()

    def get_weights(self, current_step: int) -> Dict[str, float]:
        """获取当前权重（兼容原接口）"""
        return self.current_weights.copy()

    def get_rate_warmup_factor(self, step: int, warmup_steps: int) -> float:
        """码率损失预热因子"""
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0


class AdaptiveLossWeights:
    """动态损失权重调度器（原版）"""

    def __init__(self, total_steps: int = 8000):
        self.total_steps = total_steps

    def get_weights(self, current_step: int) -> Dict[str, float]:
        """
        根据训练步骤返回动态权重

        三阶段策略:
        1. 码率探索 (0-1000步): 重点学习码率控制
        2. 质量-码率平衡 (1000-3000步): 平衡质量和码率
        3. 质量优化 (3000+步): 优先保证质量
        """
        progress = current_step / self.total_steps

        if current_step < 1000:
            # 阶段1：码率探索期
            return {
                'feat': 0.5,      # 特征重构
                'wave': 0.6,      # 波形质量
                'semantic': 0.4,   # 语义保持
                'quality': 0.2,    # 质量估计
                'commitment': 0.3  # commitment权重
            }
        elif current_step < 3000:
            # 阶段2：质量-码率平衡期
            alpha = (current_step - 1000) / 2000  # 0到1的进度
            return {
                'feat': 0.5 - 0.1 * alpha,     # 0.5 -> 0.4
                'wave': 0.6 + 0.1 * alpha,     # 0.6 -> 0.7
                'semantic': 0.4,               # 保持不变
                'quality': 0.2 + 0.1 * alpha,   # 0.2 -> 0.3
                'commitment': 0.3 - 0.1 * alpha # 0.3 -> 0.2
            }
        else:
            # 阶段3：质量优化期
            return {
                'feat': 0.3,      # 适度特征重构
                'wave': 0.8,      # 优先保证质量
                'semantic': 0.3,   # 增强语义一致性
                'quality': 0.4,    # 增强质量监督
                'commitment': 0.1  # 降低量化约束
            }

    def get_rate_warmup_factor(self, current_step: int, warmup_steps: int = 500) -> float:
        """码率约束预热因子 - 禁用warmup让rate loss立即生效"""
        # 禁用warmup，让rate loss从第一步就有效
        return 1.0

class Stage5ComprehensiveLoss:
    """Stage5综合损失计算器"""

    def __init__(self, config: Dict):
        self.config = config

        # 自动计算总训练步数
        total_steps = self._calculate_total_steps(config)

        # 选择权重调度器类型
        enable_gradient_balancing = config.get('enable_gradient_balancing', True)
        if enable_gradient_balancing:
            self.weight_scheduler = GradientAwareLossWeights(
                total_steps=total_steps,
                adaptation_rate=config.get('gradient_adaptation_rate', 0.01)
            )
            self.is_gradient_aware = True
            print("✅ Multi-task gradient balancing enabled")
            print(f"   Alpha: {config.get('gradient_balance_alpha', 0.16)}")
            print(f"   Adaptation rate: {config.get('gradient_adaptation_rate', 0.01)}")
        else:
            self.weight_scheduler = AdaptiveLossWeights(
                total_steps=total_steps
            )
            self.is_gradient_aware = False
            print("📊 Using adaptive loss weights without gradient balancing")

        # 注入CLI权重设置（修复CLI参数未生效的问题）
        init_w = self.config.get('initial_loss_weights')
        if isinstance(init_w, dict):
            for k, v in init_w.items():
                if k in self.weight_scheduler.current_weights and isinstance(v, (int, float)):
                    self.weight_scheduler.current_weights[k] = float(v)
                    print(f"   ✅ CLI权重注入: {k}={v}")

        # 损失历史(用于稳定性监控)
        self.loss_history = {
            'rate': [],
            'quality': [],
            'stability': []
        }

        # Dual-ascent state for closed-loop bitrate control
        self._lambda_rate: float = float(self.config.get('initial_lambda_rate', 0.0))
        self._rate_ema: float = 0.0
        self._dual_eta: float = float(self.config.get('dual_eta', 1e-3))
        bounds = self.config.get('lambda_rate_bounds', (0.0, 5.0))
        self._lambda_min: float = float(bounds[0])
        self._lambda_max: float = float(bounds[1])
        self._lambda_rate_scale: float = float(self.config.get('lambda_rate_scale', 1.0))

    def _calculate_total_steps(self, config: Dict) -> int:
        """
        自动计算总训练步数
        优先级：
        1. 直接指定 total_steps
        2. num_epochs * steps_per_epoch
        3. num_epochs * (dataset_size / batch_size)
        4. 默认值 8000
        """
        # 优先级1：直接指定
        if 'total_steps' in config and config['total_steps'] is not None:
            return config['total_steps']

        # 优先级2：epochs * steps_per_epoch
        if 'num_epochs' in config and 'steps_per_epoch' in config:
            total_steps = config['num_epochs'] * config['steps_per_epoch']
            print(f"Auto-calculated total_steps: {config['num_epochs']} epochs × {config['steps_per_epoch']} steps = {total_steps}")
            return total_steps

        # 优先级3：epochs * (dataset_size / batch_size)
        if all(k in config for k in ['num_epochs', 'dataset_size', 'batch_size']):
            steps_per_epoch = max(1, config['dataset_size'] // config['batch_size'])
            total_steps = config['num_epochs'] * steps_per_epoch
            print(f"Auto-calculated total_steps: {config['num_epochs']} epochs × {steps_per_epoch} steps/epoch = {total_steps}")
            print(f"  (dataset_size={config['dataset_size']}, batch_size={config['batch_size']})")
            return total_steps

        # 优先级4：从训练器配置推断
        if 'num_epochs' in config:
            estimated_steps_per_epoch = config.get('estimated_steps_per_epoch', 1000)  # 默认估计
            total_steps = config['num_epochs'] * estimated_steps_per_epoch
            print(f"Estimated total_steps: {config['num_epochs']} epochs × {estimated_steps_per_epoch} steps/epoch = {total_steps}")
            print(f"  (using estimated_steps_per_epoch, may not be accurate)")
            return total_steps

        # 默认回退
        default_steps = 8000
        print(f"Using default total_steps: {default_steps} (no epoch/dataset info provided)")
        return default_steps

    def compute_comprehensive_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        current_step: int,
        model: Optional[torch.nn.Module] = None  # 新增：用于梯度感知权重调整
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算Stage5综合损失

        Args:
            model_outputs: 模型输出字典
            targets: 目标数据字典
            current_step: 当前训练步数
        """
        # P6修复：统一断言特征维度一致性
        if 'original_features' in targets:
            target_features = targets['original_features']
            assert target_features.dim() == 3, f"target_features must be 3D [B,T,D], got {target_features.shape}"
            assert target_features.shape[-1] == 36, f"target_features last dim must be 36, got {target_features.shape[-1]}"

        if 'reconstructed_features' in model_outputs:
            recon_features = model_outputs['reconstructed_features']
            assert recon_features.dim() == 3, f"reconstructed_features must be 3D [B,T,D], got {recon_features.shape}"
            assert recon_features.shape[-1] == 36, f"reconstructed_features last dim must be 36, got {recon_features.shape[-1]}"

        device = model_outputs['quantized_latent'].device

        # === 1. 获取动态权重 ===
        weights = self.weight_scheduler.get_weights(current_step)
        recon_only = bool(self.config.get('recon_only', False))
        if recon_only:
            weights.update({'feat': 1.0, 'wave': 1.0, 'semantic': 0.0, 'quality': 0.0, 'commitment': 0.0, 'rate': 0.0})

        # === 2. 基础重构损失（在raw空间度量） ===
        y_hat_feat = model_outputs.get('recon_features_raw', model_outputs['reconstructed_features'])
        y_ref_feat = targets['original_features']
        feat_recon_loss = F.l1_loss(y_hat_feat, y_ref_feat)

        # Whitened reconstruction (per-dim) to penalize scale mismatch explicitly（raw空间）
        eps_whiten = 1e-6
        try:
            mu_tgt = y_ref_feat.mean(dim=(0, 1))              # [D]
            std_tgt = y_ref_feat.std(dim=(0, 1))               # [D]
            y_hat_w = (y_hat_feat - mu_tgt) / (std_tgt + eps_whiten)
            y_ref_w = (y_ref_feat - mu_tgt) / (std_tgt + eps_whiten)
            feat_whiten_mse = torch.nn.functional.mse_loss(y_hat_w, y_ref_w)
        except Exception:
            feat_whiten_mse = torch.tensor(0.0, device=y_hat_feat.device)

        # Predicted log-std calibration (decoder three-head) – supervise typical per-dim scale
        pred_logstd = model_outputs.get('pred_logstd', None)
        if torch.is_tensor(pred_logstd):
            try:
                # Average predicted logstd over B,T to get a stable per-dim estimate
                pred_logstd_mean = pred_logstd.mean(dim=(0, 1))  # [D]
                logstd_target = (std_tgt + eps_whiten).log()     # [D]
                logstd_mse = torch.nn.functional.mse_loss(pred_logstd_mean, logstd_target)
            except Exception:
                logstd_mse = torch.tensor(0.0, device=y_hat_feat.device)
        else:
            logstd_mse = torch.tensor(0.0, device=y_hat_feat.device)

        # 记录特征重构损失用于语义学习控制
        self._last_feat_loss = feat_recon_loss.item()

        # === 3. 波形感知损失 ===
        synthesized = model_outputs.get('synthesized_audio')
        target = targets.get('target_audio')

        # 调试：检查音频数据（每50步）
        if current_step % 50 == 0:
            synth_status = f"shape={synthesized.shape}, range=[{synthesized.min():.3f}, {synthesized.max():.3f}]" if synthesized is not None else "None"
            target_status = f"shape={target.shape}, range=[{target.min():.3f}, {target.max():.3f}]" if target is not None else "None"
            print(f"[Wave Debug] Step {current_step}: Synth={synth_status}, Target={target_status}")

        if synthesized is not None and target is not None:
            # 修复音频格式匹配问题
            # 1. 维度对齐：[B, 1, L] -> [B, L]
            if len(synthesized.shape) == 3 and synthesized.shape[1] == 1:
                synthesized = synthesized.squeeze(1)

            # 2. 长度对齐：裁剪到相同长度
            min_len = min(synthesized.shape[-1], target.shape[-1])
            synthesized = synthesized[..., :min_len]
            target = target[..., :min_len]

            # 调试：验证gain修复效果
            if current_step % 50 == 0:
                synth_max = synthesized.abs().max()
                if synth_max > 2.0:
                    print(f"[Wave Warning] Step {current_step}: Audio amplitude still high: {synth_max:.3f}")
                else:
                    print(f"[Wave OK] Step {current_step}: Audio amplitude normal: {synth_max:.3f}")

            wave_loss, wave_details = fargan_wave_losses(
                synthesized, target, targets.get('period', None), device=device
            )
            # 额外：加入SI-SDR以增强对听感相关误差的敏感度
            def _si_sdr(pred, ref, eps=1e-8):
                # pred/ref: [B, L]
                ref_energy = (ref ** 2).sum(dim=-1, keepdim=True) + eps
                proj = ((pred * ref).sum(dim=-1, keepdim=True) / ref_energy) * ref
                e_noise = pred - proj
                sdr = (proj ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps)
                return 10 * torch.log10(sdr + eps)  # [B]

            if bool(self.config.get('enable_sisdr', True)):
                lambda_sisdr = float(self.config.get('lambda_sisdr', 0.5))
                sisdr = _si_sdr(synthesized, target)
                sisdr_loss = (-sisdr.mean())
                wave_loss = wave_loss + lambda_sisdr * sisdr_loss
                try:
                    wave_details = {**wave_details, 'sisdr_db': float(sisdr.mean().item())}
                except Exception:
                    pass

            # 额外：LSD(dB)观测与可选损失
            if bool(self.config.get('enable_lsd', True)):
                try:
                    n_fft = int(self.config.get('lsd_n_fft', 512))
                    hop = int(self.config.get('lsd_hop', 160))
                    win = int(self.config.get('lsd_win', 320))
                    window = torch.hann_window(win, device=device)
                    def _spec_db(x):
                        # x: [B,L]
                        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                                       window=window, return_complex=True)
                        mag = (X.abs() + 1e-8)
                        return 20.0 * torch.log10(mag)
                    Xdb = _spec_db(synthesized)
                    Ydb = _spec_db(target)
                    # LSD per frame: sqrt(mean_f (ΔdB^2)), then mean over time and batch
                    lsd_per_frame = torch.sqrt(((Xdb - Ydb) ** 2).mean(dim=1) + 1e-8)  # [B,T]
                    lsd_db = lsd_per_frame.mean()
                    # 可选加入到loss
                    lambda_lsd = float(self.config.get('lambda_lsd', 0.0))
                    if lambda_lsd > 0.0:
                        wave_loss = wave_loss + lambda_lsd * lsd_db
                    wave_details = {**wave_details, 'lsd_db': float(lsd_db.item())}
                except Exception:
                    pass
        else:
            wave_loss = torch.tensor(0.0, device=device)
            wave_details = {}

        # === 4. 有效码率损失（可微） ===============================================
        # 读取配置
        # 信息瓶颈总预算（VIB与Rate共享） - 必须在最早定义
        total_info_budget = self.config.get('total_info_regularization_budget', 1.0)
        beta_vib = float(self.config.get('beta_vib', 1e-3))
        vib_warmup = int(self.config.get('vib_warmup_steps', 3000))  # 修复3：拉长warmup，让RVQ先站稳
        sem_tau = float(self.config.get('semantic_temperature', 0.2))
        # 温度退火逻辑（按照用户建议）
        if self.config.get('semantic_temp_annealing', False):
            tau_end  = float(self.config.get('semantic_temp_end', 0.07))
            tau_steps = int(self.config.get('semantic_temp_steps', 3000))
            # 线性退火：从0.2 -> 0.07
            t = min(max(current_step, 0), tau_steps)
            if tau_steps > 0:
                sem_tau = sem_tau + (tau_end - sem_tau) * (t / float(tau_steps))
            else:
                # If tau_steps is 0, use final temperature immediately
                sem_tau = tau_end

        rate_type = self.config.get('rate_loss_type', 'soft_entropy')  # fixed to soft_entropy
        if rate_type != 'soft_entropy':
            rate_type = 'soft_entropy'
        lambda_rate = float(self.config.get('lambda_rate', 0.5))
        target_kbps = float(self.config.get('target_kbps', 1.2))
        tol_kbps = float(self.config.get('rate_tolerance', 0.1))
        frame_rate = float(self.config.get('frame_rate', 50.0))

        # 有效码率计算
        rate_loss = torch.tensor(0.0, device=device)
        eff_bits_per_frame = torch.tensor(0.0, device=device)
        nom_bits_per_frame = torch.tensor(0.0, device=device)

        if (not recon_only) and rate_type == 'soft_entropy' and ('rvq_soft_probs' in model_outputs):
            # Differentiable soft-entropy from soft assignments; supports gradient to encoder/codebook
            stage_sizes = model_outputs.get('rvq_stage_sizes', [1024, 512, 256])
            q_list = model_outputs.get('rvq_soft_probs', [])  # list of [B, T, K]
            gate_soft_list = model_outputs.get('stage_gate_soft', [])  # list of [B, T]

            eff_bpf = torch.tensor(0.0, device=device)

            for i, K in enumerate(stage_sizes):
                if i < len(q_list) and q_list[i] is not None and torch.is_tensor(q_list[i]):
                    q = q_list[i]  # [B, T, K]
                    # Per-frame entropy
                    ent_bT = (-(q * torch.log2(q.clamp_min(1e-12))).sum(dim=-1))  # [B,T]
                    # Apply soft gate if available
                    if i < len(gate_soft_list) and gate_soft_list[i] is not None and torch.is_tensor(gate_soft_list[i]):
                        ent_bT = ent_bT * gate_soft_list[i]
                    ent = ent_bT.mean()
                else:
                    # Fallback to nominal bits if soft probs not present
                    ent = torch.tensor(math.log2(K), device=device)

                eff_bpf = eff_bpf + ent

            eff_bits_per_frame = eff_bpf
            kbps_eff = eff_bits_per_frame * frame_rate / 1000.0

            # Dual-ascent closed-loop control: λ * (kbps_eff - target)
            try:
                err_inst = float((kbps_eff - target_kbps).detach().item())
            except Exception:
                err_inst = 0.0
            # EMA smoothing for rate error
            self._rate_ema = 0.95 * self._rate_ema + 0.05 * err_inst
            # Update λ with clipped EMA
            ema_clipped = max(min(self._rate_ema, 2.0), -2.0)
            self._lambda_rate += self._dual_eta * ema_clipped
            self._lambda_rate = max(min(self._lambda_rate, self._lambda_max), self._lambda_min)

            # Linear penalty keeps gradients to eff rate; detach λ
            lambda_rate_t = torch.tensor(self._lambda_rate, device=device).detach()
            rate_loss = self._lambda_rate_scale * lambda_rate_t * (kbps_eff - target_kbps)
            # For logging: nominal bits is full sum log2(K)
            nom_bits_per_frame = torch.tensor(sum([math.log2(K) for K in stage_sizes]), device=device)

        else:
            kbps_eff = torch.tensor(0.0, device=device)

        # === 5. 语义感知：支持 InfoNCE 与 MSE 两种模式 =============================
        semantic_mode = self.config.get('semantic_mode', 'mse')  # 'nce' or 'mse'

        if (not recon_only) and semantic_mode == 'nce' and ('student_sem' in model_outputs) and ('teacher_sem' in model_outputs):
            # 改进的门控：基于码本健康度而不是commitment_loss
            codebook_healthy = False
            feat_recon_stable = False

            # 1. 检查码本健康度（困惑度）
            rvq_details = model_outputs.get('rvq_details', {})
            if isinstance(rvq_details, dict) and rvq_details.get('stage_perplexities') is not None:
                perplexities = rvq_details.get('stage_perplexities')  # [num_stages] - 修复字段名
                codebook_sizes = model_outputs.get('rvq_stage_sizes', [512, 512, 512])  # 默认值

                # 修复：处理perplexities可能是list的情况
                if isinstance(perplexities, list):
                    if len(perplexities) > 0 and len(codebook_sizes) >= len(perplexities):
                        # 转换为tensor并检查困惑度
                        perp_tensor = torch.tensor(perplexities, device=device)
                        size_tensor = torch.tensor(codebook_sizes[:len(perplexities)], device=device)
                        perplexity_ratios = perp_tensor / size_tensor.float()
                        codebook_healthy = (perplexity_ratios > 0.1).all().item()  # 所有stage困惑度 > 10%
                elif torch.is_tensor(perplexities) and len(codebook_sizes) >= len(perplexities):
                    # 原有tensor处理逻辑
                    size_tensor = torch.tensor(codebook_sizes[:len(perplexities)], device=perplexities.device)
                    perplexity_ratios = perplexities / size_tensor.float()
                    codebook_healthy = (perplexity_ratios > 0.1).all().item()  # 所有stage困惑度 > 10%
                else:
                    # 备用：使用码率健康度判断
                    if 'effective_rate_bpf' in rvq_details and 'expected_rate_bpf' in rvq_details:
                        eff_rate = rvq_details['effective_rate_bpf']
                        exp_rate = rvq_details['expected_rate_bpf']
                        rate_ratio = eff_rate / exp_rate.clamp_min(0.1)  # 避免除零
                        codebook_healthy = 0.5 <= rate_ratio <= 2.0  # 码率在期望值的50%-200%范围内

            # 2. 检查特征重建稳定性
            if hasattr(self, '_last_feat_loss'):
                feat_recon_stable = self._last_feat_loss < 2.0  # 放宽阈值：2.0以下认为稳定
            else:
                feat_recon_stable = False

            # 3. 强制启用机制
            force_enable_step = int(self.config.get('semantic_force_enable_step', 1500))  # 1500步后强制启用

            # 调试：每50步检查一次条件
            if current_step % 50 == 0:
                feat_val = getattr(self, '_last_feat_loss', 'N/A')
                perp_info = ""
                if isinstance(rvq_details, dict) and rvq_details.get('stage_perplexities') is not None:
                    perp_data = rvq_details.get('stage_perplexities', [])
                    # 修复：安全处理perplexities的tolist()
                    if isinstance(perp_data, list):
                        perp_vals = perp_data
                    elif torch.is_tensor(perp_data):
                        perp_vals = perp_data.tolist()
                    else:
                        perp_vals = []
                    perp_info = f"perp={perp_vals}"
                print(f"[Semantic Debug] Step {current_step}: codebook_healthy={codebook_healthy}, "
                      f"feat_loss={feat_val}, feat_stable={feat_recon_stable}, {perp_info}")

            # 改进的门控条件：码本健康 + 特征稳定 或 强制启用
            if (codebook_healthy and feat_recon_stable) or current_step >= force_enable_step:
                # RVQ相对稳定，开始语义对比学习
                # 使用时间粒度NT-Xent增加负样本数量（按照用户建议）
                semantic_loss = nt_xent_loss(
                    F.normalize(model_outputs['student_sem'], dim=1),  # [B,D,T]
                    F.normalize(model_outputs['teacher_sem'], dim=1),
                    temperature=sem_tau
                )

                if current_step % 50 == 0:
                    print(f"[Semantic] Step {current_step}: ✅ NT-Xent ACTIVE, loss={semantic_loss:.4f}, τ={sem_tau:.3f}")
            else:
                # 码本不健康或特征不稳定，暂停语义学习
                semantic_loss = torch.tensor(0.0, device=device)

                if current_step % 50 == 0:
                    print(f"[Semantic] Step {current_step}: ❌ InfoNCE DISABLED (codebook_healthy={codebook_healthy}, feat_stable={feat_recon_stable})")

            # 可选：添加少量MSE辅助
            if self.config.get('add_mse_auxiliary', False) and 'semantic_features' in model_outputs and 'reference_semantic' in targets:
                semantic_target = targets['reference_semantic']
                if semantic_target.dim() == 3 and semantic_target.shape[1] != model_outputs['semantic_features'].shape[1]:
                    semantic_target = semantic_target.transpose(1, 2)
                pred_semantic = F.normalize(model_outputs['semantic_features'], p=2, dim=1)
                target_semantic = F.normalize(semantic_target, p=2, dim=1)
                mse_aux = F.mse_loss(pred_semantic, target_semantic)
                semantic_loss = semantic_loss + 0.1 * mse_aux

        elif 'semantic_features' in model_outputs and 'reference_semantic' in targets:
            # 回退：原始MSE语义损失
            semantic_target = targets['reference_semantic']
            if semantic_target.dim() == 3 and semantic_target.shape[1] != model_outputs['semantic_features'].shape[1]:
                semantic_target = semantic_target.transpose(1, 2)
            pred_semantic = F.normalize(model_outputs['semantic_features'], p=2, dim=1)
            target_semantic = F.normalize(semantic_target, p=2, dim=1)
            semantic_loss = F.mse_loss(pred_semantic, target_semantic)

        else:
            # 无语义数据可用
            semantic_loss = torch.tensor(0.0, device=device)

        # === 6. 时间一致性损失（移除，RVQ本身就是离散的）===
        temporal_loss = torch.tensor(0.0, device=device)  # 移除无意义的temporal loss

        # === 7. 真实PESQ质量损失 ===
        quality_loss = compute_real_pesq_loss(
            model_outputs=model_outputs,
            targets=targets,
            minimum_pesq_threshold=self.config.get('min_quality_threshold', 2.5),
            current_step=current_step,
            device=device
        )

        # === 8. RVQ相关损失（增强版）===
        commitment_loss = model_outputs.get('commitment_loss', torch.tensor(0.0, device=device))

        # 新增：RVQ重构一致性损失（确保编码-解码一致性）
        rvq_reconstruction_loss = torch.tensor(0.0, device=device)
        if model_outputs.get('compression_ready', False) and 'rvq_details' in model_outputs:
            rvq_details = model_outputs['rvq_details']
            stage_indices = rvq_details.get('stage_indices')

            if stage_indices is not None and 'quantized_latent' in model_outputs:
                # 计算RVQ重构误差（模拟解码器的重构精度）
                try:
                    # 这里需要访问模型的RVQ解码器
                    # 在实际使用中，这个计算应该在模型前向传播中完成
                    original_latent = model_outputs.get('encoded_latent')  # [B, T, D]
                    if original_latent is not None:
                        # 转换为[B, D, T]格式进行RVQ重构误差计算
                        original_bct = original_latent.transpose(1, 2)
                        quantized_bct = model_outputs['quantized_latent'].transpose(1, 2)
                        rvq_reconstruction_loss = F.mse_loss(quantized_bct, original_bct)
                except Exception:
                    # 如果计算失败，使用默认值
                    pass

        # === 9. 可选：对比学习损失 ===
        contrastive_loss = torch.tensor(0.0, device=device)
        if hasattr(targets, 'contrastive_pairs'):
            # 实现对比学习(可选)
            pass

        # === 信息瓶颈错峰调度配置 ===================================
        vib_phase_end = self.config.get('vib_phase_end_step', 3000)  # VIB主导期结束

        # === 6. VIB-KL (错峰调度版) ===============================================

        vib_kld = model_outputs.get('vib_kld', None)
        # Optional: PI-controlled scaling from rate controller
        vib_scale = 1.0
        try:
            if model is not None and hasattr(model, 'rate_controller'):
                knobs = model.rate_controller.get_last_controls()
                vib_scale = float(knobs.get('vib_beta_scale', 1.0))
        except Exception:
            pass
        if vib_kld is not None:
            # VIB权重：早期高，后期逐渐降低为Rate让路
            if current_step <= vib_warmup:
                vib_weight = beta_vib * float(current_step) / max(1, vib_warmup)
            elif current_step <= vib_phase_end:
                vib_weight = beta_vib  # 保持最大值
            else:
                # 后期指数衰减，为rate让路
                decay_steps = current_step - vib_phase_end
                vib_weight = beta_vib * (0.5 ** (decay_steps / 1000))  # 每1000步衰减一半

            # 应用总预算约束（VIB与Rate错峰）
            vib_budget_ratio = 0.3  # VIB占总预算30%
            vib_loss = vib_scale * vib_weight * total_info_budget * vib_budget_ratio * vib_kld
        else:
            vib_loss = torch.tensor(0.0, device=device)

        # === 9. RVQ Diversity Loss (码书使用熵最大化) ==========================
        diversity_loss = torch.tensor(0.0, device=device)
        if (not recon_only) and self.config.get('enable_rvq_diversity', True):  # 默认开启
            diversity_weight = self.config.get('rvq_diversity_weight', 2e-3)  # 小权重
            flip_step = int(self.config.get('diversity_flip_step', 2000))
            # Early: encourage balanced usage; Late: discourage entropy to lower rate
            direction = 1.0 if current_step < flip_step else -1.0
            rvq_details = model_outputs.get('rvq_details', {})

            if rvq_details.get('stage_perplexities') is not None:
                perplexities = rvq_details.get('stage_perplexities')  # list of tensors or list
                stage_sizes = model_outputs.get('rvq_stage_sizes', [1024, 512, 256])

                # 修复：安全处理perplexities类型
                if isinstance(perplexities, list):
                    # 如果是Python list，转换为tensor
                    if len(perplexities) > 0:
                        perp_tensor = torch.tensor(perplexities, device=device)
                        for i, K in enumerate(stage_sizes[:len(perplexities)]):
                            perp = perp_tensor[i]
                            # 鼓励高困惑度（接近最大值log2(K)）
                            max_perp = math.log2(K)
                            perp_ratio = perp / max_perp  # 归一化到[0,1]
                        # 负熵损失：方向可切换（早期鼓励高困惑度，后期抑制熵）
                        diversity_loss += direction * (-torch.log(perp_ratio.clamp_min(1e-8)))
                else:
                    # 原有tensor处理逻辑
                    for perp, K in zip(perplexities, stage_sizes):
                        if torch.is_tensor(perp) and perp.numel() > 0:
                            # 鼓励高困惑度（接近最大值log2(K)）
                            max_perp = math.log2(K)
                            perp_ratio = perp / max_perp  # 归一化到[0,1]
                            # 负熵损失：方向可切换
                            diversity_loss += direction * (-torch.log(perp_ratio.clamp_min(1e-8)))

                diversity_loss *= diversity_weight

            # === 添加显式熵奖励 ===
            if rvq_details.get('stage_indices') is not None:
                for stage_idx, indices in enumerate(rvq_details.get('stage_indices')):
                    # 计算符号使用熵
                    flat_indices = indices.reshape(-1)
                    K = model_outputs.get('rvq_stage_sizes', [1024, 512, 256])[stage_idx]
                    counts = torch.bincount(flat_indices, minlength=K).float()
                    probs = counts / counts.sum()
                    H_usage = -(probs * torch.log2(probs + 1e-12)).sum()

                    # 熵奖励/惩罚：方向可切换
                    diversity_loss += (-direction) * diversity_weight * H_usage

        # === Codebook mask sparsity (effective-K control) ===
        mask_sparsity_penalty = torch.tensor(0.0, device=device)
        if (not recon_only) and 'codebook_mask_usage' in model_outputs:
            usage_list = model_outputs['codebook_mask_usage']  # list of scalars
            if isinstance(usage_list, list) and len(usage_list) > 0:
                mask_mean = torch.stack([u if torch.is_tensor(u) else torch.tensor(float(u), device=device) for u in usage_list]).mean()
                base_w = float(self.config.get('mask_sparsity_weight', 0.0))
                # Optional PI boost from controller
                try:
                    if model is not None and hasattr(model, 'rate_controller'):
                        knobs = model.rate_controller.get_last_controls()
                        base_w = base_w + float(knobs.get('mask_sparsity_boost', 0.0))
                except Exception:
                    pass
                # If rate is under target - tol, disable mask sparsity entirely
                try:
                    if 'kbps_eff' in locals():
                        if float(kbps_eff.item()) < (target_kbps - tol_kbps):
                            base_w = 0.0
                except Exception:
                    pass
                mask_sparsity_penalty = base_w * mask_mean

        # === Stage-gate sparsity (frame-level L0 proxy on higher stages) ===
        gate_l0_penalty = torch.tensor(0.0, device=device)
        try:
            gate_list = model_outputs.get('stage_gate_soft', [])  # list of [B,T]
            if (not recon_only) and isinstance(gate_list, list) and len(gate_list) > 1:
                # Exclude stage 0; penalize average open probability of s>=1
                mean_gates = []
                for i, g in enumerate(gate_list):
                    if i == 0 or (not torch.is_tensor(g)):
                        continue
                    mean_gates.append(g.mean())
                if len(mean_gates) > 0:
                    mean_gate_val = torch.stack(mean_gates).mean()
                    base = float(self.config.get('lambda_gate_l0', 0.0))
                    over = 0.0
                    try:
                        over = max(float((kbps_eff - target_kbps).detach().item()), 0.0) / max(target_kbps, 1e-6)
                    except Exception:
                        over = 0.0
                    gain = float(self.config.get('lambda_gate_over_gain', 0.5))
                    w_gate = base + gain * over
                    gate_l0_penalty = w_gate * mean_gate_val
        except Exception:
            gate_l0_penalty = torch.tensor(0.0, device=device)

        # === Stage-gate Concrete-Bernoulli KL prior (encourage open/close by rate state) ===
        gate_kl_penalty = torch.tensor(0.0, device=device)
        try:
            gate_list = model_outputs.get('stage_gate_soft', [])
            if (not recon_only) and isinstance(gate_list, list) and len(gate_list) > 1:
                # Choose prior pi based on rate condition
                pi_under = float(self.config.get('gate_prior_pi_under', 0.8))
                pi_mid   = float(self.config.get('gate_prior_pi_mid', 0.5))
                pi_over  = float(self.config.get('gate_prior_pi_over', 0.3))
                try:
                    kbps_val = float(kbps_eff.item()) if torch.is_tensor(kbps_eff) else float(kbps_eff)
                except Exception:
                    kbps_val = target_kbps
                if kbps_val < (target_kbps - tol_kbps):
                    pi = pi_under
                elif kbps_val > (target_kbps + tol_kbps):
                    pi = pi_over
                else:
                    pi = pi_mid
                # KL(q||Bernoulli(pi)) averaged over frames, stages>=1
                kl_list = []
                pi_t = torch.tensor(pi, device=device).clamp(1e-4, 1-1e-4)
                for i, g in enumerate(gate_list):
                    if i == 0 or (not torch.is_tensor(g)):
                        continue
                    q = g.clamp(1e-6, 1-1e-6)
                    kl = (q * (q/pi_t).log() + (1-q) * ((1-q)/(1-pi_t)).log()).mean()
                    kl_list.append(kl)
                if kl_list:
                    gate_kl = torch.stack(kl_list).mean()
                    gate_kl_penalty = float(self.config.get('lambda_gate_kl', 0.05)) * gate_kl
        except Exception:
            gate_kl_penalty = torch.tensor(0.0, device=device)

        # === 9. 阶段感知的多任务权重调整 ==========================================
        if self.is_gradient_aware and model is not None:
            # 收集所有损失项（包括新增的VIB和rate loss）
            individual_losses = {
                'feat': feat_recon_loss,
                'wave': wave_loss,
                'semantic': semantic_loss,
                'quality': quality_loss,
                'commitment': commitment_loss,
                'rate': rate_loss
            }

            # 阶段感知权重调整：结合阶段策略和梯度平衡
            try:
                updated_weights = self._compute_stage_aware_weights(
                    model, individual_losses, weights, current_step
                )
                # 使用更新后的权重
                old_weights = weights.copy()
                weights.update(updated_weights)

                # 每50步打印权重变化
                if current_step % 50 == 0:
                    print(f"Multi-task gradient balancing (step {current_step}):")
                    for key in ['feat', 'wave', 'semantic', 'quality', 'commitment', 'rate']:
                        old_w = old_weights.get(key, 0)
                        new_w = weights.get(key, 0)
                        change = new_w - old_w
                        print(f"  {key}: {old_w:.4f} -> {new_w:.4f} ({change:+.4f})")

            except Exception as e:
                # 如果梯度计算失败，使用原权重
                print(f"Warning: Multi-task gradient balancing failed: {e}")

        # === 10. 综合损失汇总 ====================================================
        # 修复A: 去除VIB双重缩放 - vib_loss已包含beta_vib*warmup，直接相加
        # 可选：输出端 ceps 仿射正则（解码器暴露在debug_stats中）
        calib_reg = torch.tensor(0.0, device=device)
        if isinstance(model_outputs, dict):
            dbg = model_outputs.get('debug_stats', {}) if isinstance(model_outputs, dict) else {}
            if isinstance(dbg, dict) and 'calib' in dbg:
                # 仅用于日志展示；真实的正则从decoder读取较合适，但此处保底值 0.0
                pass
        # 从模型读取更可靠（若可用）
        try:
            if (model is not None and hasattr(model, 'decoder') and
                hasattr(model.decoder, '_last_calib_reg') and model.decoder._last_calib_reg is not None):
                calib_reg = model.decoder._last_calib_reg.to(device=device)
        except Exception:
            pass
        lambda_calib = float(self.config.get('lambda_ceps_calib_reg', 1e-4))

        # Variance band regularizer + temporal difference matching
        stat_band_reg = torch.tensor(0.0, device=device)
        tv_loss = torch.tensor(0.0, device=device)
        lambda_stat = float(self.config.get('lambda_stat_reg', 0.02))
        lambda_tv = float(self.config.get('lambda_tv', 0.02))
        r_lo = float(self.config.get('stat_ratio_lo', 0.6))
        r_hi = float(self.config.get('stat_ratio_hi', 1.4))
        # Mean alignment + domain constraints (frame_corr range)
        mean_match_reg = torch.tensor(0.0, device=device)
        fc_range_penalty = torch.tensor(0.0, device=device)
        lambda_mean = float(self.config.get('lambda_mean_reg', 0.02))
        lambda_fc_range = float(self.config.get('lambda_fc_range', 0.02))
        try:
            from models.feature_adapter import FARGANFeatureSpec
            sl_ceps = FARGANFeatureSpec.get_feature_slice('ceps')
            sl_lpc = FARGANFeatureSpec.get_feature_slice('lpc')
            sl_fc = FARGANFeatureSpec.get_feature_slice('frame_corr')
            y_hat = y_hat_feat  # 使用raw空间统计
            y_ref = y_ref_feat
            def _band_penalty(a, b):
                std_a = a.std(dim=(0, 1))
                std_b = b.std(dim=(0, 1))
                ratio = (std_a + 1e-6) / (std_b + 1e-6)
                low = torch.relu(torch.tensor(r_lo, device=a.device) - ratio)
                high = torch.relu(ratio - torch.tensor(r_hi, device=a.device))
                return (low.pow(2) + high.pow(2)).mean()
            stat_band_reg = _band_penalty(y_hat[..., sl_ceps], y_ref[..., sl_ceps])
            stat_band_reg = stat_band_reg + _band_penalty(y_hat[..., sl_lpc], y_ref[..., sl_lpc])
            stat_band_reg = stat_band_reg + _band_penalty(y_hat[..., sl_fc], y_ref[..., sl_fc])
            # Mean alignment (block-wise)
            def _mean_l2(a, b):
                mu_a = a.mean(dim=(0, 1))
                mu_b = b.mean(dim=(0, 1))
                return torch.nn.functional.mse_loss(mu_a, mu_b)
            mean_match_reg = _mean_l2(y_hat[..., sl_ceps], y_ref[..., sl_ceps])
            mean_match_reg = mean_match_reg + _mean_l2(y_hat[..., sl_lpc], y_ref[..., sl_lpc])
            mean_match_reg = mean_match_reg + _mean_l2(y_hat[..., sl_fc], y_ref[..., sl_fc])
            # FrameCorr domain: encourage |fc| <= 0.5 (target range)
            fc_hat = y_hat[..., sl_fc]
            over = torch.relu(fc_hat.abs() - 0.5)
            fc_range_penalty = (over.pow(2)).mean()
            # Temporal difference matching over all dims
            if y_hat.size(1) > 1 and y_ref.size(1) > 1:
                dy_hat = y_hat[:, 1:, :] - y_hat[:, :-1, :]
                dy_ref = y_ref[:, 1:, :] - y_ref[:, :-1, :]
                tv_loss = torch.nn.functional.mse_loss(dy_hat, dy_ref)
        except Exception:
            stat_band_reg = torch.tensor(0.0, device=device)
            tv_loss = torch.tensor(0.0, device=device)
            mean_match_reg = torch.tensor(0.0, device=device)
            fc_range_penalty = torch.tensor(0.0, device=device)

        lambda_whiten = float(self.config.get('lambda_whiten', 0.5))
        lambda_logstd = float(self.config.get('lambda_logstd', 0.2))

        total_loss = (weights['feat'] * feat_recon_loss
                      + weights['wave'] * wave_loss
                      + weights['semantic'] * semantic_loss
                      + weights['quality'] * quality_loss
                      + weights['commitment'] * commitment_loss
                      + vib_loss  # 直接相加，不再乘weights['vib']
                      + rate_loss
                      + lambda_calib * calib_reg
                      + diversity_loss
                      + mask_sparsity_penalty
                      + lambda_whiten * feat_whiten_mse
                      + lambda_logstd * logstd_mse
                      + lambda_stat * stat_band_reg
                      + lambda_tv * tv_loss
                      + lambda_mean * mean_match_reg
                      + lambda_fc_range * fc_range_penalty
                      + gate_l0_penalty
                      + gate_kl_penalty)

        # === 11. 额外的稳定性约束 ===
        if self.config.get('enable_stability_loss', True):
            stability_loss = self.compute_stability_loss(model_outputs, current_step)
            total_loss += 0.1 * stability_loss
        else:
            stability_loss = torch.tensor(0.0, device=device)

        # === 12. 损失详情（简化版）===
        loss_details = {
            'total': total_loss.item(),
            # 综合特征误差（更贴近听感）：L1(raw) + 辅助项
            'feat': float((feat_recon_loss
                           + lambda_whiten * (feat_whiten_mse if torch.is_tensor(feat_whiten_mse) else 0.0)
                           + lambda_logstd * (logstd_mse if torch.is_tensor(logstd_mse) else 0.0)
                           + lambda_stat   * (stat_band_reg if torch.is_tensor(stat_band_reg) else 0.0)
                           + lambda_tv     * (tv_loss if torch.is_tensor(tv_loss) else 0.0)).item() if torch.is_tensor(feat_recon_loss) else 0.0),
            'feat_recon': feat_recon_loss.item(),
            'feat_whiten_mse': feat_whiten_mse.item() if torch.is_tensor(feat_whiten_mse) else 0.0,
            'logstd_mse': logstd_mse.item() if torch.is_tensor(logstd_mse) else 0.0,
            'stat_band': stat_band_reg.item() if torch.is_tensor(stat_band_reg) else 0.0,
            'tv_loss': tv_loss.item() if torch.is_tensor(tv_loss) else 0.0,
            'mean_match': mean_match_reg.item() if torch.is_tensor(mean_match_reg) else 0.0,
            'fc_range': fc_range_penalty.item() if torch.is_tensor(fc_range_penalty) else 0.0,
            'gate_l0': float(gate_l0_penalty.item()) if torch.is_tensor(gate_l0_penalty) else 0.0,
            'gate_kl': float(gate_kl_penalty.item()) if torch.is_tensor(gate_kl_penalty) else 0.0,
            'wave': wave_loss.item(),
            'wave_mrstft': float(wave_details.get('mrstft', 0.0)) if isinstance(wave_details, dict) else 0.0,
            'wave_l1': float(wave_details.get('l1', 0.0)) if isinstance(wave_details, dict) else 0.0,
            'sisdr_db': float(wave_details.get('sisdr_db', 0.0)) if isinstance(wave_details, dict) else 0.0,
            'semantic': semantic_loss.item(),
            'semantic_loss': semantic_loss.item(),  # 兼容性
            'semantic_mode': semantic_mode,
            'quality': quality_loss.item(),
            'commitment': commitment_loss.item(),
            'vib_kld': float(vib_kld.item()) if vib_kld is not None else 0.0,
            'vib_beta': beta_vib,
            'diversity': diversity_loss.item(),
            'calib_reg': float(calib_reg.item()) if torch.is_tensor(calib_reg) else 0.0,
            'mask_sparsity': float(mask_sparsity_penalty.item()) if torch.is_tensor(mask_sparsity_penalty) else 0.0,
            'eff_bits_per_frame': float(eff_bits_per_frame.item()),
            'rate_bits_per_frame': float(eff_bits_per_frame.item()),  # 修复：供日志使用
            'nom_bits_per_frame': float(nom_bits_per_frame.item()),
            'kbps_eff': float(kbps_eff.item()) if isinstance(kbps_eff, torch.Tensor) else float(kbps_eff),
            'rate_loss': float(rate_loss.item()),
            'current_kbps': float(kbps_eff) if 'kbps_eff' in locals() else target_kbps,
            'lambda_rate': float(self._lambda_rate),
            'loss_weights': weights,
            # 新增：RVQ系统状态
            'compression_ready': model_outputs.get('compression_ready', False),
            'using_real_rate': model_outputs.get('compression_ready', False),
            # 新增：多任务梯度平衡状态
            'gradient_balancing_active': self.is_gradient_aware,
            'weight_adaptation_rate': getattr(self.weight_scheduler, 'adaptation_rate', 0.0),
            'gradient_balance_alpha': self.config.get('gradient_balance_alpha', 0.16),
            'initial_loss_ratios_set': hasattr(self, '_initial_loss_ratios')
        }

        # 更新损失历史
        self.update_loss_history(loss_details)

        return total_loss, loss_details

    def _compute_gradient_balanced_weights(
        self,
        model: torch.nn.Module,
        individual_losses: Dict[str, torch.Tensor],
        current_weights: Dict[str, float],
        current_step: int
    ) -> Dict[str, float]:
        """
        基于Stop Gradient的高效多任务权重平衡

        优势：
        1. 不需要计算梯度，避免干扰主训练过程
        2. 计算开销小，基于损失值变化趋势
        3. 数值稳定，不依赖梯度范数
        """
        # 配置参数
        adaptation_rate = self.config.get('gradient_adaptation_rate', 0.01)
        smoothing_factor = 0.9  # EMA平滑因子

        # 只对有效损失进行平衡 - 包括小数值的VIB和rate loss
        valid_losses = {}
        for k, v in individual_losses.items():
            if v is not None:
                # 对VIB和rate允许更小的阈值（因为它们的loss scale较小）
                if k in ['vib', 'rate']:
                    if v.item() > 1e-12:  # 更小的阈值
                        valid_losses[k] = v
                else:
                    if v > 1e-8:  # 其他loss的正常阈值
                        valid_losses[k] = v

        if len(valid_losses) < 2:
            return current_weights

        # 使用 .detach() 阻止梯度传播，只观察损失值
        current_loss_values = {k: v.detach().item() for k, v in valid_losses.items()}

        # 初始化损失历史记录
        if not hasattr(self, '_loss_history'):
            self._loss_history = {}
            self._loss_ema = {}

        # 更新损失EMA (指数移动平均)
        for task_name, loss_val in current_loss_values.items():
            if task_name not in self._loss_ema:
                self._loss_ema[task_name] = loss_val
                self._loss_history[task_name] = []
            else:
                # EMA更新
                self._loss_ema[task_name] = (smoothing_factor * self._loss_ema[task_name] +
                                            (1 - smoothing_factor) * loss_val)

            # 保留最近的损失历史（用于趋势分析）
            self._loss_history[task_name].append(loss_val)
            if len(self._loss_history[task_name]) > 10:  # 只保留最近10步
                self._loss_history[task_name].pop(0)

        # 如果历史数据不够，返回当前权重
        if current_step < 20:  # 前20步不调整
            return current_weights

        # 计算损失平衡因子
        updated_weights = current_weights.copy()

        # 确保rate权重被包含（如果它在valid_losses中但不在current_weights中）
        for task_name in valid_losses.keys():
            if task_name not in updated_weights:
                if task_name == 'rate':
                    updated_weights[task_name] = 0.5     # Rate的初始权重
                else:
                    updated_weights[task_name] = 0.1     # 其他loss的默认权重

        # 计算平均损失EMA用于归一化
        avg_loss_ema = sum(self._loss_ema.values()) / len(self._loss_ema)

        for task_name in valid_losses.keys():
            if task_name not in updated_weights:
                continue  # 跳过未初始化的权重

            # 当前任务的相对损失大小
            task_loss_ema = self._loss_ema[task_name]
            relative_loss = task_loss_ema / (avg_loss_ema + 1e-8)

            # 损失趋势：最近3步的平均 vs 前面3步的平均
            if len(self._loss_history[task_name]) >= 6:
                recent_avg = sum(self._loss_history[task_name][-3:]) / 3
                earlier_avg = sum(self._loss_history[task_name][-6:-3]) / 3
                trend = (recent_avg - earlier_avg) / (earlier_avg + 1e-8)
            else:
                trend = 0.0

            # 权重调整逻辑：
            # 1. 如果损失相对较大且还在上升，增加权重
            # 2. 如果损失相对较小且在下降，减少权重
            current_w = updated_weights[task_name]

            # 基于相对损失大小的调整
            if relative_loss > 1.5:  # 损失明显大于平均值
                weight_factor = 1.0 + adaptation_rate
            elif relative_loss < 0.7:  # 损失明显小于平均值
                weight_factor = 1.0 - adaptation_rate * 0.5
            else:
                weight_factor = 1.0

            # 基于趋势的微调
            if abs(trend) > 0.1:  # 趋势明显
                if trend > 0:  # 损失上升，增加权重
                    weight_factor *= (1.0 + adaptation_rate * 0.5)
                else:  # 损失下降，略减权重
                    weight_factor *= (1.0 - adaptation_rate * 0.3)

            # 平滑更新权重
            new_weight = current_w * weight_factor

            # 权重约束
            new_weight = max(0.01, min(5.0, new_weight))
            updated_weights[task_name] = new_weight

        # 权重归一化（保持总和稳定） - 修复：只对原有权重进行归一化
        # 分离原有权重和新增权重
        original_keys = {'feat', 'wave', 'semantic', 'quality', 'commitment'}
        new_keys = {'rate'}

        original_weights = {k: v for k, v in updated_weights.items() if k in original_keys}
        new_weights = {k: v for k, v in updated_weights.items() if k in new_keys}

        # 只对原有权重进行归一化
        original_total = sum(original_weights.values())
        target_original = sum(current_weights[k] for k in original_keys if k in current_weights)

        # 调试：检查归一化前后的权重
        if current_step % 100 == 0:
            print(f"Before normalization: rate={new_weights.get('rate', 0):.6f}")
            print(f"Original weights total: {original_total:.4f} -> target: {target_original:.4f}")

        if original_total > 0 and target_original > 0:
            norm_factor = target_original / original_total
            for k in original_keys:
                if k in updated_weights:
                    updated_weights[k] *= norm_factor

        # 新权重不归一化，保持动态调整的效果
        if current_step % 100 == 0:
            print(f"After normalization: rate={updated_weights.get('rate', 0):.6f} (not normalized)")

        # 日志记录（每100步）
        if current_step % 100 == 0:
            print("\n=== Multi-task Loss-based Balancing (Stop Gradient) ===")
            for task_name in valid_losses.keys():
                loss_ema = self._loss_ema.get(task_name, 0)
                relative = loss_ema / (avg_loss_ema + 1e-8)
                trend = 0.0
                if len(self._loss_history.get(task_name, [])) >= 6:
                    recent = sum(self._loss_history[task_name][-3:]) / 3
                    earlier = sum(self._loss_history[task_name][-6:-3]) / 3
                    trend = (recent - earlier) / (earlier + 1e-8)

                print(f"  {task_name}: loss_ema={loss_ema:.3f} relative={relative:.2f} trend={trend:+.3f}")

        return updated_weights

    def _compute_stage_aware_weights(
        self,
        model: torch.nn.Module,
        individual_losses: Dict[str, torch.Tensor],
        current_weights: Dict[str, float],
        current_step: int
    ) -> Dict[str, float]:
        """
        阶段感知的权重调整：结合训练阶段策略和梯度平衡

        策略：
        1. 获取当前阶段的目标权重（来自模型的get_dynamic_loss_weights）
        2. 使用梯度平衡进行微调，但限制在阶段权重范围内
        3. 确保不违背阶段性训练策略
        """
        # 1. 获取阶段目标权重
        if hasattr(model, 'get_dynamic_loss_weights'):
            stage_weights = model.get_dynamic_loss_weights(current_step)
        else:
            stage_weights = current_weights

        # 2. 基于阶段权重的梯度平衡微调
        balanced_weights = self._compute_gradient_balanced_weights(
            model, individual_losses, stage_weights, current_step
        )

        # 3. 约束调整幅度，避免偏离阶段策略太远
        final_weights = {}

        # 合并所有需要处理的权重键（current_weights + stage_weights）
        all_keys = set(current_weights.keys()) | set(stage_weights.keys())

        for key in all_keys:
            current_value = current_weights.get(key, 0.0)
            stage_target = stage_weights.get(key, current_value)
            balanced_value = balanced_weights.get(key, stage_target)

            # 限制偏离阶段目标的幅度，为语义损失提供特殊保护
            if key == 'semantic':
                # 语义损失的权重不允许被降低太多，防止语义学习崩溃
                max_deviation = 0.3  # 语义权重最大偏离30%
                min_weight = stage_target * 0.8  # 语义权重不能低于阶段目标的80%
                max_weight = stage_target * (1 + max_deviation)
            elif key == 'wave':
                # wave权重不允许过度增加，防止抢夺其他任务的学习
                max_deviation = 0.4  # wave权重最大偏离40%
                min_weight = stage_target * (1 - max_deviation)
                max_weight = stage_target * 1.3  # wave权重不能超过阶段目标的130%
            else:
                # 其他权重的标准约束
                max_deviation = 0.5  # 最大偏离50%
                min_weight = stage_target * (1 - max_deviation)
                max_weight = stage_target * (1 + max_deviation)

            final_weights[key] = max(min_weight, min(max_weight, balanced_value))

        # 4. 调试输出（每100步）
        if current_step % 100 == 0:
            print(f"\n=== Stage-Aware Weight Adjustment (Step {current_step}) ===")
            for key in ['feat', 'wave', 'semantic', 'commitment', 'rate', 'quality']:
                if key in stage_weights and key in final_weights:
                    stage_w = stage_weights[key]
                    final_w = final_weights[key]
                    print(f"  {key}: stage={stage_w:.3f} → final={final_w:.3f}")

        return final_weights

    def compute_stability_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        current_step: int
    ) -> torch.Tensor:
        """计算训练稳定性损失"""
        device = model_outputs['quantized_latent'].device

        # 1. RVQ perplexity稳定性
        if 'rvq_details' in model_outputs:
            rvq_details = model_outputs['rvq_details']
            if rvq_details.get('stage_perplexities') is not None:
                perplexities_data = rvq_details.get('stage_perplexities')

                # 修复：安全处理perplexities类型
                if isinstance(perplexities_data, list):
                    if len(perplexities_data) > 0:
                        perplexities = torch.tensor(perplexities_data, device=device)
                    else:
                        perplexities = []
                elif torch.is_tensor(perplexities_data):
                    perplexities = perplexities_data
                else:
                    perplexities = []

                if len(perplexities) > 0:
                    # 期望每个阶段都有合理的perplexity(避免codebook collapse)
                    target_perplexities = torch.tensor([64.0, 32.0, 16.0], device=device)[:len(perplexities)]
                    perplexity_loss = sum([
                        F.mse_loss(p.unsqueeze(0), target.unsqueeze(0))
                        for p, target in zip(perplexities, target_perplexities)
                    ]) / len(perplexities)
                else:
                    perplexity_loss = torch.tensor(0.0, device=device)
            else:
                perplexity_loss = torch.tensor(0.0, device=device)
        else:
            perplexity_loss = torch.tensor(0.0, device=device)

        # 2. 码率方差约束(防止码率震荡)
        rate_stats = model_outputs.get('rate_stats', {})
        if 'std_kbps' in rate_stats and rate_stats['std_kbps'] > 0:
            rate_variance_penalty = torch.tensor(rate_stats['std_kbps'], device=device).clamp(min=0, max=1.0)
        else:
            rate_variance_penalty = torch.tensor(0.0, device=device)

        return perplexity_loss + 0.1 * rate_variance_penalty

    def update_loss_history(self, loss_details: Dict[str, float]):
        """更新损失历史用于监控"""
        self.loss_history['rate'].append(loss_details['current_kbps'])
        self.loss_history['quality'].append(loss_details['wave'])
        self.loss_history['stability'].append(loss_details['total'])

        # 保持历史长度
        max_history = 100
        for key in self.loss_history:
            if len(self.loss_history[key]) > max_history:
                self.loss_history[key] = self.loss_history[key][-max_history:]

    def get_training_diagnostics(self) -> Dict[str, float]:
        """获取训练诊断信息"""
        if not self.loss_history['rate']:
            return {}

        import numpy as np

        recent_rates = self.loss_history['rate'][-20:]  # 最近20步
        recent_quality = self.loss_history['quality'][-20:]
        recent_stability = self.loss_history['stability'][-20:]

        return {
            'avg_rate_kbps': np.mean(recent_rates),
            'rate_std': np.std(recent_rates),
            'rate_in_range': np.mean([1.1 <= r <= 1.3 for r in recent_rates]),
            'avg_quality': np.mean(recent_quality),
            'stability_trend': np.mean(np.diff(recent_stability)) if len(recent_stability) > 1 else 0.0
        }

# === 便利函数 ===

def create_stage5_loss_computer(config: Dict) -> Stage5ComprehensiveLoss:
    """创建Stage5损失计算器"""
    return Stage5ComprehensiveLoss(config)


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试数据
    batch_size, feature_dim, seq_len = 2, 24, 100

    model_outputs = {
        'quantized_features': torch.randn(batch_size, feature_dim, seq_len, device=device),
        'reconstructed_features': torch.randn(batch_size, 36, seq_len, device=device),
        'synthesized_audio': torch.randn(batch_size, 1, seq_len * 160, device=device),
        'semantic_features': torch.randn(batch_size, 36, seq_len, device=device),
        'quality_prediction': torch.randn(batch_size, 3, device=device),
        'rate_bits_per_frame': torch.tensor(24.0, device=device),
        'rate_loss': torch.tensor(0.1, device=device),
        'commitment_loss': torch.tensor(0.05, device=device),
        'rate_stats': {'mean_kbps': 1.2, 'std_kbps': 0.05, 'in_range_ratio': 0.95}
    }

    targets = {
        'original_features': torch.randn(batch_size, 36, seq_len, device=device),
        'target_audio': torch.randn(batch_size, 1, seq_len * 160, device=device),
        'period': torch.randint(20, 200, (batch_size, seq_len), device=device)
    }

    # 测试损失计算
    config = {
        'total_steps': 8000,
        'rate_warmup_steps': 500,
        'frame_rate': 50,
        'temporal_smoothness': 0.1,
        'max_jump_threshold': 2.0,
        'min_quality_threshold': 2.5,
        'enable_stability_loss': True
    }

    loss_computer = create_stage5_loss_computer(config)

    for step in [100, 1500, 4000]:  # 测试不同阶段
        total_loss, details = loss_computer.compute_comprehensive_loss(
            model_outputs, targets, step
        )

        print(f"\nStep {step}:")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Current kbps: {details['current_kbps']:.3f}")
        print(f"  Rate warmup factor: {details['rate_warmup_factor']:.3f}")
        print(f"  Loss weights: feat={details['loss_weights']['feat']:.2f}, "
              f"wave={details['loss_weights']['wave']:.2f}, "
              f"rate={details['loss_weights']['rate']:.2f}")

    # 测试训练诊断
    print(f"\nTraining diagnostics:")
    diagnostics = loss_computer.get_training_diagnostics()
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.4f}")
