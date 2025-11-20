#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标损失平衡器 - 解决wave_loss和feat_loss冲突问题

核心设计理念：
1. 动态权重调整：根据各损失的相对大小和变化趋势调整权重
2. 梯度平衡：确保不同损失的梯度范数在合理范围内
3. 损失尺度归一化：避免不同损失尺度差异造成的训练不稳定
4. 自适应学习率：为不同目标设置差异化学习率
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import math


class MultiObjectiveBalancer:
    """多目标损失平衡器"""

    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        gradient_clip_norm: float = 1.0,
        history_length: int = 100,
        min_weight: float = 0.01,
        max_weight: float = 10.0
    ):
        """
        Args:
            loss_names: 损失函数名称列表，如 ['feat_loss', 'wave_loss', 'moe_loss']
            initial_weights: 初始权重字典
            adaptation_rate: 权重适应速度
            gradient_clip_norm: 梯度裁剪范数
            history_length: 历史记录长度
            min_weight: 最小权重值
            max_weight: 最大权重值
        """
        self.loss_names = loss_names
        self.adaptation_rate = adaptation_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.history_length = history_length
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 初始化权重
        if initial_weights is None:
            self.weights = {name: 1.0 for name in loss_names}
        else:
            self.weights = {name: initial_weights.get(name, 1.0) for name in loss_names}

        # 历史记录
        self.loss_history = {name: deque(maxlen=history_length) for name in loss_names}
        self.gradient_history = {name: deque(maxlen=history_length) for name in loss_names}

        # 统计信息
        self.step_count = 0
        self.weight_history = {name: deque(maxlen=history_length) for name in loss_names}

    def compute_loss_statistics(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算损失统计信息"""
        stats = {}

        for name, loss in losses.items():
            if name in self.loss_names:
                loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

                # 更新历史记录
                self.loss_history[name].append(loss_val)

                # 计算统计量
                history = list(self.loss_history[name])
                if len(history) > 1:
                    recent_avg = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                    long_avg = np.mean(history)
                    variance = np.var(history)
                    trend = np.polyfit(range(len(history)), history, 1)[0] if len(history) > 5 else 0.0

                    stats[name] = {
                        'current': loss_val,
                        'recent_avg': recent_avg,
                        'long_avg': long_avg,
                        'variance': variance,
                        'trend': trend,
                        'relative_scale': loss_val / (long_avg + 1e-8)
                    }
                else:
                    stats[name] = {
                        'current': loss_val,
                        'recent_avg': loss_val,
                        'long_avg': loss_val,
                        'variance': 0.0,
                        'trend': 0.0,
                        'relative_scale': 1.0
                    }

        return stats

    def compute_gradient_norms(self, model: nn.Module, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算各损失对参数的梯度范数"""
        gradient_norms = {}

        # 保存当前梯度
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        for loss_name, loss in losses.items():
            if loss_name in self.loss_names and isinstance(loss, torch.Tensor):
                # 清零梯度
                model.zero_grad()

                # 计算当前损失的梯度
                loss.backward(retain_graph=True)

                # 计算梯度范数
                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                gradient_norms[loss_name] = total_norm
                self.gradient_history[loss_name].append(total_norm)

        # 恢复原始梯度
        model.zero_grad()
        for name, param in model.named_parameters():
            if name in original_grads:
                param.grad = original_grads[name]

        return gradient_norms

    def update_weights_dynamic(self, loss_stats: Dict[str, Dict], gradient_norms: Dict[str, float]) -> Dict[str, float]:
        """动态更新损失权重"""
        new_weights = {}

        # 计算相对重要性
        total_gradient_norm = sum(gradient_norms.values()) + 1e-8

        for name in self.loss_names:
            if name in loss_stats and name in gradient_norms:
                stats = loss_stats[name]
                grad_norm = gradient_norms[name]

                # 策略1: 梯度范数平衡 - 如果某个损失的梯度太大，降低其权重
                grad_ratio = grad_norm / (total_gradient_norm / len(self.loss_names))
                grad_adjustment = 1.0 / (1.0 + grad_ratio) if grad_ratio > 1.0 else 1.0

                # 策略2: 损失趋势调整 - 如果某个损失在持续上升，增加其权重
                trend_adjustment = 1.0
                if abs(stats['trend']) > 1e-6:
                    # 上升趋势增加权重，下降趋势减少权重
                    trend_adjustment = 1.0 + self.adaptation_rate * stats['trend']

                # 策略3: 相对尺度调整 - 如果某个损失相对于历史平均值过大，降低权重
                scale_adjustment = 1.0 / stats['relative_scale'] if stats['relative_scale'] > 1.0 else 1.0

                # 综合调整
                current_weight = self.weights[name]
                adjustment = grad_adjustment * trend_adjustment * scale_adjustment
                new_weight = current_weight * (1.0 - self.adaptation_rate + self.adaptation_rate * adjustment)

                # 权重限制
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                new_weights[name] = new_weight

                # 更新权重历史
                self.weight_history[name].append(new_weight)
            else:
                new_weights[name] = self.weights[name]

        # 归一化权重（可选）
        if len(new_weights) > 1:
            total_weight = sum(new_weights.values())
            for name in new_weights:
                new_weights[name] = new_weights[name] / total_weight * len(new_weights)

        self.weights.update(new_weights)
        return new_weights

    def compute_balanced_loss(
        self,
        losses: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        enable_gradient_balancing: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """
        计算平衡后的总损失

        Args:
            losses: 各损失函数的值
            model: 模型（用于梯度分析）
            enable_gradient_balancing: 是否启用梯度平衡

        Returns:
            total_loss: 平衡后的总损失
            weights_used: 使用的权重
            loss_details: 损失详细信息
        """
        self.step_count += 1

        # 计算损失统计信息
        loss_stats = self.compute_loss_statistics(losses)

        # 计算梯度范数（如果提供了模型）
        gradient_norms = {}
        if model is not None and enable_gradient_balancing:
            try:
                gradient_norms = self.compute_gradient_norms(model, losses)
            except Exception as e:
                print(f"Warning: Failed to compute gradient norms: {e}")

        # 动态更新权重（每10步更新一次）
        if self.step_count % 10 == 0 and gradient_norms:
            self.update_weights_dynamic(loss_stats, gradient_norms)

        # 计算加权总损失
        total_loss = torch.zeros((), device=next(iter(losses.values())).device)
        weights_used = {}
        loss_details = {}

        for name, loss in losses.items():
            if name in self.loss_names:
                weight = self.weights[name]
                weighted_loss = weight * loss
                total_loss = total_loss + weighted_loss

                weights_used[name] = weight
                loss_details[f"{name}_raw"] = loss.item() if isinstance(loss, torch.Tensor) else loss
                loss_details[f"{name}_weighted"] = weighted_loss.item() if isinstance(weighted_loss, torch.Tensor) else weighted_loss

        return total_loss, weights_used, loss_details

    def get_status_report(self) -> Dict:
        """获取平衡器状态报告"""
        report = {
            'step_count': self.step_count,
            'current_weights': self.weights.copy(),
            'weight_trends': {},
            'loss_trends': {}
        }

        # 计算权重趋势
        for name in self.loss_names:
            if len(self.weight_history[name]) > 5:
                history = list(self.weight_history[name])
                trend = np.polyfit(range(len(history)), history, 1)[0]
                report['weight_trends'][name] = trend

        # 计算损失趋势
        for name in self.loss_names:
            if len(self.loss_history[name]) > 5:
                history = list(self.loss_history[name])
                trend = np.polyfit(range(len(history)), history, 1)[0]
                avg_loss = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                report['loss_trends'][name] = {
                    'trend': trend,
                    'recent_avg': avg_loss,
                    'variance': np.var(history)
                }

        return report


class GradientBalancer:
    """梯度平衡器 - 确保不同损失的梯度范数平衡"""

    def __init__(self, target_gradient_ratio: float = 1.0, adaptation_rate: float = 0.1):
        """
        Args:
            target_gradient_ratio: 目标梯度比率
            adaptation_rate: 适应速度
        """
        self.target_gradient_ratio = target_gradient_ratio
        self.adaptation_rate = adaptation_rate
        self.gradient_scales = {}

    def balance_gradients(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        平衡梯度并返回调整后的权重

        Args:
            model: 模型
            losses: 损失字典
            loss_weights: 当前损失权重

        Returns:
            调整后的权重
        """
        if len(losses) <= 1:
            return loss_weights

        # 计算各损失的梯度范数
        gradient_norms = {}
        original_grads = {}

        # 保存原始梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        # 分别计算各损失的梯度范数
        for loss_name, loss in losses.items():
            model.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward(retain_graph=True)

                total_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)

                gradient_norms[loss_name] = total_norm

        # 恢复原始梯度
        model.zero_grad()
        for name, param in model.named_parameters():
            if name in original_grads:
                param.grad = original_grads[name]

        # 计算平衡权重
        if gradient_norms:
            avg_grad_norm = sum(gradient_norms.values()) / len(gradient_norms)
            balanced_weights = {}

            for loss_name, weight in loss_weights.items():
                if loss_name in gradient_norms:
                    grad_norm = gradient_norms[loss_name]
                    if grad_norm > 0:
                        # 目标：使所有损失的梯度范数接近平均值
                        scale_factor = avg_grad_norm / grad_norm
                        # 平滑调整
                        if loss_name not in self.gradient_scales:
                            self.gradient_scales[loss_name] = 1.0

                        target_scale = self.gradient_scales[loss_name] * scale_factor
                        self.gradient_scales[loss_name] = (
                            (1 - self.adaptation_rate) * self.gradient_scales[loss_name] +
                            self.adaptation_rate * target_scale
                        )

                        balanced_weights[loss_name] = weight * self.gradient_scales[loss_name]
                    else:
                        balanced_weights[loss_name] = weight
                else:
                    balanced_weights[loss_name] = weight

            return balanced_weights

        return loss_weights


def create_stage3_loss_balancer() -> MultiObjectiveBalancer:
    """为Stage3训练创建损失平衡器"""
    return MultiObjectiveBalancer(
        loss_names=['feat_loss', 'wave_loss', 'moe_loss'],
        initial_weights={
            'feat_loss': 0.4,   # 特征重建：中等权重
            'wave_loss': 0.5,   # 波形质量：稍高权重
            'moe_loss': 0.1     # MoE平衡：较低权重
        },
        adaptation_rate=0.02,  # 较慢的适应速度，避免振荡
        gradient_clip_norm=1.0,
        history_length=200,
        min_weight=0.01,
        max_weight=5.0
    )


def create_gradient_balancer() -> GradientBalancer:
    """创建梯度平衡器"""
    return GradientBalancer(
        target_gradient_ratio=1.0,
        adaptation_rate=0.05
    )


if __name__ == "__main__":
    # 简单测试
    import torch.nn as nn

    # 创建测试模型
    model = nn.Linear(10, 1)

    # 创建测试损失
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    pred = model(x)

    losses = {
        'feat_loss': nn.functional.mse_loss(pred, y),
        'wave_loss': nn.functional.l1_loss(pred, y),
        'moe_loss': torch.tensor(0.1)
    }

    # 测试平衡器
    balancer = create_stage3_loss_balancer()

    for step in range(10):
        total_loss, weights, details = balancer.compute_balanced_loss(losses, model)
        print(f"Step {step}: Total loss: {total_loss.item():.4f}")
        print(f"  Weights: {weights}")
        print(f"  Details: {details}")
        print()

        # 模拟训练步骤
        total_loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= 0.01 * param.grad
        model.zero_grad()

        # 重新计算损失
        pred = model(x)
        losses = {
            'feat_loss': nn.functional.mse_loss(pred, y),
            'wave_loss': nn.functional.l1_loss(pred, y),
            'moe_loss': torch.tensor(0.1)
        }

    print("Final status report:")
    print(balancer.get_status_report())