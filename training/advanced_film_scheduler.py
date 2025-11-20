#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced FiLM Scheduler - 分段+滞后+失稳回退的FiLM激活调度器
基于用户反馈的稳健性改进
"""

import torch
from typing import Tuple, List, Optional
import numpy as np


class AdvancedFiLMScheduler:
    """高级FiLM调度器：分段激活+β滞后+失稳治理"""

    def __init__(
        self,
        total_warmup_steps: int = 300,
        film_start_ratio: float = 0.25,
        beta_lag_steps: int = 30,
        beta_scale: float = 0.95,
        instability_threshold: float = 1.25,  # 25%损失跳升视为失稳(更宽松)
        recovery_factor: float = 0.95,        # 失稳时回退5%(更温和)
        min_ratio: float = 0.7               # 最低不低于70%
    ):
        self.total_steps = total_warmup_steps
        self.start_ratio = film_start_ratio
        self.beta_lag = beta_lag_steps
        self.beta_scale = beta_scale
        self.threshold = instability_threshold
        self.recovery = recovery_factor
        self.min_ratio = min_ratio

        # 状态跟踪
        self.loss_history: List[float] = []
        self.ratio_history: List[float] = []
        self.recovery_count = 0
        self.spike_steps: List[int] = []  # 记录失稳步数

    def film_schedule(self, step: int) -> float:
        """改进的分段式FiLM激活调度 - 确保到达100%"""
        if step <= 80:
            # 第一段: 0.25 → 0.65 (快速启动)
            progress = step / 80.0
            ratio = self.start_ratio + (0.65 - self.start_ratio) * progress

        elif step <= 160:
            # 第二段: 0.65 → 0.85 (稳定爬升)
            progress = (step - 80) / 80.0
            ratio = 0.65 + 0.20 * progress

        elif step <= self.total_steps - 50:
            # 第三段: 0.85 → 0.95 (缓慢接近)
            progress = (step - 160) / (self.total_steps - 160 - 50)
            ratio = 0.85 + 0.10 * progress

        else:
            # 最后50步: 0.95 → 1.00 (确保到达)
            progress = (step - (self.total_steps - 50)) / 50.0
            ratio = 0.95 + 0.05 * progress

        return min(1.0, ratio)

    def beta_schedule(self, step: int) -> float:
        """β调度：滞后30步且略缩放"""
        lagged_step = max(0, step - self.beta_lag)
        return self.film_schedule(lagged_step) * self.beta_scale

    def governor(self, current_loss: float, film_ratio: float, step: int) -> float:
        """改进的稳态治理：平滑检测+适应性回退"""
        self.loss_history.append(current_loss)

        # 需要至少3个历史点才开始检测
        if len(self.loss_history) < 3:
            return film_ratio

        # 使用3点平均来平滑噪声
        recent_losses = self.loss_history[-3:]
        avg_prev = np.mean(recent_losses[:-1])

        # 检测显著跳升（使用平滑值）
        if current_loss > avg_prev * self.threshold:
            self.recovery_count += 1
            self.spike_steps.append(step)

            # 适应性回退：根据跳升幅度调整回退程度
            spike_magnitude = current_loss / avg_prev
            if spike_magnitude > 1.5:  # 50%以上跳升
                recovery_factor = 0.85  # 强回退
            elif spike_magnitude > 1.3:  # 30-50%跳升
                recovery_factor = 0.92  # 中等回退
            else:  # 25-30%跳升
                recovery_factor = 0.97  # 轻微回退

            recovered_ratio = max(self.min_ratio, film_ratio * recovery_factor)
            print(f"    🚨 失稳检测: loss {avg_prev:.4f}→{current_loss:.4f} "
                  f"(+{(spike_magnitude-1)*100:.1f}%) film {film_ratio:.3f}→{recovered_ratio:.3f}")
            return recovered_ratio

        return film_ratio

    def get_activation_ratios(self, step: int, current_loss: float) -> Tuple[float, float]:
        """获取当前步的FiLM激活比例"""
        # 基础调度
        film_ratio = self.film_schedule(step)
        beta_ratio = self.beta_schedule(step)

        # 失稳治理
        film_ratio = self.governor(current_loss, film_ratio, step)

        # 记录历史
        self.ratio_history.append(film_ratio)

        return film_ratio, beta_ratio

    def get_statistics(self) -> dict:
        """获取调度统计信息"""
        # 计算最后50步的跳升次数
        total_steps = len(self.ratio_history)
        last_50_spikes = sum(1 for step in self.spike_steps if step >= max(1, total_steps - 50))

        return {
            'total_recovery_events': self.recovery_count,
            'final_film_ratio': self.ratio_history[-1] if self.ratio_history else 0.0,
            'avg_film_ratio': np.mean(self.ratio_history) if self.ratio_history else 0.0,
            'film_stability': 1.0 - (len([r for i, r in enumerate(self.ratio_history[1:])
                                         if abs(r - self.ratio_history[i]) > 0.05]) / max(1, len(self.ratio_history))),
            'spikes_last_50': last_50_spikes,
            'loss_trend': 'decreasing' if len(self.loss_history) > 10 and
                         self.loss_history[-1] < self.loss_history[10] else 'unstable'
        }


def create_film_parameter_groups(encoder, base_lr: float = 4e-4):
    """创建FiLM专用参数组"""
    film_params = []
    other_params = []

    for name, param in encoder.named_parameters():
        if 'film' in name.lower():
            film_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': 1e-5
        },
        {
            'params': film_params,
            'lr': base_lr * 1.2,  # FiLM专用更高学习率
            'weight_decay': 0.0   # FiLM参数不使用权重衰减
        }
    ]

    print(f"参数组配置: 基础={len(other_params)}个参数 lr={base_lr}, "
          f"FiLM={len(film_params)}个参数 lr={base_lr * 1.2}")

    return param_groups


def test_scheduler_behavior():
    """测试调度器行为"""
    print("🧪 测试AdvancedFiLMScheduler")
    print("=" * 50)

    scheduler = AdvancedFiLMScheduler(total_warmup_steps=300)

    # 模拟训练过程
    print("测试正常训练过程:")
    losses = [0.5, 0.45, 0.42, 0.40, 0.38, 0.35]  # 正常下降

    for step in [1, 50, 100, 150, 200, 250]:
        loss = losses[min(step//50, len(losses)-1)]
        film_ratio, beta_ratio = scheduler.get_activation_ratios(step, loss)
        print(f"  Step {step:3d}: film={film_ratio:.3f} β={beta_ratio:.3f} loss={loss:.3f}")

    print("\n测试失稳恢复:")
    # 模拟第150步失稳
    step = 150
    stable_loss = 0.35
    spike_loss = 0.45  # 28%跳升

    film_before, beta_before = scheduler.get_activation_ratios(step, stable_loss)
    film_after, beta_after = scheduler.get_activation_ratios(step+1, spike_loss)

    print(f"  失稳前: film={film_before:.3f} β={beta_before:.3f}")
    print(f"  失稳后: film={film_after:.3f} β={beta_after:.3f}")
    print(f"  回退幅度: {(film_before - film_after) / film_before * 100:.1f}%")

    # 统计信息
    stats = scheduler.get_statistics()
    print(f"\n📊 调度统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_scheduler_behavior()