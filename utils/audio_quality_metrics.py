#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实音频质量指标评估 - 替代PESQ-like代理指标
集成PESQ、STOI等标准音频质量度量
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# 抑制PESQ库的warnings
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False


class RealAudioQualityAssessment:
    """真实音频质量评估器"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.pesq_available = PESQ_AVAILABLE
        self.stoi_available = STOI_AVAILABLE

        if not self.pesq_available:
            print("警告: PESQ库未安装，将跳过PESQ评估")
        if not self.stoi_available:
            print("警告: PySTOI库未安装，将跳过STOI评估")

    def compute_snr_db(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """计算信噪比 (dB)"""
        if len(original) != len(reconstructed):
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]

        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)

        if noise_power < 1e-10:
            return 60.0  # 极高SNR的上限

        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)

    def compute_pesq(self, original: np.ndarray, reconstructed: np.ndarray) -> Optional[float]:
        """计算真实PESQ分数"""
        if not self.pesq_available:
            return None

        try:
            if len(original) != len(reconstructed):
                min_len = min(len(original), len(reconstructed))
                original = original[:min_len]
                reconstructed = reconstructed[:min_len]

            # 确保音频长度至少0.25秒 (PESQ要求)
            min_samples = int(0.25 * self.sample_rate)
            if len(original) < min_samples:
                return None

            # 归一化到[-1, 1]范围
            original = np.clip(original / (np.max(np.abs(original)) + 1e-8), -1, 1)
            reconstructed = np.clip(reconstructed / (np.max(np.abs(reconstructed)) + 1e-8), -1, 1)

            # 计算PESQ (使用窄带模式 'nb' 对16kHz采样率)
            pesq_score = pesq(self.sample_rate, original, reconstructed, 'nb')
            return float(pesq_score)

        except Exception as e:
            print(f"PESQ计算失败: {e}")
            return None

    def compute_estoi(self, original: np.ndarray, reconstructed: np.ndarray) -> Optional[float]:
        """计算扩展短时客观可理解性指数 (ESTOI)"""
        if not self.stoi_available:
            return None

        try:
            if len(original) != len(reconstructed):
                min_len = min(len(original), len(reconstructed))
                original = original[:min_len]
                reconstructed = reconstructed[:min_len]

            # STOI要求至少0.3秒的音频
            min_samples = int(0.3 * self.sample_rate)
            if len(original) < min_samples:
                return None

            # 归一化
            original = original / (np.max(np.abs(original)) + 1e-8)
            reconstructed = reconstructed / (np.max(np.abs(reconstructed)) + 1e-8)

            # 计算ESTOI
            estoi_score = stoi(original, reconstructed, self.sample_rate, extended=True)
            return float(estoi_score)

        except Exception as e:
            print(f"ESTOI计算失败: {e}")
            return None

    def compute_spectral_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """计算频谱域指标"""
        if len(original) != len(reconstructed):
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]

        # 计算频谱
        orig_fft = np.fft.rfft(original)
        recon_fft = np.fft.rfft(reconstructed)

        orig_mag = np.abs(orig_fft)
        recon_mag = np.abs(recon_fft)

        # 频谱相关性
        correlation = np.corrcoef(orig_mag, recon_mag)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # 频谱失真 (对数域均方误差)
        log_orig = np.log(orig_mag + 1e-8)
        log_recon = np.log(recon_mag + 1e-8)
        spectral_distortion = np.mean((log_orig - log_recon) ** 2)

        return {
            'spectral_correlation': float(correlation),
            'spectral_distortion': float(spectral_distortion)
        }

    def assess_audio_quality(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """全面音频质量评估"""
        # 转换为numpy并确保是1D
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy().flatten()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy().flatten()

        results = {}

        # 基础指标
        results['snr_db'] = self.compute_snr_db(original, reconstructed)

        # 真实PESQ
        pesq_score = self.compute_pesq(original, reconstructed)
        if pesq_score is not None:
            results['pesq'] = pesq_score

        # ESTOI
        estoi_score = self.compute_estoi(original, reconstructed)
        if estoi_score is not None:
            results['estoi'] = estoi_score

        # 频谱指标
        spectral_metrics = self.compute_spectral_metrics(original, reconstructed)
        results.update(spectral_metrics)

        return results

    def get_quality_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """获取质量等级总结"""
        summary = {}

        # SNR等级
        snr = metrics.get('snr_db', 0)
        if snr >= 15:
            summary['snr_level'] = 'Excellent'
        elif snr >= 10:
            summary['snr_level'] = 'Good'
        elif snr >= 5:
            summary['snr_level'] = 'Fair'
        else:
            summary['snr_level'] = 'Poor'

        # PESQ等级 (1.0-4.5范围)
        if 'pesq' in metrics:
            pesq = metrics['pesq']
            if pesq >= 3.5:
                summary['pesq_level'] = 'Excellent'
            elif pesq >= 2.5:
                summary['pesq_level'] = 'Good'
            elif pesq >= 1.5:
                summary['pesq_level'] = 'Fair'
            else:
                summary['pesq_level'] = 'Poor'

        # ESTOI等级 (0-1范围)
        if 'estoi' in metrics:
            estoi = metrics['estoi']
            if estoi >= 0.9:
                summary['estoi_level'] = 'Excellent'
            elif estoi >= 0.7:
                summary['estoi_level'] = 'Good'
            elif estoi >= 0.5:
                summary['estoi_level'] = 'Fair'
            else:
                summary['estoi_level'] = 'Poor'

        # 整体质量等级
        levels = [summary.get('snr_level'), summary.get('pesq_level'), summary.get('estoi_level')]
        excellent_count = levels.count('Excellent')
        good_count = levels.count('Good')

        if excellent_count >= 2:
            summary['overall_quality'] = 'Excellent'
        elif excellent_count + good_count >= 2:
            summary['overall_quality'] = 'Good'
        elif 'Poor' not in levels:
            summary['overall_quality'] = 'Fair'
        else:
            summary['overall_quality'] = 'Poor'

        return summary


def test_audio_quality_metrics():
    """测试音频质量指标"""
    print("🧪 测试真实音频质量指标")
    print("=" * 40)

    # 创建测试音频
    sr = 16000
    duration = 1.0  # 1秒
    t = np.linspace(0, duration, int(sr * duration))

    # 原始信号：纯音
    freq = 440  # A4音符
    original = np.sin(2 * np.pi * freq * t)

    # 重建信号：加噪声
    noise_level = 0.1
    reconstructed = original + noise_level * np.random.randn(len(original))

    # 评估
    assessor = RealAudioQualityAssessment(sr)
    metrics = assessor.assess_audio_quality(original, reconstructed)
    summary = assessor.get_quality_summary(metrics)

    print("指标结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n质量等级:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n可用指标: PESQ={'✅' if assessor.pesq_available else '❌'}, STOI={'✅' if assessor.stoi_available else '❌'}")


if __name__ == "__main__":
    test_audio_quality_metrics()