# -*- coding: utf-8 -*-
"""
Loss helpers for AETHER training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import math


def l1_stft_loss(y_hat: torch.Tensor, y: torch.Tensor, lambda_stft: float) -> torch.Tensor:
    """
    Combination of L1 reconstruction and a lightweight magnitude STFT loss.
    """
    recon = torch.nn.functional.l1_loss(y_hat, y)
    if lambda_stft <= 0:
        return recon

    y_hat_spec = torch.fft.rfft(y_hat.transpose(1, 2).float(), dim=-1)
    y_spec = torch.fft.rfft(y.transpose(1, 2).float(), dim=-1)
    mag_hat = torch.log1p(y_hat_spec.abs())
    mag_ref = torch.log1p(y_spec.abs())
    mag_diff = torch.nn.functional.l1_loss(mag_hat, mag_ref)
    return recon + lambda_stft * mag_diff


def rate_loss(z: torch.Tensor, lambda_rate: float) -> torch.Tensor:
    """
    Penalise high-energy latent codes as a proxy for bitrate control.
    """
    if lambda_rate <= 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    # Check for NaN/Inf in z and clean if necessary (mixed precision stability)
    if torch.isnan(z).any() or torch.isinf(z).any():
        print(f"⚠️ NaN/Inf detected in z tensor for rate_loss, cleaning...")
        z_clean = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return lambda_rate * z_clean.pow(2).mean()

    return lambda_rate * z.pow(2).mean()


def balance_loss(
    aux: Optional[Dict[str, torch.Tensor]],
    lambda_balance: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Encourage fair expert utilisation.
    """
    if not aux or lambda_balance <= 0 or "balance_loss" not in aux:
        target_device = device or (aux["prob"].device if aux and "prob" in aux else torch.device("cpu"))
        return torch.zeros((), device=target_device)
    return lambda_balance * aux["balance_loss"]


def router_consistency_loss(
    p1: Optional[torch.Tensor],
    p2: Optional[torch.Tensor],
    lambda_cons: float,
) -> torch.Tensor:
    """
    Make router decisions stable under different masking schedules.
    """
    if p1 is None or p2 is None or lambda_cons <= 0:
        device = p1.device if p1 is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    return lambda_cons * torch.mean((p1 - p2) ** 2)


def compute_layered_loss(
    pred_features,
    target_features,
    current_step: int,
    feature_spec_type: str = "aether",
    disable_f0: bool = False,
    scale_correction: bool = False,
):
    """
    计算分层特征损失，支持AETHER和FARGAN特征规范

    Args:
        pred_features: 预测特征 [B, T, D]
        target_features: 目标特征 [B, T, D]
        current_step: 当前训练步数
        feature_spec_type: 特征规范类型 ("aether" 或 "fargan")
    """
    if feature_spec_type == "fargan":
        try:
            from models.feature_adapter import get_fargan_feature_spec
            spec = get_fargan_feature_spec()

            if current_step < 1000:
                names, stage = spec.get_perceptually_critical_features(), "critical_only"
            elif current_step < 3000:
                names, stage = ["ceps", "dnn_pitch", "frame_corr"], "high_priority"
            else:
                names, stage = ["ceps", "dnn_pitch", "frame_corr", "lpc"], "full_features"

            weights = spec.get_feature_importance_weights()

        except ImportError:
            # 回退到统一损失
            return F.mse_loss(pred_features, target_features), {"unified_mse": F.mse_loss(pred_features, target_features).item()}, "unified"
            if disable_f0:
                # 移除 F0（dnn_pitch）相关项
                names = [n for n in names if n not in ("dnn_pitch",)]
        except ImportError:
            # 回退到统一损失
            return F.mse_loss(pred_features, target_features), {"unified_mse": F.mse_loss(pred_features, target_features).item()}, "unified"
    else:
        from utils.feature_spec import get_default_feature_spec
        spec = get_default_feature_spec()
        if current_step < 1000:
            names, stage = spec.get_perceptually_critical_features(), "critical_only"
        elif current_step < 3000:
            names, stage = spec.get_high_priority_features(0.7), "high_priority"
        else:
            names, stage = ["ceps","f0","voicing","enhanced","lpc","prosodic"], "full_features"

        if disable_f0:
            # 移除 F0 项
            names = [n for n in names if n != "f0"]

        weights = spec.get_feature_importance_weights()

    total, details = pred_features.new_zeros(()), {}

    for n in names:
        try:
            if feature_spec_type == "fargan":
                # 统一使用 get_feature_slice 方法
                sl = spec.get_feature_slice(n)
                p, t = pred_features[..., sl], target_features[..., sl]
            else:
                sl = spec.get_feature_slice(n)
                p, t = pred_features[..., sl], target_features[..., sl]

            w = float(weights.get(n, 1.0))

            # 数值稳定性保护：限制损失的最大值，防止损失爆炸
            loss = torch.clamp(F.mse_loss(p, t), max=50.0)

            # 特殊处理不同特征类型
            if n in ("f0", "dnn_pitch"):
                l1_component = torch.clamp(F.l1_loss(p, t), max=25.0)
                loss = loss + 0.5 * l1_component
            elif n in ("voicing", "frame_corr"):
                # 对于相关系数或voicing，使用Huber损失更稳定
                huber_component = torch.clamp(F.huber_loss(p, t, delta=0.1), max=25.0)
                loss = loss + 0.3 * huber_component

            total = total + w * loss
            details[n] = float((w * loss).item())

        except (KeyError, ValueError) as e:
            # 如果特征不存在，跳过
            continue

    # 添加改进的监控信息：既显示平均损失，也显示每序列总和（更直观）
    seq_len = pred_features.size(1) if len(pred_features.shape) > 2 else 1
    per_seq_total = total * seq_len  # 每序列的总损失

    # 在details中添加监控信息
    details['mean_loss'] = float(total.item())
    details['per_seq_total'] = float(per_seq_total.item())
    details['scale_factor'] = seq_len
    details['stage'] = stage

    return total, details, stage


def extract_mfcc_features(audio: torch.Tensor, sample_rate: int = 16000, n_mfcc: int = 13) -> torch.Tensor:
    """
    提取MFCC特征用于音频相似度计算

    Args:
        audio: [B, T] 音频信号
        sample_rate: 采样率
        n_mfcc: MFCC维数

    Returns:
        mfcc: [B, n_mfcc, T'] MFCC特征
    """
    try:
        # 确保音频是2D并且长度足够
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # 检查音频长度，太短的音频无法处理
        if audio.size(-1) < 512:
            return torch.zeros(audio.size(0), n_mfcc, 1, device=audio.device)

        # 使用hann窗避免频谱泄漏警告
        window = torch.hann_window(400, device=audio.device)

        # 计算STFT
        stft = torch.stft(
            audio,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            pad_mode='constant'
        )

        # 计算幅度谱
        mag_spec = torch.abs(stft)  # [B, F, T']

        # 确保有足够的时间帧
        if mag_spec.size(-1) == 0:
            return torch.zeros(audio.size(0), n_mfcc, 1, device=audio.device)

        # 生成Mel滤波器组
        n_freqs = mag_spec.size(1)  # 频率bin数量
        n_mels = max(n_mfcc * 2, 26)  # 确保足够的mel bins

        mel_filters = AF.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=80.0,  # 人声最低频率
            f_max=min(7600.0, sample_rate // 2),  # 人声最高频率
            n_mels=n_mels,
            sample_rate=sample_rate
        ).to(audio.device)

        # 应用Mel滤波器：[B, F, T'] x [F, n_mels] -> [B, n_mels, T']
        mel_spec = torch.matmul(mag_spec.transpose(1, 2), mel_filters).transpose(1, 2)

        # 取对数mel谱
        log_mel = torch.log(mel_spec + 1e-10)

        # 简化的DCT（取前n_mfcc个系数）
        mfcc = log_mel[:, :n_mfcc, :]  # [B, n_mfcc, T']

        return mfcc

    except Exception as e:
        # 失败时返回有意义的特征而不是零
        batch_size = audio.size(0)
        # 生成基于音频能量的简单特征
        if audio.size(-1) > 0:
            energy = torch.mean(audio.pow(2), dim=-1, keepdim=True)  # [B, 1]
            features = energy.unsqueeze(-1).expand(batch_size, n_mfcc, 1)
        else:
            features = torch.zeros(batch_size, n_mfcc, 1, device=audio.device)
        return features


def mfcc_cosine_similarity(audio_pred: torch.Tensor, audio_gt: torch.Tensor) -> torch.Tensor:
    """
    计算两个音频的MFCC余弦相似度

    Args:
        audio_pred: [B, T] 预测音频
        audio_gt: [B, T] 目标音频

    Returns:
        similarity: [B] 相似度分数 (0-1)
    """
    # 提取MFCC特征
    mfcc_pred = extract_mfcc_features(audio_pred)  # [B, n_mfcc, T']
    mfcc_gt = extract_mfcc_features(audio_gt)

    # 时间维度对齐
    min_time = min(mfcc_pred.size(-1), mfcc_gt.size(-1))
    if min_time == 0:
        return torch.zeros(audio_pred.size(0), device=audio_pred.device)

    mfcc_pred = mfcc_pred[..., :min_time]  # [B, n_mfcc, T']
    mfcc_gt = mfcc_gt[..., :min_time]

    # 展平为向量计算余弦相似度
    pred_flat = mfcc_pred.view(mfcc_pred.size(0), -1)  # [B, n_mfcc*T']
    gt_flat = mfcc_gt.view(mfcc_gt.size(0), -1)

    # 归一化
    pred_norm = F.normalize(pred_flat, p=2, dim=1)
    gt_norm = F.normalize(gt_flat, p=2, dim=1)

    # 余弦相似度
    similarity = torch.sum(pred_norm * gt_norm, dim=1)  # [B]
    return torch.clamp(similarity, 0.0, 1.0)


def detect_audio_anomalies(audio: torch.Tensor, sample_rate: int = 16000) -> Dict[str, torch.Tensor]:
    """
    检测音频中的异常情况

    Args:
        audio: [B, T] 音频信号
        sample_rate: 采样率

    Returns:
        anomalies: 各种异常指标的字典
    """
    batch_size = audio.size(0)
    device = audio.device

    # 1. 静音检测（改进的能量检测）
    audio_energy = torch.mean(audio.pow(2), dim=-1)  # [B]
    # 动态阈值：相对于最大能量的比例
    max_energy = torch.max(audio_energy) + 1e-8
    silence_threshold = max_energy * 1e-4  # 相对阈值
    silence_penalty = torch.relu(silence_threshold - audio_energy) / silence_threshold

    # 2. 爆音检测（改进的幅度检测）
    clipping_threshold = 0.95
    clipping_ratio = (torch.abs(audio) > clipping_threshold).float().mean(dim=-1)  # [B]
    clipping_penalty = torch.clamp(clipping_ratio * 10.0, 0.0, 1.0)  # 放大并限制

    # 3. 简化的周期性检测（替代复杂的F0检测）
    f0_validity = torch.zeros(batch_size, device=device)

    try:
        # 使用频域能量分布检测周期性
        if audio.size(-1) >= 512:
            window = torch.hann_window(400, device=device)
            stft = torch.stft(
                audio,
                n_fft=512,
                hop_length=160,
                win_length=400,
                window=window,
                return_complex=True,
                pad_mode='constant'
            )

            # 计算低频能量比例（50-400Hz对应的bins）
            mag_spec = torch.abs(stft)  # [B, F, T]
            total_energy = torch.sum(mag_spec, dim=1)  # [B, T]

            # 对应50-400Hz的频率bins
            f_min_bin = int(50 * 512 / sample_rate)
            f_max_bin = int(400 * 512 / sample_rate)
            voice_energy = torch.sum(mag_spec[:, f_min_bin:f_max_bin, :], dim=1)  # [B, T]

            # 计算语音频段能量比例
            voice_ratio = voice_energy / (total_energy + 1e-8)
            avg_voice_ratio = torch.mean(voice_ratio, dim=-1)  # [B]

            # 如果语音频段能量比例太低，认为可能有问题
            f0_validity = torch.relu(0.1 - avg_voice_ratio)

    except Exception:
        # 失败时给轻微惩罚
        f0_validity = torch.full((batch_size,), 0.1, device=device)

    return {
        'silence_penalty': silence_penalty,
        'clipping_penalty': clipping_penalty,
        'f0_validity_penalty': f0_validity
    }


def basic_audibility_loss(audio_pred: torch.Tensor, audio_gt: torch.Tensor) -> torch.Tensor:
    """
    基础可听性损失：确保音频不会出现静音、爆音等明显问题

    Args:
        audio_pred: [B, T] 预测音频
        audio_gt: [B, T] 目标音频

    Returns:
        loss: 基础可听性损失
    """
    anomalies = detect_audio_anomalies(audio_pred)

    # 组合各种异常惩罚
    total_penalty = (
        anomalies['silence_penalty'] +
        anomalies['clipping_penalty'] +
        anomalies['f0_validity_penalty']
    )

    return total_penalty.mean()


def speech_intelligibility_loss(audio_pred: torch.Tensor, audio_gt: torch.Tensor) -> torch.Tensor:
    """
    语音清晰度损失：基于MFCC相似度的语音质量评估

    Args:
        audio_pred: [B, T] 预测音频
        audio_gt: [B, T] 目标音频

    Returns:
        loss: 语音清晰度损失
    """
    # 计算MFCC相似度
    similarity = mfcc_cosine_similarity(audio_pred, audio_gt)  # [B]

    # 目标相似度阈值
    target_similarity = 0.8

    # 如果相似度低于阈值，给予惩罚
    penalty = torch.relu(target_similarity - similarity)

    return penalty.mean()


def perceptual_quality_loss(audio_pred: torch.Tensor, audio_gt: torch.Tensor) -> torch.Tensor:
    """
    感知质量损失：基于频谱距离的质量评估（PESQ/STOI代理）

    Args:
        audio_pred: [B, T] 预测音频
        audio_gt: [B, T] 目标音频

    Returns:
        loss: 感知质量损失
    """
    # 时间域对齐
    min_len = min(audio_pred.size(-1), audio_gt.size(-1))
    if min_len < 512:  # 需要足够长度进行频谱分析
        return F.l1_loss(audio_pred[..., :min_len], audio_gt[..., :min_len])

    audio_pred = audio_pred[..., :min_len]
    audio_gt = audio_gt[..., :min_len]

    # 频谱域损失（作为感知质量代理）
    try:
        # 使用hann窗避免频谱泄漏
        window = torch.hann_window(400, device=audio_pred.device)

        # STFT
        pred_stft = torch.stft(
            audio_pred,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            pad_mode='constant'
        )
        gt_stft = torch.stft(
            audio_gt,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            return_complex=True,
            pad_mode='constant'
        )

        # 幅度谱
        pred_mag = torch.abs(pred_stft) + 1e-10
        gt_mag = torch.abs(gt_stft) + 1e-10

        # 对数幅度谱损失（更稳定的版本）
        pred_log_mag = torch.log(pred_mag)
        gt_log_mag = torch.log(gt_mag)
        spectral_loss = F.l1_loss(pred_log_mag, gt_log_mag)

        # 限制损失范围，避免数值爆炸
        spectral_loss = torch.clamp(spectral_loss, 0.0, 10.0)

        # 能量匹配损失
        pred_energy = torch.sum(pred_mag.pow(2), dim=1)  # [B, T]
        gt_energy = torch.sum(gt_mag.pow(2), dim=1)      # [B, T]
        energy_loss = F.l1_loss(pred_energy, gt_energy) * 0.1
        energy_loss = torch.clamp(energy_loss, 0.0, 1.0)

        # 相位一致性（简化版）
        phase_consistency = torch.cos(torch.angle(pred_stft) - torch.angle(gt_stft))
        phase_loss = (1.0 - phase_consistency.mean()) * 0.05
        phase_loss = torch.clamp(phase_loss, 0.0, 0.5)

        total_loss = spectral_loss + energy_loss + phase_loss
        return torch.clamp(total_loss, 0.0, 10.0)

    except Exception as e:
        # 失败时回退到时域损失
        l1_loss = F.l1_loss(audio_pred, audio_gt)
        return torch.clamp(l1_loss, 0.0, 2.0)


def audio_usability_loss(
    audio_pred: torch.Tensor,
    audio_gt: torch.Tensor,
    stage: str = "balanced",
    sample_rate: int = 16000
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    音频可用性损失：分层评估音频质量和可用性

    Args:
        audio_pred: [B, T] 预测音频
        audio_gt: [B, T] 目标音频
        stage: 训练阶段 ("foundation", "balanced", "quality")
        sample_rate: 采样率

    Returns:
        total_loss: 总损失
        loss_details: 各组件损失详情
    """
    # 输入验证和预处理
    if audio_pred.size() != audio_gt.size():
        min_len = min(audio_pred.size(-1), audio_gt.size(-1))
        audio_pred = audio_pred[..., :min_len]
        audio_gt = audio_gt[..., :min_len]

    # 如果音频太短，使用简化损失
    if audio_pred.size(-1) < 160:  # 少于10ms
        simple_loss = F.l1_loss(audio_pred, audio_gt)
        return simple_loss, {
            'audibility': simple_loss.item(),
            'intelligibility': 0.0,
            'quality': 0.0,
            'total': simple_loss.item(),
            'stage': stage,
            'weights': [1.0, 0.0, 0.0]
        }

    # 计算各组件损失，添加数值稳定性保护
    try:
        loss_basic = basic_audibility_loss(audio_pred, audio_gt)
        # 归一化到0-1范围：假设原始范围是0-2，映射到0-1
        loss_basic = torch.clamp(loss_basic / 2.0, 0.0, 1.0)
    except Exception:
        loss_basic = torch.clamp(F.l1_loss(audio_pred, audio_gt), 0.0, 1.0)

    try:
        loss_intelligibility = speech_intelligibility_loss(audio_pred, audio_gt)
        loss_intelligibility = torch.clamp(loss_intelligibility, 0.0, 1.0)
    except Exception:
        loss_intelligibility = F.l1_loss(audio_pred, audio_gt) * 0.5

    try:
        loss_quality = perceptual_quality_loss(audio_pred, audio_gt)
        # 修正归一化：perceptual_quality_loss返回0-10范围，映射到0-1
        # 使用更合理的归一化，避免损失值过小
        loss_quality = torch.clamp(loss_quality / 10.0, 0.0, 1.0)
        # 添加最小阈值，确保损失值不会完全为0
        loss_quality = torch.clamp(loss_quality + 1e-6, 0.0, 1.0)
    except Exception:
        loss_quality = torch.clamp(F.l1_loss(audio_pred, audio_gt), 0.0, 1.0)

    # 根据训练阶段调整权重
    if stage == "foundation":
        weights = [0.6, 0.3, 0.1]  # [可听性, 清晰度, 质量]
    elif stage == "balanced":
        weights = [0.3, 0.5, 0.2]
    elif stage == "quality":
        weights = [0.2, 0.3, 0.5]
    else:
        weights = [0.33, 0.33, 0.34]

    # 计算加权总损失
    total_loss = (
        weights[0] * loss_basic +
        weights[1] * loss_intelligibility +
        weights[2] * loss_quality
    )

    # 确保总损失在合理范围内
    total_loss = torch.clamp(total_loss, 0.0, 10.0)

    # 损失详情（安全地转换为float）
    try:
        loss_details = {
            'audibility': float(loss_basic.item()),
            'intelligibility': float(loss_intelligibility.item()),
            'quality': float(loss_quality.item()),
            'total': float(total_loss.item()),
            'stage': stage,
            'weights': weights
        }
    except Exception:
        # 备用版本
        loss_details = {
            'audibility': 0.0,
            'intelligibility': 0.0,
            'quality': 0.0,
            'total': float(total_loss.item()) if isinstance(total_loss, torch.Tensor) else 0.0,
            'stage': stage,
            'weights': weights
        }

    return total_loss, loss_details


__all__ = [
    "l1_stft_loss",
    "rate_loss",
    "balance_loss",
    "router_consistency_loss",
    "compute_layered_loss",
    "audio_usability_loss",
    "mfcc_cosine_similarity",
    "basic_audibility_loss",
    "speech_intelligibility_loss",
    "perceptual_quality_loss",
]
