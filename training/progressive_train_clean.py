#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AETHER Progressive Training - 精简版
移除了调试、wandb日志等冗余代码，专注核心训练逻辑
"""

import argparse
import random
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import traceback
import torchaudio

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    def _create_grad_scaler(enabled: bool) -> "_GradScaler":
        return _GradScaler(init_scale=64.0, growth_interval=1000, enabled=enabled)
    def _autocast_ctx(enabled: bool):
        return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    def _create_grad_scaler(enabled: bool) -> "_GradScaler":
        return _GradScaler(enabled=enabled)
    def _autocast_ctx(enabled: bool):
        return _autocast(enabled=enabled)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.aether_encoder_decoder import AETHEREncoder, AETHERDecoder
from training.losses import compute_layered_loss
from training.f0_losses import (
    compute_enhanced_f0_loss,
    compute_f0_variance_regularization,
    audio_f0_alignment_loss,      # ← 新增
)
 # , compute_f0_constraint_loss


# -- Global constants ------------------------------------------------------- #
SAMPLE_RATE = 16000
FRAME_HOP_SAMPLES = 160  # 10 ms @ 16 kHz


# -- Cached Mel helper ----------------------------------------------------- #
_MEL_CACHE: Dict[torch.device, torchaudio.transforms.MelSpectrogram] = {}

# -- Stage2 FARGAN-only training helpers ---------------------------------- #
def load_frozen_aether_models(checkpoint_path: str, device: torch.device, feature_dim: int = 36) -> Tuple[nn.Module, nn.Module]:
    """加载并冻结阶段一训练好的Aether编解码器"""
    from models.aether_encoder_decoder import AETHEREncoder, AETHERDecoder

    print(f"Loading Stage1 Aether models from: {checkpoint_path}")

    # 创建编解码器
    encoder = AETHEREncoder(feature_dim=feature_dim).to(device)
    decoder = AETHERDecoder(feature_dim=feature_dim).to(device)

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    elif 'encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder'])
    else:
        raise KeyError("No encoder state found in checkpoint")

    if 'decoder_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'decoder' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        raise KeyError("No decoder state found in checkpoint")

    # 冻结所有参数
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    # 设置为评估模式
    encoder.eval()
    decoder.eval()

    print(f"Frozen Aether models loaded successfully")
    print(f"  Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,} (frozen)")
    print(f"  Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,} (frozen)")

    return encoder, decoder

def create_independent_fargan_wavehead(device: torch.device) -> nn.Module:
    """创建独立的FARGAN波形头"""
    from models.fargan_decoder import FARGANDecoder

    fargan_wavehead = FARGANDecoder().to(device)

    # 确保所有参数都可训练
    for param in fargan_wavehead.parameters():
        param.requires_grad = True

    print(f"Independent FARGAN wavehead created")
    print(f"  FARGAN parameters: {sum(p.numel() for p in fargan_wavehead.parameters()):,} (trainable)")

    return fargan_wavehead

def train_stage2_fargan_only(
    frozen_encoder: nn.Module,
    frozen_decoder: nn.Module,
    fargan_wavehead: nn.Module,
    train_loader: Any,
    device: torch.device,
    args: Any,
    checkpoint_dir: Path
) -> Dict[str, float]:
    """
    阶段二独立FARGAN训练主函数
    使用冻结的Aether编解码器 + 独立训练FARGAN波形头
    完全复用train_fargan.py的训练配置
    """
    print("🔧 配置阶段二独立FARGAN训练")

    # 导入train_fargan.py的损失函数
    from training.fargan_losses import compute_fargan_training_loss, compute_fargan_original_style_loss

    # 创建优化器 - 复用train_fargan.py的配置
    optimizer = optim.AdamW(
        fargan_wavehead.parameters(),
        lr=args.fargan_learning_rate,
        weight_decay=1e-5,
        eps=1e-8,
        betas=(0.8, 0.95),
    )

    # 创建学习率调度器 - 复用train_fargan.py的配置
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: 1.0 / (1.0 + args.fargan_lr_decay * float(s))
    )

    # 训练配置
    num_epochs = 50  # 默认训练轮数
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    print(f"📊 训练配置:")
    print(f"   学习率: {args.fargan_learning_rate}")
    print(f"   衰减率: {args.fargan_lr_decay}")
    print(f"   原版损失轮数: {args.fargan_original_epochs}")
    print(f"   渐变轮数: {args.fargan_ramp_epochs}")
    print(f"   总训练轮数: {num_epochs}")
    print(f"   总训练步数: {total_steps}")

    # 训练模式
    fargan_wavehead.train()
    frozen_encoder.eval()  # 冻结编码器
    frozen_decoder.eval()  # 冻结解码器

    # 训练循环
    best_loss = float('inf')
    step_count = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (features, target_audio) in enumerate(train_loader):
            features = features.to(device)
            target_audio = target_audio.to(device)

            # === 阶段二独立FARGAN训练步骤 ===
            result = stage2_fargan_only_training_step(
                batch=(features, target_audio),
                frozen_encoder=frozen_encoder,
                frozen_decoder=frozen_decoder,
                fargan_wavehead=fargan_wavehead,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                step=step_count,
                args=args
            )

            epoch_loss += result['total_loss']
            epoch_steps += 1
            step_count += 1

            # 学习率调度
            scheduler.step()

            # 打印进度
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Step {step_count}: Loss={result['total_loss']:.6f}, LR={current_lr:.2e}")

        # Epoch结束统计
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('inf')
        print(f"Epoch {epoch + 1} 完成: 平均损失={avg_epoch_loss:.6f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_path = checkpoint_dir / "stage2_fargan_best.pt"
            torch.save({
                'epoch': epoch,
                'fargan_wavehead_state_dict': fargan_wavehead.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'step_count': step_count,
                'args': args
            }, checkpoint_path)
            print(f"💾 已保存最佳模型: {checkpoint_path}")

    print(f"🎉 阶段二独立FARGAN训练完成! 最佳损失: {best_loss:.6f}")
    return {'best_loss': best_loss, 'total_steps': step_count}


def stage2_fargan_only_training_step(
    batch: Tuple[torch.Tensor, torch.Tensor],
    frozen_encoder: nn.Module,
    frozen_decoder: nn.Module,
    fargan_wavehead: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    step: int,
    args: Any
) -> Dict[str, float]:
    """阶段二独立FARGAN训练步骤"""
    features, target_audio = batch
    features = features.to(device, non_blocking=True)
    target_audio = target_audio.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    loss_dict = {}

    try:
        # 使用冻结的编解码器提取FARGAN特征
        with torch.no_grad():
            encoded = frozen_encoder(features)
            decoded_features = frozen_decoder(encoded)  # [B, T, 36] FARGAN特征

        # 准备teacher forcing数据
        nb_pre_frames = 2
        pre = target_audio[..., :nb_pre_frames * 160]

        # 使用FARGAN波形头生成音频
        period, pred_audio = fargan_wavehead(decoded_features, pre=pre)
        pred_audio = pred_audio.squeeze(1)
        pred_audio = torch.cat([pre, pred_audio], dim=-1)

        # 对齐音频长度
        min_len = min(pred_audio.size(-1), target_audio.size(-1))
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]

        # 使用train_fargan.py的损失函数配置
        from training.fargan_losses import (
            compute_fargan_training_loss,
            compute_fargan_original_style_loss
        )

        # 损失函数选择和权重调度
        if args.fargan_original_epochs > 0:
            if epoch <= args.fargan_original_epochs:
                alpha = 0.0
            else:
                alpha = 1.0 if args.fargan_ramp_epochs <= 0 else min(
                    1.0, (epoch - args.fargan_original_epochs) / float(args.fargan_ramp_epochs)
                )

            # 原版损失
            orig_loss, orig_dict = compute_fargan_original_style_loss(
                pred_audio, target_audio, device=device,
                frame_size=160, focus_start=nb_pre_frames * 160,
            )

            # 训练损失
            comp_loss, comp_dict = compute_fargan_training_loss(
                pred_audio, target_audio, period, device=device
            )

            # 混合损失
            fargan_loss = (1.0 - alpha) * orig_loss + alpha * comp_loss
            loss_dict.update({f'orig_{k}': v for k, v in orig_dict.items()})
            loss_dict.update(comp_dict)
            loss_dict['alpha'] = alpha
        else:
            # 仅使用训练损失
            fargan_loss, loss_dict = compute_fargan_training_loss(
                pred_audio, target_audio, period, device=device
            )

        fargan_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(fargan_wavehead.parameters(), max_norm=0.5)

        optimizer.step()

        loss_dict['total_loss'] = fargan_loss.item()

    except Exception as e:
        print(f"Stage2 FARGAN training step failed: {e}")
        traceback.print_exc()
        loss_dict['total_loss'] = float('inf')

    return loss_dict
def _safe_mse(pred: torch.Tensor, target: torch.Tensor, step: int = 0) -> torch.Tensor:
    """安全的重建损失：前1000步用SmoothL1，后续用MSE。

    注意：先在原始张量上计算有限性掩码，再进行数值清洗，避免把非有限值
    变成大幅度有限值后参与损失，导致损失飙升。
    """
    # 在清洗前记录哪些位置是有限的
    orig_mask = torch.isfinite(pred) & torch.isfinite(target)
    # 数值清洗（不改变掩码）
    pred   = torch.nan_to_num(pred,   nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
    mask = orig_mask
    if not mask.any():
        return pred.new_zeros(())

    pred_masked = pred[mask]
    target_masked = target[mask]

    # 早期使用SmoothL1，更稳定；后期切回MSE
    if step < 1000:
        return F.smooth_l1_loss(pred_masked, target_masked, beta=0.5)
    else:
        diff = pred_masked - target_masked
        return (diff * diff).mean()
def _finite_scalar(x: torch.Tensor, name: str, step: int) -> torch.Tensor:
    """确保标量有限；否则打印并置零。"""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(float(x), device='cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.isfinite(x).all():
        try:
            val = float(x.detach().float().mean().cpu())
        except Exception:
            val = '<?>'
        print(f"⚠️ Step {step}: 非有限损失项 -> {name}={val} ; 已置零")
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x

def _sanitize_f0_losses(f0_losses: dict, step: int) -> dict:
    """逐项净化 f0 损失字典，返回同名新字典。"""
    clean = {}
    for k, v in f0_losses.items():
        if isinstance(v, torch.Tensor):
            if not torch.isfinite(v).all():
                try:
                    val = float(v.detach().float().mean().cpu())
                except Exception:
                    val = '<?>'
                print(f"⚠️ Step {step}: f0_losses['{k}'] 非有限={val} -> 置零")
                v = torch.zeros((), device=v.device, dtype=v.dtype)
            clean[k] = v
        else:
            clean[k] = v
    return clean

def _clean_tensor(x: torch.Tensor,
                  clip: float = 1e4) -> torch.Tensor:
    """
    数值清洗：把 NaN/±Inf 变成有限值，并做一次可选软限幅，避免极端值把损失拉爆。
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None and clip > 0:
        x = x.clamp(min=-clip, max=clip)
    return x


def _get_mel_transform(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    """Create or fetch a cached 80-bin Mel transform on the requested device."""
    transform = _MEL_CACHE.get(device)
    if transform is None:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            win_length=400,
            hop_length=FRAME_HOP_SAMPLES,
            f_min=50.0,
            f_max=7600.0,
            n_mels=80,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        _MEL_CACHE[device] = transform
    else:
        transform = transform.to(device)
    return transform


def _logmel_80(x: torch.Tensor) -> torch.Tensor:
    """
    Compute log-mel features pooled over time, producing shape [B, 80].
    Accepts [B, T], [B, 1, T] or [T].
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 3:
        x = x.squeeze(1)

    # 确保输入是2D波形 [B, T]
    if x.dim() != 2:
        raise ValueError(f"Expected 2D input [B, T], got {x.shape}")

    device = x.device
    mel_transform = _get_mel_transform(device)

    try:
        mel = mel_transform(x)  # [B, 80, Frames]

        # 验证mel输出维度
        if mel.dim() != 3 or mel.size(1) != 80:
            # 如果mel_transform输出不是[B, 80, T]格式，强制修复
            print(f"⚠️ Mel transform输出异常: {mel.shape}, 预期: [B, 80, T]")
            # 使用STFT后手动计算mel谱
            stft = torch.stft(x, n_fft=1024, hop_length=FRAME_HOP_SAMPLES,
                            win_length=400, center=True, return_complex=True)
            magnitude = torch.abs(stft)  # [B, freq_bins, time]

            # 如果频率bins不是513，裁剪或填充到合理范围
            if magnitude.size(1) != 513:
                magnitude = F.interpolate(magnitude.unsqueeze(1),
                                       size=(513, magnitude.size(-1)),
                                       mode='bilinear', align_corners=False).squeeze(1)

            # 使用现有的mel_transform的filter bank矩阵
            mel_filters = mel_transform.mel_scale.fb
            if mel_filters.size(0) != 80 or mel_filters.size(1) != magnitude.size(1):
                # 重新创建正确的mel filter bank
                from torchaudio.functional import melscale_fbanks
                mel_filters = melscale_fbanks(
                    n_freqs=magnitude.size(1), f_min=50.0, f_max=7600.0,
                    n_mels=80, sample_rate=SAMPLE_RATE
                ).to(device)

            mel = torch.matmul(mel_filters, magnitude)  # [B, 80, time]

        mel = (mel + 1e-8).log()
        mel_pooled = mel.mean(dim=-1)  # [B, 80]
        mel_norm = F.layer_norm(mel_pooled, mel_pooled.shape[-1:])
        return mel_norm

    except Exception as e:
        print(f"⚠️ Mel spectrogram计算失败: {e}")
        # 回退：返回零向量
        return torch.zeros(x.size(0), 80, device=device, dtype=x.dtype)

# 🚀 GPU优化的轻量MR-STFT Loss (预热专用)
# --- progressive_train_clean.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Sequence

# --- BEGIN PATCH: robust MR-STFT loss ----------------------------------------
class MRSTFTLoss(nn.Module):
    """
    兼容两种构造方式：
      1) MRSTFTLoss(cfgs=[(n_fft, hop, win), ...])
      2) MRSTFTLoss(fft_sizes=(...), hop_sizes=(...), win_sizes=(...), alpha_l1=..., alpha_mag=..., alpha_sc=...)
    并做数值稳定处理，避免 NaN/Inf。
    """
    def __init__(
        self,
        cfgs: Optional[Sequence[Tuple[int, int, int]]] = None,
        *,
        fft_sizes: Sequence[int] = (256, 512, 1024),
        hop_sizes: Sequence[int] = (64, 128, 256),
        win_sizes: Optional[Sequence[int]] = None,
        alpha_l1: float = 0.0,
        alpha_mag: float = 1.0,
        alpha_sc: float = 0.08,
        center: bool = False,
        power_mag: float = 1.0,
        lightweight: bool = False,  # 兼容多余命名参数
        **kwargs
    ):
        super().__init__()
        if cfgs is not None:
            # 允许 cfgs 中只给 (n_fft, hop)；未给 win 时默认 win=n_fft
            _cfgs = []
            for it in cfgs:
                if len(it) == 2:
                    n_fft, hop = int(it[0]), int(it[1])
                    _cfgs.append((n_fft, hop, n_fft))
                else:
                    n_fft, hop, win = int(it[0]), int(it[1]), int(it[2])
                    _cfgs.append((n_fft, hop, win))
            self.cfgs = _cfgs
        else:
            if win_sizes is None:
                win_sizes = fft_sizes
            assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
            self.cfgs = [(int(n), int(h), int(w)) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)]

        self.alpha_l1 = float(alpha_l1)
        self.alpha_mag = float(alpha_mag)
        self.alpha_sc = float(alpha_sc)
        self.center = bool(center)
        self.power_mag = float(power_mag)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, center=self.center, return_complex=True
        )
        mag = spec.abs().pow(self.power_mag)
        # 数值稳定：避免 log(0) 和除零
        return torch.clamp(mag, min=1e-7)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 输入通常是 [B,1,T]，压掉通道维
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred[:, 0, :]
        if target.dim() == 3 and target.size(1) == 1:
            target = target[:, 0, :]

        total = pred.new_tensor(0.0)
        count = 0

        for (n_fft, hop, win) in self.cfgs:
            if pred.shape[-1] < win or target.shape[-1] < win:
                continue
            p = self._stft_mag(pred, n_fft, hop, win)
            t = self._stft_mag(target, n_fft, hop, win)

            # 幅度 L1
            l_mag = F.l1_loss(p, t)

            # 谱收敛（按常见定义）： ||P-T||_F / ||T||_F
            num = torch.linalg.vector_norm(p - t, ord=2, dim=(-2, -1))
            den = torch.linalg.vector_norm(t,     ord=2, dim=(-2, -1)).clamp_min(1e-7)
            l_sc = (num / den).mean()

            total = total + (self.alpha_mag * l_mag + self.alpha_sc * l_sc)
            count += 1

        if count == 0:
            # 回退到时域 L1，确保梯度不中断
            total = F.l1_loss(pred, target)
        else:
            total = total / count

        if self.alpha_l1 > 0:
            total = total + self.alpha_l1 * F.l1_loss(pred, target)

        # 最终确保无 NaN/Inf
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)




from training.config import TrainConfig
from training.losses import l1_stft_loss, rate_loss
from training.fargan_losses import compute_fargan_original_style_loss
from utils.real_data_loader import create_aether_data_loader, AETHERRealDataset
from training.advanced_film_scheduler import AdvancedFiLMScheduler, create_film_parameter_groups
from utils.audio_validation_generator import integrate_audio_validation, export_validation_audio
from models.utils import extract_acoustic_priors
from utils.feature_spec import get_default_feature_spec
from models.feature_adapter import get_fargan_feature_spec


def get_feature_spec(feature_spec_type: str = "aether"):
    """根据特征规范类型获取对应的特征规范"""
    if feature_spec_type == "fargan":
        return get_fargan_feature_spec()
    else:
        return get_default_feature_spec()


@dataclass
class ProgressiveStage:
    """渐进式训练阶段配置"""
    name: str
    description: str

    steps: Optional[int] = None
    batches: Optional[int] = None
    epochs: Optional[float] = None

    use_film: bool = False
    use_moe: bool = False
    use_quantization: bool = False
    apply_channel: bool = False

    channel_type: str = "clean"
    layered_loss: bool = False

    film_warmup_steps: int = 0
    film_start_ratio: float = 1.0
    film_beta_scale_start: float = 1.0

    learning_rate: float = 2e-4
    lambda_rate: float = 0.0
    lambda_balance: float = 0.0
    lambda_cons: float = 0.0

    min_convergence_rate: float = 5.0
    max_final_loss: float = 2.0
    early_stop_loss: float = 0.01

    enable_audio_quality: bool = False
    min_snr_db: float = 5.0
    min_mel_cos: float = 0.85
    max_mel_l2: float = 0.15
    max_spectral_distortion: float = 0.65
    max_rms_delta_db: float = 3.0

    use_advanced_scheduler: bool = False

    min_final_film_ratio: float = 0.0
    max_recovery_events: int = 999
    max_spikes_last_50: int = 999
    wave_start_step: int = 0
    wave_full_start_step: int = 0
    wave_lowpass_weight: float = 0.5
    wave_full_weight: float = 1.0
    wave_lowpass_schedule: List[Tuple[int, float]] = field(default_factory=list)
    wave_full_schedule: List[Tuple[int, float]] = field(default_factory=list)
    train_wave_head_only: bool = False
    target_kbps: float = 0.0
    max_kbps_p90: float = 0.0
    preheat_mix_start_step: int = 1
    preheat_mix_end_step: int = 0
    preheat_chunk_frames: int = 0

    def calculate_steps(self, total_batches: int) -> int:
        """根据配置计算实际训练步数"""
        if self.steps is not None:
            return max(1, self.steps)
        elif self.batches is not None:
            return max(1, self.batches)
        elif self.epochs is not None:
            return max(1, int(self.epochs * total_batches))
        else:
            return total_batches


def configure_stage_model(encoder: nn.Module, decoder: nn.Module, stage: ProgressiveStage) -> None:
    """根据阶段配置模型状态"""
    encoder.set_stage("C" if stage.use_film or stage.use_moe else "A")


def _rg(t, digits=6):
    """安全梯度范围显示"""
    if t is None or not isinstance(t, torch.Tensor):
        return "N/A"
    try:
        if t.requires_grad and t.grad is not None:
            return f"{float(t.grad.min().cpu()):.{digits}e}~{float(t.grad.max().cpu()):.{digits}e}"
        return "no_grad"
    except:
        return "err"


def _grad_ok(modules: List[nn.Module], max_norm: float = 1000.0, debug: bool = False) -> bool:
    """检查梯度是否正常 - 仅检查NaN/Inf，允许大梯度"""
    try:
        for i, module in enumerate(modules):
            for name, param in module.named_parameters():
                if param.grad is not None:
                    # 只检查NaN和Inf，不限制梯度大小
                    if torch.isnan(param.grad).any():
                        if debug:
                            print(f"⚠️ NaN梯度在模块{i} {name}")
                        return False
                    if torch.isinf(param.grad).any():
                        if debug:
                            print(f"⚠️ Inf梯度在模块{i} {name}")
                        return False
        return True
    except Exception as e:
        if debug:
            print(f"⚠️ 梯度检查异常: {e}")
        return False


def compute_si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate SI-SNR for batch tensors [B, T]."""
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 3 and target.size(1) == 1:
        target = target[:, 0]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    pred_zm = pred - pred.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(pred_zm * target_zm, dim=-1, keepdim=True)
    target_power = torch.sum(target_zm.pow(2), dim=-1, keepdim=True) + eps
    proj = dot / target_power * target_zm
    noise = pred_zm - proj

    ratio = (proj.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def calculate_audio_quality(y_hat_feats, y_orig_feats, wave_head, original_audio, csi_dict=None):
    """计算音频质量指标 - 从原版提取的核心逻辑"""
    try:
        device = y_hat_feats.device
        eps = 1e-12

        # 生成重建音频
        frame_count = y_hat_feats.size(1) if y_hat_feats.dim() >= 2 else y_hat_feats.size(-1)
        target_len = int(frame_count * FRAME_HOP_SAMPLES)

        audio_aligned = None
        if original_audio is not None:
            audio_aligned = original_audio.clone()
            while audio_aligned.dim() > 2:
                audio_aligned = audio_aligned.squeeze(1)
            if audio_aligned.dim() == 1:
                audio_aligned = audio_aligned.unsqueeze(0)
            current_len = audio_aligned.size(-1)
            if current_len < target_len:
                pad = target_len - current_len
                audio_aligned = F.pad(audio_aligned, (0, pad))
            elif current_len > target_len:
                audio_aligned = audio_aligned[..., :target_len]
        else:
            return {'error': 'No original audio provided'}

        with torch.no_grad():
            if getattr(wave_head, '_is_exciter', False) and csi_dict is not None:
                try:
                    y_hat_audio = wave_head(y_hat_feats, target_len=target_len, csi_dict=csi_dict)
                except:
                    y_hat_audio = wave_head(y_hat_feats, target_len=target_len)
            else:
                y_hat_audio = wave_head(y_hat_feats, target_len=target_len)

        # 音频对齐（长度 + 时间偏移校正）
        y_target = audio_aligned
        min_len = min(y_hat_audio.size(-1), y_target.size(-1))
        y_hat_audio = y_hat_audio[..., :min_len]
        y_target = y_target[..., :min_len]

        # 通过互相关估计固定延迟并补偿（提升 SNR/SI-SNR 的可信度）
        def _xcorr_align(pred: torch.Tensor, ref: torch.Tensor, max_shift: int = 640):
            # 输入形状 [B, 1, T] 或 [B, T]
            if pred.dim() == 3:
                pred = pred.squeeze(1)
            if ref.dim() == 3:
                ref = ref.squeeze(1)
            B, T = pred.size(0), pred.size(-1)
            max_shift = min(max_shift, T - 1) if T > 1 else 0
            if max_shift <= 0:
                return pred, torch.zeros(B, dtype=torch.long, device=pred.device)

            # 归一化避免幅度主导
            pred_n = pred - pred.mean(dim=-1, keepdim=True)
            ref_n = ref - ref.mean(dim=-1, keepdim=True)

            # 通过有限范围滑动计算分段相关（效率足够）
            best_shifts = []
            aligned = []
            for b in range(B):
                p = pred_n[b]
                r = ref_n[b]
                best_score = -1e9
                best_k = 0
                for k in range(-max_shift, max_shift + 1):
                    if k < 0:
                        # pred 提前 -> 向右移
                        s = (p[-k:] * r[: T + k]).sum()
                    elif k > 0:
                        s = (p[: T - k] * r[k:]).sum()
                    else:
                        s = (p * r).sum()
                    if s > best_score:
                        best_score = s
                        best_k = k
                best_shifts.append(best_k)
                # 应用移位
                if best_k < 0:
                    # pred 向右移 |k|
                    pad = torch.zeros(-best_k, device=pred.device, dtype=pred.dtype)
                    aligned_pred = torch.cat([pad, pred[b, : T + best_k]], dim=0)
                elif best_k > 0:
                    aligned_pred = torch.cat([pred[b, best_k:], torch.zeros(best_k, device=pred.device, dtype=pred.dtype)], dim=0)
                else:
                    aligned_pred = pred[b]
                aligned.append(aligned_pred.unsqueeze(0))
            return torch.cat(aligned, dim=0), torch.tensor(best_shifts, device=pred.device)

        y_hat_audio_aligned, _shifts = _xcorr_align(y_hat_audio, y_target, max_shift=640)
        y_hat_audio = y_hat_audio_aligned

        # 计算增益对齐
        pred_energy = y_hat_audio.pow(2).mean(dim=-1, keepdim=True)
        tgt_energy = y_target.pow(2).mean(dim=-1, keepdim=True)
        gain = torch.sqrt((tgt_energy + eps) / (pred_energy + eps))
        gain = torch.clamp(gain, 0.1, 3.0)
        y_hat_aligned = gain * y_hat_audio

        # SNR计算
        signal_power = torch.mean(y_target.pow(2), dim=-1)
        noise_power = torch.mean((y_hat_aligned - y_target).pow(2), dim=-1)
        snr_linear = (signal_power + eps) / (noise_power + eps)
        snr_db = 10.0 * torch.log10(snr_linear + eps)

        # SI-SNR计算
        target_norm = y_target - torch.mean(y_target, dim=-1, keepdim=True)
        pred_norm = y_hat_aligned - torch.mean(y_hat_aligned, dim=-1, keepdim=True)

        # 投影
        dot_product = torch.sum(pred_norm * target_norm, dim=-1, keepdim=True)
        target_energy = torch.sum(target_norm.pow(2), dim=-1, keepdim=True)
        projection = (dot_product / (target_energy + eps)) * target_norm

        # SI-SNR
        signal_power_si = torch.sum(projection.pow(2), dim=-1)
        noise_power_si = torch.sum((pred_norm - projection).pow(2), dim=-1)
        si_snr_linear = (signal_power_si + eps) / (noise_power_si + eps)
        si_snr_db = 10.0 * torch.log10(si_snr_linear + eps)

        # 正确的80-bin Mel指标
        mel_pred = _logmel_80(y_hat_aligned)
        mel_target = _logmel_80(y_target)

        mel_cos = F.cosine_similarity(mel_pred, mel_target, dim=-1).mean()
        mel_l2 = F.mse_loss(mel_pred, mel_target)

        # PESQ-like score (0-5范围，基于 Mel 相似度与 SNR)
        pesq_like = 1.0 + 4.0 * torch.sigmoid(0.1 * snr_db.mean() + 2.0 * mel_cos)

        # Mel谱失真 (均方根误差)
        spectral_distortion = torch.sqrt(mel_l2 + eps)

        # 特征层面相关性
        y_hat_flat = y_hat_feats.flatten()
        y_orig_flat = y_orig_feats.flatten()
        feature_correlation = torch.corrcoef(torch.stack([y_hat_flat, y_orig_flat]))[0, 1]
        feature_correlation = torch.nan_to_num(feature_correlation, nan=0.0)

        return {
            'snr_db': float(snr_db.mean().cpu()),
            'si_snr_db': float(si_snr_db.mean().cpu()),
            'pesq_like': float(pesq_like.cpu()),
            'spectral_distortion': float(spectral_distortion.cpu()),
            'mel_cos': float(mel_cos.cpu()),
            'mel_l2': float(mel_l2.cpu()),
            'feature_correlation': float(feature_correlation.cpu()),
            'pred_rms_db': float(20 * torch.log10(torch.sqrt(torch.mean(y_hat_aligned.pow(2))) + eps).cpu()),
            'target_rms_db': float(20 * torch.log10(torch.sqrt(torch.mean(y_target.pow(2))) + eps).cpu())
        }

    except Exception as e:
        return {'error': str(e)}


def check_energy_anomaly(pred_audio, step, anomaly_state=None):
    """RMS能量哨兵 - 检测静音/爆噪异常"""
    if anomaly_state is None:
        anomaly_state = {'low_energy_count': 0, 'high_energy_count': 0, 'last_warning_step': -999}

    eps = 1e-12
    rms = torch.sqrt(torch.mean(pred_audio.pow(2)) + eps)
    rms_db = 20 * torch.log10(rms + eps)

    # 检测持续低能量 (静音)
    if rms_db < -35.0:
        anomaly_state['low_energy_count'] += 1
        if anomaly_state['low_energy_count'] >= 5 and (step - anomaly_state['last_warning_step']) > 50:
            print(f"⚠️ 步骤 {step}: 检测到持续低能量 (静音) - RMS: {rms_db:.1f}dB, 连续{anomaly_state['low_energy_count']}次")
            anomaly_state['last_warning_step'] = step
            return 'low_energy'
    else:
        anomaly_state['low_energy_count'] = max(0, anomaly_state['low_energy_count'] - 1)

    # 检测能量过高 (爆噪)
    if rms_db > -5.0:
        anomaly_state['high_energy_count'] += 1
        if anomaly_state['high_energy_count'] >= 3 and (step - anomaly_state['last_warning_step']) > 50:
            print(f"⚠️ 步骤 {step}: 检测到过高能量 (爆噪) - RMS: {rms_db:.1f}dB, 连续{anomaly_state['high_energy_count']}次")
            anomaly_state['last_warning_step'] = step
            return 'high_energy'
    else:
        anomaly_state['high_energy_count'] = max(0, anomaly_state['high_energy_count'] - 1)

    return None


def apply_energy_rescue(wave_head, anomaly_type, step):
    """能量异常自救机制"""
    if anomaly_type == 'low_energy':
        print(f"🔧 步骤 {step}: 应用低能量自救 - 添加小幅偏置")
        # 为最后一层添加小偏置
        for name, param in wave_head.named_parameters():
            if 'bias' in name and param.dim() == 1:
                with torch.no_grad():
                    param.data += 0.01 * torch.randn_like(param.data)
                break
        return True
    elif anomaly_type == 'high_energy':
        print(f"🔧 步骤 {step}: 应用高能量自救 - 权重抑制")
        # 轻微抑制权重
        for param in wave_head.parameters():
            if param.dim() > 1:  # 只处理权重矩阵
                with torch.no_grad():
                    param.data *= 0.95
                break
        return True
    return False


def monitor_f0_health(f0_pred, f0_target, step, threshold_corr=0.95, threshold_unique=0.01):
    """监控F0健康状态，自动调整权重"""
    try:
        # 计算相关性
        f0_pred_flat = f0_pred.flatten()
        f0_target_flat = f0_target.flatten()

        # 过滤掉无效值
        valid_mask = torch.isfinite(f0_pred_flat) & torch.isfinite(f0_target_flat)
        if valid_mask.sum() < 10:  # 至少需要10个有效值
            return False

        f0_pred_valid = f0_pred_flat[valid_mask]
        f0_target_valid = f0_target_flat[valid_mask]

        # 计算相关性
        try:
            f0_corr = torch.corrcoef(torch.stack([f0_pred_valid, f0_target_valid]))[0, 1].item()
            # 检查NaN并使用NumPy作为fallback
            if not torch.isfinite(torch.tensor(f0_corr)):
                import numpy as np
                pred_np = f0_pred_valid.detach().cpu().numpy()
                target_np = f0_target_valid.detach().cpu().numpy()
                f0_corr = float(np.corrcoef(pred_np, target_np)[0, 1])
        except:
            f0_corr = 0.0

        # 计算唯一值比例
        f0_rounded = f0_pred_valid.round(decimals=2)
        unique_ratio = len(torch.unique(f0_rounded)) / f0_rounded.numel()

        # 检测塌缩
        f0_collapsed = (f0_corr < threshold_corr) and (unique_ratio < threshold_unique)

        if f0_collapsed and step % 100 == 0:
            print(f"⚠️ Step {step}: F0 collapse detected! corr={f0_corr:.3f}, unique_ratio={unique_ratio:.3f}")
            return True  # 需要调整权重

        # 每500步报告健康状态
        if step % 500 == 0:
            print(f"📊 Step {step}: F0 health - corr={f0_corr:.3f}, unique_ratio={unique_ratio:.3f}")

        return False

    except Exception as e:
        print(f"⚠️ F0监控失败: {e}")
        return False


def train_progressive_stage(
    stage: ProgressiveStage,
    encoder: nn.Module,
    decoder: nn.Module,
    wave_head: nn.Module,
    wave_loss: nn.Module,
    train_loader: DataLoader,
    train_dataset: Optional[AETHERRealDataset],
    device: torch.device,
    checkpoint_dir: Path,
    current_stage_index: int,
    total_stages: int,
    checkpoint_every: int = 500,
    feature_spec_type: str = "fargan",
    decoder_type: str = "aether",
    disable_f0_loss: bool = False,
) -> Dict[str, Any]:
    """训练单个渐进阶段"""

    print(f"\n{'='*60}")
    print(f"🚀 开始阶段 {current_stage_index+1}/{total_stages}: {stage.name}")
    print(f"📝 描述: {stage.description}")
    print(f"{'='*60}")

    # 计算训练步数
    total_batches = len(train_loader)
    actual_steps = stage.calculate_steps(total_batches)
    planned_epochs = actual_steps / total_batches
    effective_batches_per_epoch = int(total_batches)

    print(f"最终配置: {planned_epochs:.2f} epochs, 每epoch {effective_batches_per_epoch} batches, 总步数 {actual_steps}, 学习率: {stage.learning_rate}")

    # 配置模型
    configure_stage_model(encoder, decoder, stage)

    # 设置参数梯度
    if getattr(stage, 'train_wave_head_only', False):
        # 冻结编码器
        for param in encoder.parameters():
            param.requires_grad = False

        # 处理解码器：FARGAN需要特殊处理
        if decoder_type == 'aether_fargan' and hasattr(decoder, 'fargan_core'):
            # AETHER-FARGAN: 冻结除FARGAN核心外的所有解码器参数
            for param in decoder.parameters():
                param.requires_grad = False
            # 只启用FARGAN核心参数
            for param in decoder.fargan_core.parameters():
                param.requires_grad = True
            print(f"FARGAN预热模式: 启用 fargan_core 参数，冻结其他解码器参数")
        else:
            # 其他解码器类型：冻结所有解码器参数
            for param in decoder.parameters():
                param.requires_grad = False

        # 启用波形头参数
        for param in wave_head.parameters():
            param.requires_grad = True
    else:
        for param in encoder.parameters():
            param.requires_grad = True
        for param in decoder.parameters():
            param.requires_grad = True
        for param in wave_head.parameters():
            param.requires_grad = True

    # 创建优化器 - 处理EmbeddedSynthHead的参数重复问题
    all_param_groups = []

    # 检查是否是EmbeddedSynthHead (零参数包装器)
    is_embedded_synth = hasattr(wave_head, 'decoder') and wave_head.decoder is decoder

    if stage.use_advanced_scheduler and stage.use_film:
        param_groups = create_film_parameter_groups(
            encoder, decoder, wave_head,
            base_lr=stage.learning_rate,
            film_lr_scale=2.0,
            decoder_lr_scale=0.8,
            wave_lr_scale=0.2
        )
        all_param_groups.extend(param_groups)
    else:
        if not getattr(stage, 'train_wave_head_only', False):
            encoder_params = [p for p in encoder.parameters() if p.requires_grad]
            decoder_params = [p for p in decoder.parameters() if p.requires_grad]
            if encoder_params:
                all_param_groups.append({'params': encoder_params, 'lr': stage.learning_rate, 'name': 'encoder'})
            if decoder_params:
                all_param_groups.append({'params': decoder_params, 'lr': stage.learning_rate * 0.8, 'name': 'decoder'})
        else:
            # train_wave_head_only=True的情况
            if decoder_type == 'aether_fargan':
                # AETHER-FARGAN: 检查wave_head包装器中的解码器
                actual_decoder = wave_head.decoder if hasattr(wave_head, 'decoder') else decoder
                if hasattr(actual_decoder, 'fargan_core'):
                    # 只训练FARGAN核心合成器
                    fargan_core_params = [p for p in actual_decoder.fargan_core.parameters() if p.requires_grad]
                    if fargan_core_params:
                        all_param_groups.append({'params': fargan_core_params, 'lr': stage.learning_rate, 'name': 'fargan_core'})
                        print(f"FARGAN预热模式: 只训练 fargan_core，参数数量: {len(fargan_core_params)}")
                    else:
                        # 回退：训练整个wave_head
                        wave_head_params = [p for p in wave_head.parameters() if p.requires_grad]
                        if wave_head_params:
                            all_param_groups.append({'params': wave_head_params, 'lr': stage.learning_rate, 'name': 'wave_head'})
                            print(f"回退: 训练整个 wave_head，参数数量: {len(wave_head_params)}")
                else:
                    # 回退：训练整个wave_head
                    wave_head_params = [p for p in wave_head.parameters() if p.requires_grad]
                    if wave_head_params:
                        all_param_groups.append({'params': wave_head_params, 'lr': stage.learning_rate, 'name': 'wave_head'})
                        print(f"回退: 训练整个 wave_head，参数数量: {len(wave_head_params)}")
            elif is_embedded_synth:
                # EmbeddedSynthHead: 只训练decoder的synth部分
                synth_params = [p for n, p in decoder.named_parameters() if 'synth' in n and p.requires_grad]
                if synth_params:
                    all_param_groups.append({'params': synth_params, 'lr': stage.learning_rate, 'name': 'decoder_synth'})
            else:
                # 独立的wave_head
                wave_head_params = [p for p in wave_head.parameters() if p.requires_grad]
                if wave_head_params:
                    all_param_groups.append({'params': wave_head_params, 'lr': stage.learning_rate, 'name': 'wave_head'})

        # 如果不是embedded synth或者不是wave_head_only模式，添加独立的wave_head参数
        if not is_embedded_synth and not getattr(stage, 'train_wave_head_only', False):
            wave_head_params = [p for p in wave_head.parameters() if p.requires_grad]
            if wave_head_params:
                all_param_groups.append({'params': wave_head_params, 'lr': stage.learning_rate * 0.2, 'name': 'wave_head'})

    if not all_param_groups:
        raise ValueError(f"阶段 {stage.name}: 没有可训练的参数")

    # 调试信息：检查参数组
    print(f"🔧 优化器参数组:")
    for i, group in enumerate(all_param_groups):
        name = group.get('name', f'group_{i}')
        param_count = len(group['params'])
        lr = group['lr']
        print(f"  {name}: {param_count} 参数, lr={lr:.2e}")

    # 检查参数重复
    all_params = []
    for group in all_param_groups:
        all_params.extend(group['params'])
    unique_params = set(id(p) for p in all_params)
    if len(all_params) != len(unique_params):
        print(f"⚠️ 检测到参数重复: 总参数{len(all_params)}, 唯一参数{len(unique_params)}")
        # 移除重复参数
        seen_params = set()
        for group in all_param_groups:
            unique_group_params = []
            for p in group['params']:
                if id(p) not in seen_params:
                    unique_group_params.append(p)
                    seen_params.add(id(p))
            group['params'] = unique_group_params
        print("✅ 已移除重复参数")

    optimizer = optim.AdamW(all_param_groups, weight_decay=1e-6)
    scaler = _create_grad_scaler(enabled=(device.type == 'cuda'))

    # 学习率调度器：ramp 期间使用 LambdaLR，ramp 结束后可切换到 ReduceLROnPlateau
    from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
    ramp_steps = 5000 if decoder_type == 'aether_fargan' else actual_steps  # FARGAN模式使用ramp调度

    # 初始阶段使用 LambdaLR（余弦退火）
    lr_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.5 * (1 + math.cos(math.pi * step / ramp_steps)) if step < ramp_steps else 0.1
    )

    # 用于 ramp 结束后的 Plateau 调度器
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
    )

    scheduler_switched = False  # 标记是否已切换到 plateau 调度器

    # 高级FiLM调度器
    film_scheduler = None
    if stage.use_advanced_scheduler and stage.use_film:
        film_scheduler = AdvancedFiLMScheduler(
            encoder=encoder,
            total_steps=actual_steps,
            warmup_steps=stage.film_warmup_steps,
            start_ratio=stage.film_start_ratio,
            beta_scale_start=stage.film_beta_scale_start
        )

    # 训练循环
    encoder.train()
    decoder.train()
    wave_head.train()

    # Initialize FARGAN loss logging variable
    fargan_loss_logs = None

    # 预热阶段先关闭潜在量化，避免早期量化噪声干扰F0
    _orig_quant_flag = getattr(encoder, 'quantize_latent', None)
    if stage.name == 'wave_preheat' and _orig_quant_flag is not None:
        try:
            encoder.quantize_latent = False
            print("🔧 预热阶段: 已暂时关闭latent量化")
        except Exception:
            pass

    best_loss = float('inf')
    convergence_losses = []
    step = 0

    # 波形预热相关参数
    preheat_gain = nn.Parameter(torch.tensor(3.0, device=device))
    preheat_scale = nn.Parameter(torch.tensor(0.1, device=device))

    # RMS能量异常监控状态
    energy_anomaly_state = {'low_energy_count': 0, 'high_energy_count': 0, 'last_warning_step': -999}

    # 启用数值异常检测 (仅在调试时)
    # torch.autograd.set_detect_anomaly(True)  # 暂时禁用以提高性能

    # 无限迭代数据加载器
    def batch_gen():
        while True:
            for batch in train_loader:
                yield batch

    batch_iter = batch_gen()

    # F0健康告警冷却窗口（触发后在一定步数内提升F0权重/门控波形损失）
    f0_alert_until_step = -1

    # 🚀 超轻量预热专用波形损失 - 大幅减少计算压力
    wave_loss_fast = MRSTFTLoss(
        lightweight=False,
        fft_sizes=(256, 512, 1024),              # 如够算力可扩到 (256,512,1024,2048)
        hop_sizes=(64, 128, 256),
        win_sizes=(256, 512, 1024),
        alpha_l1=2.0,                             # 从 4.0 降到 2.0
        alpha_mag=1.0,
        alpha_sc=0.08                             # 打开谱收敛
    ).to(device)

    if stage.name == "wave_preheat":
        wave_loss_fast = MRSTFTLoss(
            lightweight=False,
            fft_sizes=(256, 512, 1024, 2048),
            hop_sizes=(64, 128, 256, 512),
            win_sizes=(256, 512, 1024, 2048),
            alpha_l1=2.0,      # 从 4.0 下调，给频域让路
            alpha_mag=1.0,
            alpha_sc=0.08      # 打开谱收敛
        ).to(device)


    for step in range(1, actual_steps + 1):
        batch = next(batch_iter)
        optimizer.zero_grad()

        # 数据预处理 - 使用正确的键名
        x_gpu = batch['x'].to(device, non_blocking=True)  # 输入特征
        y_gpu = batch['y'].to(device, non_blocking=True)  # 目标特征
        csi_dict = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.get('csi', {}).items()}

        x = torch.nan_to_num(x_gpu, nan=0.0, posinf=1e4, neginf=-1e4)
        y = torch.nan_to_num(y_gpu, nan=0.0, posinf=1e4, neginf=-1e4)
        original_audio = batch.get('audio')
        if original_audio is not None:
            original_audio = original_audio.to(device, non_blocking=True)

        # 为双路训练保存原始目标特征
        y_original = y.clone()

        # 当前epoch和步数信息
        current_epoch = (step - 1) // effective_batches_per_epoch + 1
        epoch_step = (step - 1) % effective_batches_per_epoch + 1

        # 随机裁剪(仅预热阶段)
        if stage.name == "wave_preheat" and getattr(stage, 'preheat_chunk_frames', 0) > 0:
            chunk_len = stage.preheat_chunk_frames
            seq_len = x.size(1)
            if seq_len > chunk_len:
                start_idx = random.randint(0, seq_len - chunk_len)
                x = x[:, start_idx:start_idx + chunk_len, :]
                y = y[:, start_idx:start_idx + chunk_len, :]
                # 也对原始目标特征应用相同的裁剪
                y_original = y_original[:, start_idx:start_idx + chunk_len, :]
                if original_audio is not None:
                    audio_start = start_idx * FRAME_HOP_SAMPLES
                    audio_end = audio_start + chunk_len * FRAME_HOP_SAMPLES
                    original_audio = original_audio[:, audio_start:audio_end]

        # 混合精度设置
        s1_t_global = int(getattr(stage, 'preheat_mix_end_step', 0)) if hasattr(stage, 'preheat_mix_end_step') else 0
        use_amp_step = (device.type == 'cuda') and not (stage.name == "wave_preheat" and step <= s1_t_global + 300)

        with _autocast_ctx(enabled=use_amp_step):
            # 编码
            x_fp32 = x.to(torch.float32)
            try:
                with _autocast_ctx(enabled=False):
                    z, enc_logs = encoder(x_fp32, csi_dict=csi_dict, inference=False)

                # 🔍 调试信息：输入特征统计
                if step <= 10 or step % 200 == 0:
                    print(f"📊 Step {step} 输入特征统计:")
                    print(f"   编码输出z: shape={z.shape}, mean={z.mean().item():.3f}, std={z.std().item():.3f}")
                    if not disable_f0_loss:
                        spec = get_feature_spec(feature_spec_type)
                        f0_slice = spec.get_feature_slice('f0') if hasattr(spec, 'get_feature_slice') else slice(18, 19)
                        f0_input = x[:, :, f0_slice].flatten()
                        print(f"   F0输入: mean={f0_input.mean().item():.3f}, std={f0_input.std().item():.3f}, range=[{f0_input.min().item():.3f}, {f0_input.max().item():.3f}]")

            except Exception as e:
                print(f"❌ 编码失败 at step {step}: {e}")
                continue

            # 解码
            try:
                # 让解码器也感知声学先验（与编码器对齐）
                csi_dec = dict(csi_dict)
                try:
                    csi_dec["acoustic_priors"] = extract_acoustic_priors(x).detach()
                except Exception:
                    csi_dec = csi_dict
                y_hat_raw = decoder(z, csi_dict=csi_dec)
                # 计算重建损失时需要基于原始输出的有限性掩码，避免把非有限值清洗后参与损失
                y_hat = _clean_tensor(y_hat_raw)
                y     = _clean_tensor(y)
                # 🔍 调试信息：解码输出统计
                if step <= 10 or step % 200 == 0:
                    if not disable_f0_loss:
                        spec = get_feature_spec(feature_spec_type)
                        f0_slice = spec.get_feature_slice('f0') if hasattr(spec, 'get_feature_slice') else slice(18, 19)
                        f0_pred = y_hat[:, :, f0_slice].flatten()
                        f0_target = y[:, :, f0_slice].flatten()
                        print(f"   F0预测: mean={f0_pred.mean().item():.3f}, std={f0_pred.std().item():.3f}, range=[{f0_pred.min().item():.3f}, {f0_pred.max().item():.3f}]")
                        print(f"   F0目标: mean={f0_target.mean().item():.3f}, std={f0_target.std().item():.3f}, range=[{f0_target.min().item():.3f}, {f0_target.max().item():.3f}]")

            except Exception as e:
                print(f"❌ 解码失败 at step {step}: {e}")
                continue

            # ✅ 主损失：特征域重建 [B,T,feature_dims] vs [B,T,feature_dims]
            if disable_f0_loss:
                try:
                    spec = get_feature_spec(feature_spec_type)
                    if feature_spec_type == "fargan":
                        f0_sl = spec.get_feature_slices().get('dnn_pitch', slice(0,0))
                    else:
                        f0_sl = spec.get_feature_slice('f0') if hasattr(spec, 'get_feature_slice') else slice(20, 21)
                except Exception:
                    f0_sl = slice(0,0)

                def _drop_slice(t: torch.Tensor, s: slice) -> torch.Tensor:
                    if (s.stop - s.start) <= 0:
                        return t
                    left = t[..., :s.start]
                    right = t[..., s.stop:]
                    return torch.cat([left, right], dim=-1)

                recon_loss = _finite_scalar(_safe_mse(_drop_slice(y_hat, f0_sl), _drop_slice(y, f0_sl), step), "recon_mse_nof0", step)
            else:
                recon_loss = _finite_scalar(_safe_mse(y_hat, y, step), "recon_mse", step)

            # 🔍 调试信息：损失分解分析（可禁用F0）
            if step <= 10 or step % 200 == 0:
                if not disable_f0_loss:
                    try:
                        spec = get_feature_spec(feature_spec_type)
                        if feature_spec_type == "fargan":
                            slices = spec.get_feature_slices()
                            f0_slice = slices.get('dnn_pitch', slice(18, 19))
                        else:
                            f0_slice = spec.get_feature_slice('f0') if hasattr(spec, 'get_feature_slice') else slice(18, 19)
                        f0_loss = F.mse_loss(y_hat[:, :, f0_slice], y[:, :, f0_slice])
                        other_features_loss = F.mse_loss(
                            y_hat[:, :, :f0_slice.start] if f0_slice.start > 0 else torch.empty(0, device=device),
                            y[:, :, :f0_slice.start] if f0_slice.start > 0 else torch.empty(0, device=device)
                        ) if f0_slice.start > 0 else 0.0
                        print(f"💡 Step {step} 损失分解:")
                        print(f"   总重建损失: {recon_loss.item():.6f}")
                        print(f"   F0特征损失: {f0_loss.item():.6f}")
                        print(f"   其他特征损失: {other_features_loss if isinstance(other_features_loss, float) else other_features_loss.item():.6f}")
                    except Exception as e:
                        print(f"⚠️ F0损失计算失败: {e}")
                else:
                    print(f"💡 Step {step} 损失分解: 已禁用F0; 总重建损失={recon_loss.item():.6f}")

            # === 新增：分层损失（三阶段策略） ===
            # 🔧 分离损失优化：延迟分层损失，给F0分支稳定时间
            if getattr(stage, 'layered_loss', False) and step > 3000:  # 前3000步不使用分层损失
                layered_loss, ld, stage_name = compute_layered_loss(
                    y_hat, y, step, feature_spec_type, disable_f0=disable_f0_loss
                )
                recon_loss = recon_loss + layered_loss
                if step % 100 == 0:
                    print(f"[{step}] stage={stage_name} layered_loss={layered_loss.item():.4f}")
            elif getattr(stage, 'layered_loss', False) and step <= 3000:
                if step % 500 == 0:
                    print(f"[{step}] 分层损失已禁用，专注F0稳定训练")

            # 🎯 Enhanced F0 Loss: 可禁用
            f0_loss_applied = False
            try:
                if disable_f0_loss:
                    raise RuntimeError("F0 disabled")
                f0_losses = compute_enhanced_f0_loss(y, y_hat, spec=get_feature_spec(feature_spec_type))
                f0_losses = _sanitize_f0_losses(f0_losses, step)

                # === 形状对齐增强：针对有声帧的相关性与斜率一致性 ===
                try:
                    _spec_f0 = get_feature_spec(feature_spec_type)
                    f0_tgt = _spec_f0.extract_feature(y, 'f0') if hasattr(_spec_f0, 'extract_feature') else y[:, :, 18:19]  # [B,T,1] FARGAN: dnn_pitch
                    f0_hat = _spec_f0.extract_feature(y_hat, 'f0') if hasattr(_spec_f0, 'extract_feature') else y_hat[:, :, 18:19]    # [B,T,1]
                    # FARGAN没有独立的voicing特征，使用dnn_pitch > threshold作为voicing
                    if hasattr(_spec_f0, 'extract_feature'):
                        voi_tgt = _spec_f0.extract_feature(y, 'voicing')  # [B,T,1]
                        voi_hat = _spec_f0.extract_feature(y_hat, 'voicing')
                    else:
                        # FARGAN: 从dnn_pitch推导voicing
                        voi_tgt = (y[:, :, 18:19] > -1.0).float()      # [B,T,1]
                        voi_hat = (y_hat[:, :, 18:19] > -1.0).float()  # [B,T,1]

                    # 掩码硬化（0.3/0.7 阈值）+ 形态学闭运算平滑 + 高阈值滞回种子
                    mask_lo = ((voi_tgt > 0.3) & (voi_hat > 0.3)).float()  # 宽松有声
                    mask_hi = ((voi_tgt > 0.7) & (voi_hat > 0.7)).float()  # 置信有声
                    # 闭运算：先膨胀再腐蚀，去掉小孔洞与短孤立段
                    m = mask_lo.transpose(1, 2)  # [B,1,T]
                    m = F.max_pool1d(m, kernel_size=3, stride=1, padding=1)           # 膨胀
                    m = -F.max_pool1d(-m, kernel_size=3, stride=1, padding=1)         # 腐蚀（最小池化）
                    mask_closed = (m.transpose(1, 2) > 0.5).float()                    # [B,T,1]
                    # 高阈值滞回：仅保留与高阈值邻域相连区域
                    seed = F.max_pool1d(mask_hi.transpose(1, 2), kernel_size=3, stride=1, padding=1)
                    seed = (seed.transpose(1, 2) > 0.0).float()  # [B,T,1]
                    mask = (mask_closed > 0.5) & (seed > 0)
                    mask = mask.float()

                    # 小窗xcorr对齐（±2帧）提高F0 loss时序一致性
                    def _best_shift(a, b, max_k=2):
                        # a,b: [T]
                        best_k, best_s = 0, -1e9
                        for k in range(-max_k, max_k + 1):
                            if k < 0:
                                s = torch.sum(a[-k:] * b[: a.shape[0] + k])
                            elif k > 0:
                                s = torch.sum(a[: a.shape[0] - k] * b[k:])
                            else:
                                s = torch.sum(a * b)
                            if s > best_s:
                                best_s, best_k = float(s), k
                        return best_k

                    # 逐样本对齐（批量较小时开销可接受）
                    f0_tgt_aligned = []
                    f0_hat_aligned = []
                    mask_aligned = []
                    B, T = f0_tgt.shape[0], f0_tgt.shape[1]
                    for b in range(B):
                        a = f0_hat[b, :, 0]
                        bvec = f0_tgt[b, :, 0]
                        k = _best_shift(a, bvec)
                        if k < 0:
                            ah = a[-k:]
                            bh = bvec[: T + k]
                            mh = mask[b, : T + k, 0]
                        elif k > 0:
                            ah = a[: T - k]
                            bh = bvec[k:]
                            mh = mask[b, k:, 0]
                        else:
                            ah = a
                            bh = bvec
                            mh = mask[b, :, 0]
                        # 对齐后统一长度
                        L = min(ah.shape[0], bh.shape[0])
                        f0_hat_aligned.append(ah[:L])
                        f0_tgt_aligned.append(bh[:L])
                        mask_aligned.append(mh[:L])

                    f0_hat_cat = torch.cat([t.unsqueeze(0) for t in f0_hat_aligned], dim=0)
                    f0_tgt_cat = torch.cat([t.unsqueeze(0) for t in f0_tgt_aligned], dim=0)
                    mask_cat = torch.cat([t.unsqueeze(0) for t in mask_aligned], dim=0)

                    # 仅在有声掩码内计算相关性与斜率
                    eps = 1e-5
                    def _masked_norm(z, m):
                        z_m = z * m
                        mu = (z_m.sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + eps))
                        zc = z_m - mu
                        var = (zc.pow(2) * m).sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + eps)
                        std = (var + eps).sqrt()
                        return (zc / std), m

                    valid_frames = int(mask_cat.sum().item())
                    if valid_frames >= 64:  # 门槛：至少 64 个有声帧
                        x, m_used = _masked_norm(f0_hat_cat, mask_cat)
                        y_, _     = _masked_norm(f0_tgt_cat, mask_cat)
                        L_corr = 1.0 - ((x * y_) * m_used).sum(dim=1) / (m_used.sum(dim=1) + eps)
                        L_corr = torch.nan_to_num(L_corr, nan=0.0, posinf=1.0, neginf=1.0).mean()

                        dx = torch.diff(f0_hat_cat, dim=1)
                        dy = torch.diff(f0_tgt_cat, dim=1)
                        mm = (mask_cat[:, 1:] * mask_cat[:, :-1])
                        denom = (mm.sum(dim=1) + eps)
                        L_delta = (torch.abs(dx - dy) * mm).sum(dim=1) / denom
                        L_delta = torch.nan_to_num(L_delta, nan=0.0, posinf=1.0, neginf=1.0).mean()
                    else:
                        L_corr  = torch.tensor(0.0, device=y_hat.device)
                        L_delta = torch.tensor(0.0, device=y_hat.device)


                    # 以小权重加入F0整体损失（仅增益，不替代原有项）
                    f0_losses_extra = 0.2 * L_corr + 0.1 * L_delta
                except Exception:
                    f0_losses_extra = 0.0

                if (not disable_f0_loss) and stage.name == "wave_preheat":
                    # 预热阶段：较强F0约束 + 方差正则 + 形状一致性
                    # f0_weight 从2.0线性回落至1.0（接近preheat_mix_end_step）
                    f0_weight = 2.0
                    try:
                        s2 = int(getattr(stage, 'preheat_mix_end_step', 0) or 0)
                        if s2 > 0:
                            decay_start = int(0.5 * s2)
                            if step >= decay_start:
                                p = min(1.0, (step - decay_start) / max(1, s2 - decay_start))
                                f0_weight = 2.0 - p * 1.0  # 2.0 -> 1.0
                    except Exception:
                        pass
                    var_reg = 0.0
                    try:
                        _spec_var = get_feature_spec(feature_spec_type)
                        _recon_f0 = _spec_var.extract_feature(y_hat, 'f0') if hasattr(_spec_var, 'extract_feature') else y_hat[:, :, 18:19]
                        var_reg = compute_f0_variance_regularization(_recon_f0, var_floor=0.02, weight=1.5)
                    except Exception:
                        pass
                    # F0健康闭环：告警期间临时提升F0权重
                    f0_boost = 2.0 if step <= f0_alert_until_step else 1.0
                    total_f0 = (
                        _finite_scalar((f0_boost * f0_weight) * f0_losses['total_f0_loss'], "f0_total", step)
                        + _finite_scalar(f0_losses['voi_loss'], "voi_loss", step)
                        + _finite_scalar(var_reg if isinstance(var_reg, torch.Tensor) else torch.tensor(var_reg, device=y_hat.device), "f0_var_reg", step)
                    )
                    if isinstance(f0_losses_extra, torch.Tensor):
                        total_f0 = total_f0 + _finite_scalar((f0_boost * f0_weight) * f0_losses_extra, "f0_shape_extra", step)
                    if step <= 10 or step % 200 == 0:
                        print(f"💡 Step {step} 预热阶段: 增强F0损失 f0_w={(f0_boost * f0_weight):.2f} total={total_f0.item():.6f}")
                else:
                    # 正常阶段：完整F0损失（加入前期缓启，避免早期主导）
                    f0_weight = 0.1 + 1.4 * min(1.0, step / 1000.0)  # 0.1 → 1.5 in first 1k steps
                    f0_boost = 2.0 if step <= f0_alert_until_step else 1.0
                    total_f0 = (f0_boost * f0_weight) * f0_losses['total_f0_loss'] + f0_losses['voi_loss']
                    if isinstance(f0_losses_extra, torch.Tensor):
                        total_f0 = total_f0 + (f0_boost * f0_weight) * f0_losses_extra

                # 🧪 移除约束损失，测试原始代码稳定性
                # constraint_loss = compute_f0_constraint_loss(pred_f0, target_f0, weight=0.3)

                recon_loss = recon_loss + total_f0
                f0_loss_applied = True

                if step % 100 == 0 and stage.name != "wave_preheat":
                    print(
                        f"[{step}] F0 losses: base={f0_losses['f0_base'].item():.4f} "
                        f"slope={f0_losses['f0_slope'].item():.4f} "
                        f"mean={f0_losses['f0_mean'].item():.4f} "
                        f"std={f0_losses['f0_std'].item():.4f} "
                        f"voi={f0_losses['voi_loss'].item():.4f} "
                        f"core={f0_losses['f0_core'].item():.4f}"
                    )

                # 🔍 F0健康监控
                if step % 50 == 0:
                    feature_spec = get_feature_spec(feature_spec_type)
                    try:
                        orig_f0 = feature_spec.extract_feature(y, 'f0') if hasattr(feature_spec, 'extract_feature') else y[:, :, 18:19]
                        recon_f0 = feature_spec.extract_feature(y_hat, 'f0') if hasattr(feature_spec, 'extract_feature') else y_hat[:, :, 18:19]
                        needs_adjustment = monitor_f0_health(recon_f0, orig_f0, step)
                        if needs_adjustment:
                            print(f"🚨 Step {step}: F0塌缩检测到 -> 临时提升F0权重 & 门控波形损失")
                            f0_alert_until_step = max(f0_alert_until_step, step + 150)
                        # 仅对有声帧增加最小方差正则，反塌陷
                        try:
                            voi = feature_spec.extract_feature(y_hat, 'voicing')
                            voi_mask = (voi > 0.6).float()
                            # 按样本统计有声帧std，目标≥0.25
                            std_per_utt = ((recon_f0 * voi_mask).std(dim=1, unbiased=False) + 1e-6)
                            var_floor_pen = (0.25 - std_per_utt).clamp_min(0.0).mean()
                            recon_loss = recon_loss + 0.5 * var_floor_pen
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception as e:
                if not disable_f0_loss:
                    print(f"⚠️ F0损失计算失败: {e}")
                    if step <= 10 or step % 200 == 0:
                        print(f"⚠️ Step {step}: F0损失被跳过 (计算失败)")

            # 轻量反静态正则：抑制整段常数特征（避免F0/参数塌缩）
            try:
                # 时间维度上方差过小时施加惩罚 - 增强权重
                var_t = y_hat.float().var(dim=1).mean()
                anti_static_loss = _finite_scalar((1.0 / (var_t + 1e-3)).clamp(max=1e3), "anti_static", step)
                anti_static_weight = 2e-3 if stage.name == "wave_preheat" else 1e-4  # 预热期更强抑制静态
                recon_loss = recon_loss + anti_static_weight * anti_static_loss
            except Exception:
                pass

            # 码率损失
            rate_loss_val = _finite_scalar(rate_loss(enc_logs.get('latent_continuous', z), stage.lambda_rate), "rate_loss", step)

            # ✅ 辅助损失：波形域验收/引导 (仅当有音频时)
            # 流程：decoder(z) -> [B,T,48] -> wave_head([B,T,48]) -> [B,1,T_audio]
            wave_loss_val = torch.tensor(0.0, device=device)

            # 全局 MR-STFT 权重增强策略
            if decoder_type == 'aether_fargan':
                ramp_steps = 5000
                if step > ramp_steps * 0.6:
                    if step <= ramp_steps * 0.8:
                        mr_stft_boost = 1.5
                    else:
                        mr_stft_boost = 2.0  # 综合期维持强权重
                else:
                    mr_stft_boost = 1.0
            else:
                mr_stft_boost = 1.0  # 默认不加强
            if stage.enable_audio_quality and original_audio is not None:
                # 预热阶段的混合策略 - 原始特征 ↔ 重建特征渐进过渡
                if stage.name == "wave_preheat":
                    # 统一TF时间表：从s1到s2线性衰减，不fallback到总步数一半
                    s1 = int(getattr(stage, 'preheat_mix_start_step', 0) or 0)
                    s2 = int(getattr(stage, 'preheat_mix_end_step', 0) or 0)
                    if s2 <= s1:
                        teacher_ratio = 1.0 if step <= s1 else 0.0
                    else:
                        if step <= s1:
                            teacher_ratio = 1.0
                        elif step >= s2:
                            teacher_ratio = 0.0
                        else:
                            prog = (step - s1) / max(1, s2 - s1)
                            teacher_ratio = 1.0 - prog
                    teacher_ratio = float(torch.clamp(torch.tensor(teacher_ratio, device=device), 0.0, 1.0).item())
                    mix_ratio = 1.0 - teacher_ratio

                    # 双路特征混合 - 从原始目标特征渐变到编解码器重建特征
                    y_hat_features = (1.0 - mix_ratio) * y_original + mix_ratio * y_hat
                    preheat_mix_ratio = mix_ratio
                else:
                    y_hat_features = y_hat
                    preheat_mix_ratio = 1.0

                # 波形生成 - 确保target_len与chunked特征对齐
                feature_frames = y_hat_features.size(1)
                target_len = feature_frames * FRAME_HOP_SAMPLES

                if original_audio is not None:
                    while original_audio.dim() > 2:
                        original_audio = original_audio.squeeze(1)
                    if original_audio.dim() == 1:
                        original_audio = original_audio.unsqueeze(0)
                    current_len = original_audio.size(-1)
                    if current_len < target_len:
                        pad = target_len - current_len
                        original_audio = F.pad(original_audio, (0, pad))
                    elif current_len > target_len:
                        original_audio = original_audio[..., :target_len]

                # 准备目标音频张量供 FARGAN teacher forcing 使用（统一为 [B, L]）
                y_target_audio = None
                if original_audio is not None:
                    y_target_audio = original_audio.squeeze(1) if original_audio.dim() == 3 else original_audio
                    if stage.name == "wave_preheat" and getattr(stage, 'preheat_chunk_frames', 0) > 0:
                        chunk_len = stage.preheat_chunk_frames
                        expected_audio_len = chunk_len * FRAME_HOP_SAMPLES
                        if y_target_audio.size(-1) > expected_audio_len:
                            y_target_audio = y_target_audio[..., :expected_audio_len]

                detach_wave = getattr(stage, 'train_wave_head_only', False)

                # === 通道级TF喂给波形头：预热期强制使用真值F0，其他通道按mix混合 ===
                y_in = y_hat_features
                if stage.name == "wave_preheat":
                    try:
                        spec = get_feature_spec(feature_spec_type)
                        f0_slice = spec.get_feature_slice('f0') if hasattr(spec, 'get_feature_slice') else slice(18, 19)
                        y_in = y_hat_features.clone()
                        y_in[..., f0_slice] = y_original[..., f0_slice]
                    except Exception:
                        pass

                y_in = torch.nan_to_num(y_in, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-6.0, 6.0)

                # 波形生成(禁用早期AMP避免下溢)
                use_amp_wave = (device.type == 'cuda') and not (stage.name == "wave_preheat" and step <= (int(getattr(stage, 'preheat_mix_end_step', 0) or 0) + 300))
                with _autocast_ctx(enabled=use_amp_wave):
                    bypass_condition = stage.name == "wave_preheat" and step <= int(getattr(stage, 'preheat_mix_end_step', 0) or 0) + 300

                    # FARGAN Teacher Forcing: 使用 pre 参数进行早期稳定
                    fargan_pre = None
                    if (decoder_type == 'aether_fargan' and hasattr(decoder, 'fargan_core') and
                        stage.name == "wave_preheat" and 'teacher_ratio' in locals()):
                        # 在teacher forcing期间使用目标音频的前几帧作为pre
                        if teacher_ratio > 0.1:  # 只有在teacher forcing较强时才使用
                            try:
                                # 为FARGAN提供前序音频帧作为稳定引导
                                pre_frames = min(2, y_target_audio.size(-1) // 160)  # 与train_fargan保持一致（2帧）
                                if pre_frames > 0:
                                    fargan_pre = y_target_audio[..., :pre_frames * 160]
                                    # 添加到csi_dict用于传递给FARGAN
                                    if csi_dict is None:
                                        csi_dict = {}
                                    csi_dict['fargan_pre'] = fargan_pre
                            except Exception as e:
                                if step % 100 == 0:
                                    print(f"   ⚠️ FARGAN teacher forcing setup失败: {e}")

                    if getattr(wave_head, '_is_exciter', False):
                        try:
                            setattr(wave_head.exciter, "_bypass_output_tanh", bypass_condition)
                            y_hat_audio = wave_head(y_in, target_len=target_len, csi_dict=csi_dict)
                        except Exception:
                            y_hat_audio = wave_head(y_in, target_len=target_len)
                    else:
                        try:
                            y_hat_audio = wave_head(y_in, target_len=target_len, csi_dict=csi_dict)
                        except Exception:
                            y_hat_audio = wave_head(y_in, target_len=target_len)


                    # 预热阶段幅度提升
                    if getattr(stage, 'train_wave_head_only', False):
                        y_hat_audio = preheat_gain * y_hat_audio

                # 早期预热阶段注入小噪声避免死区
                if stage.name == "wave_preheat" and step <= 5000:
                    noise = 1e-4 * torch.randn_like(y_hat_audio)
                    y_hat_audio = y_hat_audio + noise

                raw_wave = y_hat_audio
                y_hat_audio = torch.nan_to_num(raw_wave, nan=0.0, posinf=1.0, neginf=-1.0)

                # RMS能量异常监控和自救
                anomaly_type = check_energy_anomaly(y_hat_audio, step, energy_anomaly_state)
                if anomaly_type is not None:
                    rescue_applied = apply_energy_rescue(wave_head, anomaly_type, step)
                    if rescue_applied and anomaly_type == 'low_energy':
                        # 重新生成波形
                        try:
                            if getattr(wave_head, '_is_exciter', False):
                                y_hat_audio = wave_head(y_in, target_len=target_len, csi_dict=csi_dict)
                            else:
                                y_hat_audio = wave_head(y_in, target_len=target_len)
                            y_hat_audio = torch.nan_to_num(y_hat_audio, nan=0.0, posinf=1.0, neginf=-1.0)
                        except:
                            pass

                # 目标音频已提前构造为 y_target_audio
                min_len = min(y_hat_audio.size(-1), y_target_audio.size(-1))
                y_hat_audio = y_hat_audio[..., :min_len]
                y_target_audio = y_target_audio[..., :min_len]

                # 计算对齐增益
                eps = 1e-12
                pred_energy = y_hat_audio.pow(2).mean(dim=-1, keepdim=True)
                tgt_energy = y_target_audio.pow(2).mean(dim=-1, keepdim=True)
                gain_pure = torch.sqrt((tgt_energy + eps) / (pred_energy + eps))
                gain_pure = torch.nan_to_num(gain_pure, nan=1.0, posinf=3.0, neginf=0.25).clamp(0.25, 3.0)
                y_hat_pure_aligned = gain_pure * y_hat_audio

                # 监听混合(仅用于监听，不参与loss)
                y_hat_play = y_hat_audio.clone()
                if stage.name == "wave_preheat" and preheat_mix_ratio < 0.5:
                    with torch.no_grad():
                        y_hat_play = preheat_scale.detach() * y_target_audio + y_hat_play

                # ✅ 明确分离：波形损失只处理波形，特征损失只处理特征
                # 强校验：确保是波形而非特征
                def validate_waveform_tensor(tensor, name):
                    if tensor.dim() != 2:  # 期望 [B, T_audio]
                        raise ValueError(f"{name} 应该是波形 [B, T_audio]，实际形状: {tensor.shape}")
                    return tensor

                pred_wav = validate_waveform_tensor(y_hat_audio.squeeze() if y_hat_audio.dim() > 2 else y_hat_audio, "pred_wav")
                target_wav = validate_waveform_tensor(y_target_audio.squeeze() if y_target_audio.dim() > 2 else y_target_audio, "target_wav")

                # 波形长度对齐
                min_len = min(pred_wav.size(-1), target_wav.size(-1))
                min_batch = min(pred_wav.size(0), target_wav.size(0))
                pred = pred_wav[:min_batch, :min_len]
                target = target_wav[:min_batch, :min_len]

                # 能量自举 - 仅在早期提供轻度增益，之后回落为1
                pred_rms = torch.sqrt(torch.mean(pred.pow(2), dim=-1) + eps)
                tgt_rms = torch.sqrt(torch.mean(target.pow(2), dim=-1) + eps)
                warmup = 500
                if step <= warmup:
                    with torch.no_grad():
                        energy_scale = (tgt_rms / (pred_rms + eps)).clamp(0.8, 1.25).unsqueeze(-1)
                else:
                    energy_scale = 1.0
                pred_for_loss = pred * energy_scale

                use_fast_loss = wave_loss_fast is not None and step > 200
                wl_obj = wave_loss_fast if use_fast_loss else wave_loss
                if stage.name == "wave_preheat" and step <= 20000:
                    with _autocast_ctx(enabled=False):
                        mrstft_loss = mr_stft_boost * wl_obj(pred_for_loss.float(), target.float())
                        # + Mel 频谱 L1（log域，80-bin，与评估管线一致）
                        mel_tf = _get_mel_transform(device)
                        def _logmel_wave(x_2d):          # x_2d: [B, T]
                            M = mel_tf(x_2d.float())
                            return (M.clamp_min(1e-8)).log()

                        mel_pred = _logmel_wave(pred_for_loss)
                        mel_tgt  = _logmel_wave(target)
                        mel_l1   = torch.mean(torch.abs(mel_pred - mel_tgt))

                        # 小权重融合到当前 mrstft_loss
                        mrstft_loss = mrstft_loss + 0.20 * mel_l1
                        # + 音频级 F0 对齐（仅在有声帧上统计）
                        _spec = get_feature_spec(feature_spec_type)
                        f0_tgt = _spec.extract_feature(y, 'f0') if hasattr(_spec, 'extract_feature') else y[:, :, 18:19]  # [B,T,1] (特征域)
                        if hasattr(_spec, 'extract_feature'):
                            voi_tgt = _spec.extract_feature(y, 'voicing')   # [B,T,1]
                        else:
                            voi_tgt = (y[:, :, 18:19] > -1.0).float()  # FARGAN: 从dnn_pitch推导
                        # 与合成器一致的 Hz 映射
                        if not disable_f0_loss:
                            f0_tgt_hz = (SAMPLE_RATE * torch.pow(2.0, f0_tgt.squeeze(-1) - 6.5)).unsqueeze(-1)
                            f0_wave_l = audio_f0_alignment_loss(
                                pred_for_loss.unsqueeze(1), target.unsqueeze(1),
                                f0_tgt_hz, sr=SAMPLE_RATE, hop=FRAME_HOP_SAMPLES, v_mask=voi_tgt
                            )
                            mrstft_loss = mrstft_loss + 0.10 * f0_wave_l


                else:
                    with _autocast_ctx(enabled=use_amp_step):
                        with _autocast_ctx(enabled=False):
                            mrstft_loss = mr_stft_boost * wl_obj(pred_for_loss.float(), target.float())
                        # + Mel 频谱 L1（log域，80-bin，与评估管线一致）
                        mel_tf = _get_mel_transform(device)
                        def _logmel_wave(x_2d):          # x_2d: [B, T]
                            M = mel_tf(x_2d.float())
                            return (M.clamp_min(1e-8)).log()

                        mel_pred = _logmel_wave(pred_for_loss)
                        mel_tgt  = _logmel_wave(target)
                        mel_l1   = torch.mean(torch.abs(mel_pred - mel_tgt))

                        # 小权重融合到当前 mrstft_loss
                        mrstft_loss = mrstft_loss + 0.20 * mel_l1
                        # + 音频级 F0 对齐（仅在有声帧上统计）
                        _spec = get_feature_spec(feature_spec_type)
                        f0_tgt = _spec.extract_feature(y, 'f0') if hasattr(_spec, 'extract_feature') else y[:, :, 18:19]  # [B,T,1] (特征域)
                        if hasattr(_spec, 'extract_feature'):
                            voi_tgt = _spec.extract_feature(y, 'voicing')   # [B,T,1]
                        else:
                            voi_tgt = (y[:, :, 18:19] > -1.0).float()  # FARGAN: 从dnn_pitch推导
                        # 与合成器一致的 Hz 映射
                        if not disable_f0_loss:
                            f0_tgt_hz = (SAMPLE_RATE * torch.pow(2.0, f0_tgt.squeeze(-1) - 6.5)).unsqueeze(-1)
                            f0_wave_l = audio_f0_alignment_loss(
                                pred_for_loss.unsqueeze(1), target.unsqueeze(1),
                                f0_tgt_hz, sr=SAMPLE_RATE, hop=FRAME_HOP_SAMPLES, v_mask=voi_tgt
                            )
                            mrstft_loss = mrstft_loss + 0.10 * f0_wave_l



                # RMS对齐损失 - 关键的起振辅助
                def rms_db(x):
                    return 20 * torch.log10(x.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt() + 1e-8)

                rms_loss = (rms_db(pred) - rms_db(target)).abs().mean()

                # 动态权重：前3000步重点学响度，之后转向细节
                warmup_steps = 3000
                if step <= warmup_steps:
                    lambda_rms = 2.0 * (1.0 - step / warmup_steps) + 0.2
                    lambda_stft = 0.5 + 0.5 * (step / warmup_steps)
                else:
                    lambda_rms = 0.2
                    lambda_stft = 1.0

                # 额外损失项
                log_rms_diff = torch.abs((pred_rms + eps).log() - (tgt_rms + eps).log()).mean()
                l_time = torch.abs(pred - target).mean()
                floor_level = 10 ** (-45.0 / 20.0)
                floor_penalty = torch.relu(floor_level - pred_rms).mean()
                dc_penalty = pred.mean(dim=-1).abs().mean()
                l_wav_l1 = F.l1_loss(pred_for_loss, target)
                l_rms = F.l1_loss(pred_rms, tgt_rms)
                si_snr_loss = 0.0
                if stage.name != "wave_preheat" or step > 400:
                    si_snr_vals = compute_si_snr(pred_for_loss, target)
                    si_snr_loss = -si_snr_vals.mean()

                # — RMS门控与F0健康闭环控制 —
                # 预热早期或弱音时，暂时关闭STFT/Si-SNR，仅保留小权重的时域L1/RMS
                try:
                    pred_rms_db_mean = float((20.0 * torch.log10(torch.clamp(pred_rms.mean(), min=1e-8))).detach().cpu().item())
                except Exception:
                    pred_rms_db_mean = -100.0

                f0_alert_active = False
                if 'f0_alert_until_step' in locals():
                    f0_alert_active = step <= f0_alert_until_step

                # 动态门控阈值：-35 dB @ step=0 线性升到 -28 dB @ step≈1000
                thr_db = -35.0 + 7.0 * min(1.0, step / 1000.0)
                gating_active = (stage.name == "wave_preheat") and ((pred_rms_db_mean < thr_db) or (step < 400) or f0_alert_active)

                if gating_active:
                    wave_loss_val = (
                        0.10 * l_time +
                        0.10 * l_rms +
                        0.02 * dc_penalty
                    )
                else:
                    # 应用动态权重的损失组合（完整）
                    wave_loss_val = (
                        lambda_stft * mrstft_loss +      # 动态STFT权重
                        lambda_rms * rms_loss +          # 动态RMS对齐权重
                        0.5 * log_rms_diff +             # 原有RMS损失
                        0.2 * l_time +                   # L1损失
                        2.0 * floor_penalty +            # 静音惩罚
                        0.05 * dc_penalty +              # 直流惩罚
                        0.05 * l_wav_l1 +                # 波形幅度约束
                        0.02 * l_rms +                   # RMS幅度约束
                        0.5 * si_snr_loss                # SI-SNR 提升时间域一致性
                    )
                if not disable_f0_loss:
                    spec = get_feature_spec(feature_spec_type)
                    f0_tgt_hz = spec.extract_feature(y, 'f0') if hasattr(spec, 'extract_feature') else y[:, :, 18:19]
                    if hasattr(spec, 'extract_feature'):
                        voi_mask = spec.extract_feature(y, 'voicing')   # [B,T,1]
                    else:
                        voi_mask = (y[:, :, 18:19] > -1.0).float()
                    f0_align  = audio_f0_alignment_loss(
                        y_hat_audio.unsqueeze(1), y_target_audio.unsqueeze(1),
                        f0_tgt_hz, sr=16000, hop=160, v_mask=voi_mask
                    )
                    wave_loss_val = wave_loss_val + 0.02 * f0_align

                if hasattr(decoder, 'synth') and hasattr(decoder.synth, 'tilt'):
                    wave_loss_val = wave_loss_val + 1e-4 * (decoder.synth.tilt ** 2)

                # AETHERFARGANDecoder: 使用 FARGAN 训练损失作为主要损失
                if decoder_type == 'aether_fargan' and hasattr(decoder, 'fargan_core'):
                    try:
                        # 从FARGAN特征估计周期（用于FARGAN训练损失）
                        if hasattr(decoder, '_estimate_period'):
                            # 使用解码器的周期估计
                            estimated_period = decoder._estimate_period(y_hat)
                        else:
                            # 回退方案：从特征手动估计周期
                            dnn_pitch = y_hat[..., 18:19] if y_hat.size(-1) > 18 else torch.zeros_like(y_hat[..., :1])
                            period_raw = 256.0 / torch.pow(2.0, dnn_pitch + 1.5)
                            estimated_period = torch.round(torch.clamp(period_raw, 32.0, 255.0)).long().squeeze(-1)

                        # 计算 FARGAN 训练损失 (comprehensive training loss)
                        from training.fargan_losses import compute_fargan_training_loss
                        fargan_train_loss, fargan_train_details = compute_fargan_training_loss(
                            y_hat_audio.squeeze(1), y_target_audio.squeeze(1), estimated_period,
                            frame_size=160, subframe_size=40, device=device
                        )

                        # 计算 FARGAN 原版损失 (作为辅助损失)，聚焦于 pre 之后的首帧
                        focus_start = 0
                        if 'fargan_pre' in (csi_dict or {}) and csi_dict['fargan_pre'] is not None:
                            focus_start = int(csi_dict['fargan_pre'].size(-1))
                        orig_total, orig_details = compute_fargan_original_style_loss(
                            y_hat_audio.squeeze(1) if y_hat_audio.dim() == 3 else y_hat_audio,
                            y_target_audio.squeeze(1) if y_target_audio.dim() == 3 else y_target_audio,
                            device=device, frame_size=160, focus_start=focus_start
                        )

                        # 权重调度：FARGAN训练损失为主，原版损失为辅
                        ramp_steps = 5000  # ramp期长度
                        if step < ramp_steps:
                            # 渐进增加FARGAN训练损失权重
                            t = step / ramp_steps
                            smooth_t = 3 * t * t - 2 * t * t * t
                            fargan_train_weight = 0.5 + 0.5 * smooth_t  # 0.5 -> 1.0
                            orig_weight = 0.3 * (1.0 - smooth_t)  # 0.3 -> 0.0
                        else:
                            fargan_train_weight = 1.0
                            orig_weight = 0.1  # 保留少量原版损失

                        # 组合损失：以FARGAN训练损失为主
                        combined_fargan_loss = (
                            fargan_train_weight * fargan_train_loss +
                            orig_weight * orig_total
                        )

                        # MR-STFT增强保持不变
                        mr_stft_boost = 1.0
                        if step > ramp_steps * 0.6:
                            if step <= ramp_steps * 0.8:
                                mr_stft_boost = 1.5
                            else:
                                mr_stft_boost = 2.0

                        # 最终波形损失：MR-STFT + FARGAN组合损失
                        wave_loss_val = mr_stft_boost * wave_loss_val + combined_fargan_loss

                        # 记录用于日志
                        fargan_loss_logs = {
                            'fargan_train_weight': fargan_train_weight,
                            'fargan_train_loss': fargan_train_loss.item(),
                            'orig_weight': orig_weight,
                            'orig_loss': orig_total.item(),
                            'combined_fargan': combined_fargan_loss.item(),
                            'mr_stft_boost': mr_stft_boost,
                            'l1': fargan_train_details.get('l1', torch.tensor(0.0)).item(),
                            'pitch_consistency': fargan_train_details.get('pitch_consistency', torch.tensor(0.0)).item(),
                            'subframe_alignment': fargan_train_details.get('subframe_alignment', torch.tensor(0.0)).item(),
                            'ramp_progress': step / ramp_steps if step < ramp_steps else 1.0
                        }
                    except Exception as e:
                        if step % 100 == 0:
                            print(f"⚠️ FARGAN损失计算失败 at step {step}: {e}")

                wave_loss_val = torch.nan_to_num(wave_loss_val, nan=0.0, posinf=1e4, neginf=1e4)


        # 总损失
        if stage.name == "clean_baseline":
            total_loss = _finite_scalar(recon_loss, "recon_total", step)
        elif stage.name == "wave_preheat":
            if stage.enable_audio_quality and original_audio is not None:
                # 放大波形损失幅度，确保合成头得到足够大梯度；保留少量F0引导
                total_f0_for_mix = total_f0 if 'total_f0' in locals() else torch.tensor(0.0, device=device)
                total_loss = _finite_scalar(30.0 * wave_loss_val, "wave_loss_scaled", step) \
                           + _finite_scalar(0.1 * total_f0_for_mix, "f0_mix_scaled", step)
            else:
                total_loss = _finite_scalar(recon_loss, "recon_total", step)
        else:
            total_loss = _finite_scalar(recon_loss, "recon_total", step) \
                       + _finite_scalar(rate_loss_val, "rate_loss", step) \
                       + _finite_scalar(wave_loss_val, "wave_loss", step)

        # 高损失批次的软跳过机制
        if not torch.isfinite(total_loss).all() or total_loss.item() > 50.0:
            if total_loss.item() > 50.0:
                print(f"⚠️ 步骤 {step}: 损失过大 ({total_loss.item():.2f}), 跳过此批次")
                optimizer.zero_grad(set_to_none=True)
                # 不执行 scaler.update()：本步未进行缩放/反传，避免 torch.amp 的 _scale 断言
                continue

        # 数值稳定性检查
        if not torch.isfinite(total_loss).all():
            try:
                spec = get_feature_spec(feature_spec_type)
                if feature_spec_type == "fargan":
                    names = ["ceps", "dnn_pitch", "frame_corr", "lpc"]
                else:
                    names = ["ceps","f0","voicing","enhanced","lpc","prosodic"]
                flags = []
                for n in names:
                    if hasattr(spec, 'get_feature_slice'):
                        sl = spec.get_feature_slice(n)
                    else:
                        # FARGAN hard-coded slices
                        slice_map = {"ceps": slice(0, 18), "dnn_pitch": slice(18, 19), "frame_corr": slice(19, 20), "lpc": slice(20, 36)}
                        sl = slice_map.get(n, slice(0, 1))
                    p, t = y_hat[..., sl], y[..., sl]
                    flags.append(f"{n}:pred_bad={torch.isnan(p).any().item() or torch.isinf(p).any().item()},"
                                f"tgt_bad={torch.isnan(t).any().item() or torch.isinf(t).any().item()}")
                print("⚠️ Loss NaN 审计: " + " | ".join(flags))
            except Exception as e:
                print(f"⚠️ 审计失败: {e}")
            print(f"⚠️ 步骤 {step}: 检测到损失异常 (NaN/Inf), 跳过此步")
            continue


        # 反向传播 - 增强的数值稳定性
        try:
            if scaler is not None:
                scaler.scale(total_loss).backward()

                # 检查梯度前需要先unscale
                scaler.unscale_(optimizer)
                grad_ok = _grad_ok([encoder, decoder, wave_head], debug=(step <= 10))

                # 🔍 调试信息：梯度统计（AMP 分支）
                if step <= 10 or step % 200 == 0:
                    # 先收集基础梯度，避免未定义变量
                    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
                    decoder_grads = [p.grad for p in decoder.parameters() if p.grad is not None]
                    # 再收集 F0 相关梯度
                    f0_encoder_grads = [
                        p.grad for n, p in encoder.named_parameters()
                        if 'f0_encoder' in n and p.grad is not None
                    ]
                    f0_decoder_grads = [
                        p.grad for n, p in decoder.named_parameters()
                        if ('f0' in n or 'voic' in n) and p.grad is not None
                    ]
                    # 打印统计
                    if encoder_grads:
                        enc_grad_norm = torch.stack([g.norm() for g in encoder_grads]).mean().item()
                        print(f"🔧 Step {step} 梯度统计:")
                        print(f"   编码器梯度范数: {enc_grad_norm:.6f}")
                    if decoder_grads:
                        dec_grad_norm = torch.stack([g.norm() for g in decoder_grads]).mean().item()
                        print(f"   解码器梯度范数: {dec_grad_norm:.6f}")
                    if f0_encoder_grads:
                        f0_enc_norm = torch.stack([g.norm() for g in f0_encoder_grads]).mean().item()
                        print(f"   🎯 F0编码器梯度范数: {f0_enc_norm:.6f}")
                    if f0_decoder_grads:
                        f0_dec_norm = torch.stack([g.norm() for g in f0_decoder_grads]).mean().item()
                        print(f"   🎯 F0解码器梯度范数: {f0_dec_norm:.6f}")


                if grad_ok:
                    # 先按值域裁剪，再按范数裁剪，双重稳健
                    torch.nn.utils.clip_grad_value_(
                        [p for group in optimizer.param_groups for p in group['params']],
                        clip_value=1.0
                    )
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group['params']],
                        max_norm=1.0
                    )
                    if step <= 10 or step % 200 == 0:
                        print(f"   裁剪前梯度范数: {grad_norm.item():.6f}")

                # ✅ 只有在梯度有限时才执行 step；无论如何都要 update()
                if grad_ok:
                    scaler.step(optimizer)
                else:
                    print(f"⚠️ 步骤 {step}: 检测到梯度异常，已跳过本次参数更新")
                    optimizer.zero_grad(set_to_none=True)
                
                scaler.update()

            else:
                total_loss.backward()
                grad_ok = _grad_ok([encoder, decoder, wave_head], debug=(step <= 10))

                # 🔍 调试信息：梯度统计 (非AMP模式)
                if step <= 10 or step % 200 == 0:
                    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
                    decoder_grads = [p.grad for p in decoder.parameters() if p.grad is not None]
                    if encoder_grads:
                        enc_grad_norm = torch.stack([g.norm() for g in encoder_grads]).mean().item()
                        print(f"🔧 Step {step} 梯度统计 (非AMP):")
                        print(f"   编码器梯度范数: {enc_grad_norm:.6f}")
                    if decoder_grads:
                        dec_grad_norm = torch.stack([g.norm() for g in decoder_grads]).mean().item()
                        print(f"   解码器梯度范数: {dec_grad_norm:.6f}")
                    print(f"   梯度检查通过: {grad_ok}")

                if grad_ok:
                    # 双重梯度裁剪：先值域裁剪，再范数裁剪（AMP分支）
                    all_params = [p for group in optimizer.param_groups for p in group['params']]
                    torch.nn.utils.clip_grad_value_(all_params, clip_value=1.0)
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    if step <= 10 or step % 200 == 0:
                        print(f"   裁剪前梯度范数: {grad_norm.item():.6f}")
                    optimizer.step()
                else:
                    print(f"⚠️ 步骤 {step}: 检测到梯度异常，已跳过本次参数更新")
                    optimizer.zero_grad(set_to_none=True)
                    continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ 步骤 {step}: GPU内存不足, 跳过此步")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"⚠️ 步骤 {step}: 反向传播异常: {e}")
                continue

        # 高级FiLM调度
        if film_scheduler is not None:
            film_scheduler.step()

        # 学习率调度器更新
        if decoder_type == 'aether_fargan':
            if step < ramp_steps and not scheduler_switched:
                # ramp 期间使用 LambdaLR
                lr_scheduler.step()
            elif step >= ramp_steps and not scheduler_switched:
                # ramp 结束，切换到 ReduceLROnPlateau
                print(f"🔄 Step {step}: 切换学习率调度器从 LambdaLR 到 ReduceLROnPlateau")
                scheduler_switched = True
            elif scheduler_switched:
                # 使用 ReduceLROnPlateau，基于验证损失调整
                plateau_scheduler.step(current_loss)
        else:
            # 非 FARGAN 模式，使用标准调度
            lr_scheduler.step()

        # 记录最佳损失
        current_loss = float(total_loss.detach().cpu())
        convergence_losses.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss

        # 中间检查点保存
        if step % checkpoint_every == 0:
            intermediate_checkpoint_path = checkpoint_dir / f"stage_{current_stage_index}_{stage.name}_step_{step}.pth"
            torch.save({
                'stage_index': current_stage_index,
                'stage_name': stage.name,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'wave_head_state_dict': wave_head.state_dict(),
                'step': step,
                'best_loss': best_loss,
                'current_loss': current_loss,
                'optimizer_state_dict': optimizer.state_dict()
            }, intermediate_checkpoint_path)
            print(f"  💾 中间检查点已保存: {intermediate_checkpoint_path.name}")

        # 日志输出
        if step % 50 == 0 or step <= 10:
            film_info = ""
            if stage.use_film and hasattr(encoder, 'film_ratio'):
                film_info = f" film={encoder.film_ratio:.3f}"

            pred_rms_db = 20 * torch.log10(torch.sqrt(torch.mean(pred.pow(2)) + 1e-12) + 1e-12).item() if 'pred' in locals() else 0

            # 显示F0损失信息
            f0_loss_str = ""
            if 'f0_losses' in locals() and f0_losses is not None:
                f0_total = f0_losses.get('total_f0_loss', torch.tensor(0.0))
                if f0_total.item() > 0:
                    f0_loss_str = f" f0_loss={f0_total.item():.6f}"

            # FARGAN损失监控
            fargan_loss_str = ""
            if fargan_loss_logs is not None and decoder_type == 'aether_fargan':
                logs = fargan_loss_logs
                if logs:
                    fargan_loss_str = (f" fargan_train={logs.get('fargan_train_loss', 0):.4f} "
                                     f"(w={logs.get('fargan_train_weight', 0):.2f}) "
                                     f"orig={logs.get('orig_loss', 0):.4f} "
                                     f"(w={logs.get('orig_weight', 0):.2f}) "
                                     f"combined={logs.get('combined_fargan', 0):.4f} "
                                     f"mr_boost={logs.get('mr_stft_boost', 1.0):.1f} "
                                     f"pitch={logs.get('pitch_consistency', 0):.4f} "
                                     f"subframe={logs.get('subframe_alignment', 0):.4f} "
                                     f"ramp={logs.get('ramp_progress', 0):.1%}")

            print(f"  步骤 {step}/{actual_steps} (epoch {current_epoch:.1f}): "
                  f"loss={current_loss:.6f} recon={recon_loss.item():.6f}{f0_loss_str}{fargan_loss_str} "
                  f"best={best_loss:.6f}{film_info} pred_rms={pred_rms_db:.1f}dB")

            # 追加时序诊断：参数时间方差与F0通道变化
            try:
                with torch.no_grad():
                    tvar = y_hat.float().std(dim=1).mean().item()
                    msg = f"     • param_t.std≈{tvar:.4f}"

                    # 使用FeatureSpec提取F0和韵律特征进行诊断
                    feature_spec = get_feature_spec(feature_spec_type)
                    if hasattr(feature_spec, 'extract_feature'):
                        f0_block = feature_spec.extract_feature(y_hat, 'f0')
                        prosodic_block = feature_spec.extract_feature(y_hat, 'prosodic')
                    else:
                        # FARGAN
                        f0_block = y_hat[:, :, 18:19]  # dnn_pitch
                        prosodic_block = y_hat[:, :, 19:20]  # frame_corr

                    f0_tvar = f0_block.float().std(dim=1).mean().item()
                    f0_mean = f0_block.float().mean().item()
                    f0_range = f0_block.float().max().item() - f0_block.float().min().item()

                    prosodic_tvar = prosodic_block.float().std(dim=1).mean().item()
                    prosodic_range = prosodic_block.float().max().item() - prosodic_block.float().min().item()

                    msg += f" | f0_t.std≈{f0_tvar:.4f} mean≈{f0_mean:.3f} range≈{f0_range:.3f}"
                    msg += f" | prosodic_t.std≈{prosodic_tvar:.4f} range≈{prosodic_range:.3f}"
                    print(msg)
            except Exception:
                pass

    # 训练后处理
    encoder.eval()
    decoder.eval()
    wave_head.eval()

    # 收敛性检查
    if len(convergence_losses) >= 10:
        recent_avg = sum(convergence_losses[-10:]) / 10
        early_avg = sum(convergence_losses[:10]) / 10 if len(convergence_losses) >= 10 else recent_avg
        convergence_rate = max(0.0, early_avg - recent_avg)
    else:
        convergence_rate = 0.0

    # 阶段验收判定
    passed = True
    fail_reasons = []
    audio_quality = {}

    # 基础损失验收
    if stage.max_final_loss > 0 and best_loss > stage.max_final_loss:
        passed = False
        fail_reasons.append(f"最终损失 {best_loss:.6f} > {stage.max_final_loss}")

    if stage.min_convergence_rate > 0 and convergence_rate < stage.min_convergence_rate:
        passed = False
        fail_reasons.append(f"收敛率 {convergence_rate:.6f} < {stage.min_convergence_rate}")

    # 音频质量验收 - 关键的PESQ/SNR Gate
    if stage.enable_audio_quality:
        print(f"\n🎵 执行音频质量验收...")
        try:
            # 设置为评估模式
            prev_encoder_mode = encoder.training
            prev_decoder_mode = decoder.training
            prev_wave_mode = wave_head.training
            encoder.eval()
            decoder.eval()
            wave_head.eval()

            # 取一个验证批次
            eval_batch = next(batch_iter)
            x_eval = eval_batch['x'].to(device, non_blocking=True)
            y_eval = eval_batch['y'].to(device, non_blocking=True)
            original_audio = eval_batch.get('audio')
            if original_audio is not None:
                original_audio = original_audio.to(device, non_blocking=True)
            csi_dict = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                       for k, v in eval_batch.get('csi', {}).items()}

            x_eval = torch.nan_to_num(x_eval, nan=0.0, posinf=1e4, neginf=-1e4)
            y_eval = torch.nan_to_num(y_eval, nan=0.0, posinf=1e4, neginf=-1e4)

            with torch.no_grad():
                # 编解码（解码端同样注入acoustic_priors，避免CSI维度不一致）
                z_eval, _ = encoder(x_eval, csi_dict=csi_dict, inference=True)
                csi_dec_eval = dict(csi_dict)
                try:
                    csi_dec_eval["acoustic_priors"] = extract_acoustic_priors(x_eval).detach()
                except Exception:
                    pass
                y_hat_eval = decoder(z_eval, csi_dict=csi_dec_eval)

            # 计算音频质量指标
            csi_eval = dict(csi_dict)
            try:
                csi_eval["acoustic_priors"] = extract_acoustic_priors(x_eval).detach()
            except Exception:
                pass

            audio_quality = calculate_audio_quality(
                y_hat_eval, y_eval, wave_head, original_audio, csi_dict=csi_eval
            )

            if 'error' not in audio_quality:
                print(f"    SNR: {audio_quality['snr_db']:.2f} dB")
                print(f"    SI-SNR: {audio_quality['si_snr_db']:.2f} dB")
                print(f"    Mel Cosine: {audio_quality['mel_cos']:.3f}")
                print(f"    Mel L2: {audio_quality['mel_l2']:.4f}")
                print(f"    光谱相关性: {audio_quality['feature_correlation']:.3f}")
                print(f"    光谱失真: {audio_quality['spectral_distortion']:.4f}")
                print(f"    预测RMS: {audio_quality['pred_rms_db']:.1f} dB")
                print(f"    目标RMS: {audio_quality['target_rms_db']:.1f} dB")

                # 验收门槛检查
                audio_passed = True
                if audio_quality['snr_db'] < stage.min_snr_db:
                    audio_passed = False
                    fail_reasons.append(f"SNR不足: {audio_quality['snr_db']:.2f} < {stage.min_snr_db}")
                if audio_quality['mel_cos'] < stage.min_mel_cos:
                    audio_passed = False
                    fail_reasons.append(
                        f"Mel Cosine不足: {audio_quality['mel_cos']:.3f} < {stage.min_mel_cos}"
                    )
                if audio_quality['mel_l2'] > stage.max_mel_l2:
                    audio_passed = False
                    fail_reasons.append(
                        f"Mel L2超限: {audio_quality['mel_l2']:.4f} > {stage.max_mel_l2}"
                    )
                if stage.max_spectral_distortion > 0.0 and audio_quality['spectral_distortion'] > stage.max_spectral_distortion:
                    audio_passed = False
                    fail_reasons.append(
                        f"谱失真超限: {audio_quality['spectral_distortion']:.4f} > {stage.max_spectral_distortion}"
                    )
                if stage.max_rms_delta_db > 0.0:
                    rms_delta = abs(audio_quality['pred_rms_db'] - audio_quality['target_rms_db'])
                    if rms_delta > stage.max_rms_delta_db:
                        audio_passed = False
                        fail_reasons.append(
                            f"RMS差值超限: {rms_delta:.2f} dB > {stage.max_rms_delta_db} dB"
                        )

                if audio_passed:
                    print(f"    ✅ 音频质量验收通过")

                    if train_dataset is not None:
                        prev_modes = (
                            encoder.training,
                            decoder.training,
                            wave_head.training
                        )
                        encoder.eval()
                        decoder.eval()
                        wave_head.eval()
                        try:
                            stage_models = {stage.name: (encoder, decoder, wave_head)}
                            summary_stage = integrate_audio_validation(
                                dataset=train_dataset,
                                trained_models=stage_models,
                                device=device,
                                output_dir=str(checkpoint_dir)
                            )
                            print(f"    🎨 阶段 {stage.name} 音频可视化已生成 (audio_validation/{stage.name})")
                            if summary_stage:
                                print(summary_stage)
                        except Exception as viz_e:
                            print(f"    ⚠️ 阶段 {stage.name} 音频可视化失败: {viz_e}")
                        finally:
                            encoder.train(prev_modes[0])
                            decoder.train(prev_modes[1])
                            wave_head.train(prev_modes[2])
                else:
                    passed = False
                    print(f"    ❌ 音频质量验收失败")

                # 导出验证音频和可视化 (无论是否通过都导出，用于诊断)
                export_validation_audio(
                    stage_name=stage.name,
                    y_hat_feats=y_hat_eval,
                    y_orig_feats=y_eval,
                    wave_head=wave_head,
                    original_audio=original_audio,
                    output_dir=checkpoint_dir.parent,
                    csi_dict=csi_dict
                )

            else:
                print(f"    ⚠️ 音频质量评估失败: {audio_quality['error']}")
                # 不因评估失败而阻止阶段通过，但记录警告
                fail_reasons.append("音频质量评估失败")

            # 恢复训练模式
            encoder.train(prev_encoder_mode)
            decoder.train(prev_decoder_mode)
            wave_head.train(prev_wave_mode)

        except Exception as e:
            print(f"    ⚠️ 音频质量评估异常: {e}")
            traceback.print_exc()
            audio_quality = {'error': str(e)}

    # 保存检查点
    checkpoint_path = checkpoint_dir / f"stage_{current_stage_index}_{stage.name}.pth"
    checkpoint = {
        'stage_index': current_stage_index,
        'stage_name': stage.name,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'wave_head_state_dict': wave_head.state_dict(),
        'best_loss': best_loss,
        'convergence_rate': convergence_rate,
        'step': step,
        'passed': passed,
        'audio_quality': audio_quality,  # 添加音频质量信息
        'fail_reasons': fail_reasons
    }
    torch.save(checkpoint, checkpoint_path)

    result = {
        'passed': passed,
        'best_loss': best_loss,
        'convergence_rate': convergence_rate,
        'final_step': step,
        'fail_reasons': fail_reasons,
        'checkpoint_path': checkpoint_path,
        'audio_quality': audio_quality
    }

    status = "✅ 通过" if passed else "❌ 未通过"
    print(f"\n🏁 阶段 {stage.name} 完成: {status}")
    print(f"   最佳损失: {best_loss:.6f}")
    print(f"   收敛率: {convergence_rate:.6f}")
    if audio_quality and 'error' not in audio_quality:
        print(
            f"   音频指标: PESQ-like={audio_quality['pesq_like']:.3f}, "
            f"SNR={audio_quality['snr_db']:.2f} dB, "
            f"MelCos={audio_quality['mel_cos']:.3f}, "
            f"RMS={audio_quality['pred_rms_db']:.1f} dB"
        )
    if fail_reasons:
        print(f"   失败原因: {'; '.join(fail_reasons)}")

    # 恢复latent量化开关
    try:
        if stage.name == 'wave_preheat' and _orig_quant_flag is not None:
            encoder.quantize_latent = _orig_quant_flag
            print("🔁 已恢复latent量化为阶段前设置")
    except Exception:
        pass

    return result


def create_progressive_stages() -> List[ProgressiveStage]:
    """创建渐进式训练阶段序列"""
    return [
        ProgressiveStage(
            name="clean_baseline",
            description="清洁基线 - 无信道干扰下的特征重建",
            epochs=1.0,
            use_film=False,
            use_moe=False,
            use_quantization=False,
            apply_channel=False,
            channel_type="clean",
            layered_loss=False,
            learning_rate=5e-5,
            lambda_rate=1e-5,
            min_convergence_rate=0.5,
            max_final_loss=0.2,
            early_stop_loss=0.01
        ),

        ProgressiveStage(
            name="wave_preheat",
            description="波形头预热 - 训练波形解码器",
            epochs=2.0,
            use_film=False,
            use_moe=False,
            use_quantization=False,
            apply_channel=False,
            channel_type="clean",
            layered_loss=True,
            enable_audio_quality=True,
            learning_rate=1e-4,
            lambda_rate=0.0,
            min_convergence_rate=-20.0,
            max_final_loss=3.0,  # 放宽从1.5到3.0
            early_stop_loss=0.0,
            min_snr_db=-5.0,  # 大幅放宽SNR门槛（如果启用的话）
            min_mel_cos=0.60,  # 放宽Mel余弦相似度门槛
            max_mel_l2=0.40,   # 放宽Mel L2误差门槛
            max_spectral_distortion=1.20,  # 放宽谱失真门槛
            max_rms_delta_db=8.0,  # 放宽RMS差值门槛
            wave_start_step=0,
            wave_full_start_step=1200,
            wave_lowpass_weight=0.9,
            wave_full_weight=1.0,
            wave_lowpass_schedule=[(0,1.2),(800,0.8),(1200,0.6)],
            wave_full_schedule=[(0,0.2),(600,0.5),(1200,0.8),(1800,1.0)],
            train_wave_head_only=True,  # 🔧 FARGAN预热阶段冻结编解码器，只训练波形合成
            preheat_mix_start_step=0,
            preheat_mix_end_step=15000,  # 🔧 进一步延缓teacher-forcing衰减，给F0更多稳定时间
            preheat_chunk_frames=128,
        ),

        ProgressiveStage(
            name="channel_adapt",
            description="信道适应（含FiLM调度）",
            epochs=3.0,
            use_film=True,
            use_moe=True,
            use_quantization=False,
            apply_channel=True,
            channel_type="awgn",
            layered_loss=True,
            learning_rate=3e-4,
            lambda_rate=0.1,
            min_convergence_rate=1.0,
            max_final_loss=1.0,  # 适中门槛
            early_stop_loss=0.01,
            enable_audio_quality=True,  # 开始启用音频质量门槛
            min_snr_db=3.0,  # 从合理的SNR开始
            min_mel_cos=0.75,  # 适中的Mel相似度要求
            max_mel_l2=0.30,   # 适中的Mel误差
            max_spectral_distortion=0.90,  # 适中的谱失真
            max_rms_delta_db=5.0,  # 适中的RMS要求
            use_advanced_scheduler=True,
            film_warmup_steps=500,
            film_start_ratio=0.1,
            film_beta_scale_start=0.1
        ),

        ProgressiveStage(
            name="full_optimization",
            description="完整优化 - 端到端训练",
            epochs=5.0,
            use_film=True,
            use_moe=True,
            use_quantization=True,
            apply_channel=True,
            channel_type="fading",
            layered_loss=True,
            learning_rate=2e-4,
            lambda_rate=0.2,
            lambda_balance=0.1,
            lambda_cons=0.05,
            min_convergence_rate=0.5,
            max_final_loss=1.0,
            early_stop_loss=0.01,
            enable_audio_quality=True,
            min_snr_db=10.0,
            min_mel_cos=0.90,
            max_mel_l2=0.12,
            max_spectral_distortion=0.60,
            max_rms_delta_db=2.5,
            target_kbps=1.2,
            max_kbps_p90=1.6
        )
    ]


def main():
    parser = argparse.ArgumentParser(description='AETHER渐进式训练(精简版)')

    # 数据参数
    parser.add_argument('--features', type=str, required=True, help='特征文件路径(.f32)')
    parser.add_argument('--pcm', type=str, required=True, help='PCM音频文件路径')
    parser.add_argument('--seq-len', type=int, default=400, help='序列长度(帧)')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--limit-seqs', type=int, default=None, help='限制序列数量')
    parser.add_argument('--feature-dims', type=int, default=36, help='特征维度 (36 for FARGAN, 48 for AETHER)')
    parser.add_argument('--feature-spec-type', type=str, default='fargan', choices=['fargan', 'aether'], help='特征规范类型')

    # 训练参数
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    parser.add_argument('--device', type=str, default='auto', help='设备(auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--wave-head-type', type=str, default='conv', choices=['conv', 'exciter'], help='波形头类型')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='检查点保存间隔(步数)')
    # 预热可调参数（可覆盖阶段默认值）
    parser.add_argument('--preheat-mix-end', type=int, default=None, help='覆盖wave_preheat的teacher-forcing结束步数')
    parser.add_argument('--preheat-chunk-frames', type=int, default=None, help='覆盖wave_preheat的chunk帧数')

    # 阶段二独立FARGAN训练参数
    parser.add_argument('--stage2-fargan-only', action='store_true', help='阶段二模式：冻结编解码器，只训练独立FARGAN波形头')
    parser.add_argument('--stage1-checkpoint', type=str, default=None, help='阶段一Aether模型checkpoint路径')
    parser.add_argument('--fargan-learning-rate', type=float, default=1e-4, help='FARGAN训练学习率')
    parser.add_argument('--fargan-lr-decay', type=float, default=2e-5, help='FARGAN学习率衰减率')
    parser.add_argument('--fargan-original-epochs', type=int, default=0, help='FARGAN原版损失训练轮数')
    parser.add_argument('--fargan-ramp-epochs', type=int, default=10, help='FARGAN损失混合渐变轮数')

    # 阶段控制
    parser.add_argument('--stages', type=str, default='all', help='训练阶段(all/stage1,stage2等)')
    parser.add_argument('--start-stage', type=int, default=1, help='开始阶段(1-based)')
    parser.add_argument('--end-stage', type=int, default=None, help='结束阶段(1-based,包含)')
    parser.add_argument('--skip-passed', action='store_true', help='跳过已通过的阶段')

    # 解码器类型
    parser.add_argument('--decoder-type', type=str, default='aether', choices=['aether', 'aether_fargan'],
                        help='解码器类型: aether(仅特征重建), aether_fargan(特征+FARGAN波形合成)')
    parser.add_argument('--disable-f0-loss', action='store_true', help='禁用所有与F0相关的损失与对齐')

    args = parser.parse_args()

    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🎯 使用设备: {device}")

    # 当使用 FARGAN 解码路径时，默认禁用 F0 损失（按你的需求）
    if args.decoder_type == 'aether_fargan':
        args.disable_f0_loss = True

    # 预初始化Mel缓存（便于后续评估阶段直接复用）
    _get_mel_transform(device)

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据加载器 - 自动从参数路径推断数据目录
    # 从features路径推断数据目录 (/path/to/data_cn/lmr_export/features_48_complete.f32 -> /path/to/data_cn)
    features_path = Path(args.features)
    if "lmr_export" in features_path.parts:
        # 从 lmr_export 目录向上找到 data_cn
        data_dir = None
        for i, part in enumerate(features_path.parts):
            if part == "lmr_export" and i > 0:
                data_dir = Path(*features_path.parts[:i])
                break
        if data_dir is None:
            data_dir = features_path.parent.parent  # fallback
    else:
        data_dir = Path("/home/bluestar/FARGAN/opus/data_cn")  # default fallback

    print(f"🗂️ 推断数据目录: {data_dir}")
    # 🔥 关键CPU优化：降低DataLoader压力防止卡死
    loader_workers = max(1, min(4, args.num_workers))
    train_loader, dataset = create_aether_data_loader(
        data_dir=str(data_dir),
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        max_samples=args.limit_seqs,
        num_workers=loader_workers,
        energy_selection=True,
        test_mode=False,
        feature_spec_type=args.feature_spec_type,
        features_file=args.features,
        audio_file=args.pcm
    )
    print(f"🗂️ 数据加载器就绪: {len(train_loader)} batches")

    # 模型初始化
    cfg = TrainConfig()

    # 阶段二独立FARGAN训练模式
    if args.stage2_fargan_only:
        print("=== STAGE2 FARGAN-ONLY TRAINING MODE ===")

        # 验证阶段一checkpoint
        if not args.stage1_checkpoint:
            raise ValueError("--stage1-checkpoint is required for --stage2-fargan-only mode")

        if not os.path.exists(args.stage1_checkpoint):
            raise FileNotFoundError(f"Stage1 checkpoint not found: {args.stage1_checkpoint}")

        # 加载冻结的Aether编解码器
        frozen_encoder, frozen_decoder = load_frozen_aether_models(
            args.stage1_checkpoint, device, args.feature_dims
        )

        # 创建独立的FARGAN波形头
        fargan_wavehead = create_independent_fargan_wavehead(device)

        # 设置模型变量 (保持兼容性)
        encoder = frozen_encoder
        decoder = frozen_decoder
        wave_head = fargan_wavehead

        print(f"Stage2 FARGAN-only setup completed")
        print(f"  Total FARGAN parameters: {sum(p.numel() for p in fargan_wavehead.parameters()):,}")

    else:
        # 正常训练模式
        encoder = AETHEREncoder(
            d_in=args.feature_dims, d_model=cfg.d_model, dz=cfg.dz,
            gla_depth=cfg.gla_depth, n_heads=cfg.n_heads, d_csi=16,  # 编码器使用16维容纳acoustic_priors
            dropout=cfg.dropout, use_film=True, use_moe=False,
            n_experts=cfg.n_experts, top_k=cfg.top_k,
            latent_bits=cfg.latent_bits, frame_rate_hz=cfg.frame_rate_hz,
            quantize_latent=True, feature_spec_type=args.feature_spec_type
        ).to(device)

        # 根据decoder_type选择解码器 (仅正常训练模式)
        if args.decoder_type == 'aether_fargan':
            from models.aether_fargan_decoder import AETHERFARGANDecoder
            decoder = AETHERFARGANDecoder(
                dz=cfg.dz, d_out=args.feature_dims, d_hidden=cfg.d_model,
                d_csi=cfg.d_csi, decoder_heads=cfg.n_heads,
                enable_synth=True, feature_spec_type=args.feature_spec_type  # 启用FARGAN波形合成
            ).to(device)
            print(f"✅ 使用 AETHERFARGANDecoder: 特征重建 + FARGAN波形合成")
        else:
            decoder = AETHERDecoder(
                dz=cfg.dz, d_out=args.feature_dims, d_hidden=cfg.d_model,
                d_csi=cfg.d_csi, decoder_heads=cfg.n_heads,
                enable_synth=True, feature_spec_type=args.feature_spec_type  # 启用内嵌合成器
            ).to(device)
            print(f"✅ 使用 AETHERDecoder: 仅特征重建")

    # 根据训练模式和解码器类型选择合适的wave_head包装器
    if args.stage2_fargan_only:
        # 阶段二模式：wave_head已经在上面设置为独立的FARGAN波形头
        print(f"阶段二独立FARGAN波形头参数: {sum(p.numel() for p in wave_head.parameters()):,}")
        print("阶段二模式：使用独立FARGAN波形头（不依赖编解码器）")
    elif args.decoder_type == "aether_fargan":
        # FARGAN版本的wave_head包装器
        class FarganWaveHead(nn.Module):
            """FARGAN解码器的波形头包装器"""
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
                self._is_exciter = False

            def forward(self, decoded_feats: torch.Tensor, target_len: int = None, csi_dict=None):
                """
                Args:
                    decoded_feats: [B, T, 36] FARGAN特征 (已经从decoder输出)
                    target_len: 目标波形长度
                    csi_dict: 可选字典；当包含 'fargan_pre' 时，将作为教师强制的前序音频段传入
                Returns:
                    waveform: [B, T_audio] 合成波形
                """
                # 直接使用FARGAN合成器；若提供 fargan_pre 则进行教师强制
                period = self.decoder._estimate_period(decoded_feats)
                fargan_pre = None
                if isinstance(csi_dict, dict):
                    fargan_pre = csi_dict.get('fargan_pre', None)
                audio = self.decoder._generate_waveform(decoded_feats, period, target_len, fargan_pre)
                # 将 pre 段拼接回输出，保持与独立FARGAN训练一致
                if fargan_pre is not None:
                    pre_seg = fargan_pre
                    if pre_seg.dim() == 3:
                        pre_seg = pre_seg.squeeze(1)
                    if audio.dim() == 3:
                        audio = torch.cat([pre_seg.unsqueeze(1), audio], dim=-1)
                    else:
                        audio = torch.cat([pre_seg, audio], dim=-1)
                return audio.squeeze(1) if audio.dim() == 3 else audio  # [B, T_audio]

        wave_head = FarganWaveHead(decoder).to(device)
        print("Using AETHER-FARGAN end-to-end synthesis (features->waveform)")
    else:
        # AETHER版本使用原有的EmbeddedSynthHead
        from models.maybe_useless.decoder_synth_head import EmbeddedSynthHead
        wave_head = EmbeddedSynthHead(decoder).to(device)
        setattr(wave_head, '_is_exciter', False)
        print("Using Decoder-Embedded OLA synthesis (no separate wave head)")

    wave_loss = MRSTFTLoss(
        fft_sizes=(256, 512, 1024, 2048),
        hop_sizes=(80, 160, 320, 640),
        win_sizes=(200, 400, 800, 1600),
        alpha_l1=2.0,
        alpha_mag=1.0,
        alpha_sc=0.08
    ).to(device)

    print(f"🏗️ 模型加载完成")
    print(f"   编码器参数: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"   解码器参数: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"   波形头参数: {sum(p.numel() for p in wave_head.parameters()):,}")

    # 创建训练阶段
    stages = create_progressive_stages()

    # 阶段筛选
    if args.stages != 'all':
        # 创建阶段名称到索引的映射
        stage_name_to_index = {stage.name: i for i, stage in enumerate(stages)}

        stage_indices = []
        for stage_spec in args.stages.split(','):
            stage_spec = stage_spec.strip()
            try:
                # 尝试解析为数字索引（1-based）
                idx = int(stage_spec) - 1
                stage_indices.append(idx)
            except ValueError:
                # 解析为阶段名称
                if stage_spec in stage_name_to_index:
                    stage_indices.append(stage_name_to_index[stage_spec])
                else:
                    available_stages = ', '.join(stage_name_to_index.keys())
                    raise ValueError(f"未知阶段名称 '{stage_spec}'，可用阶段: {available_stages}")

        stages = [stages[i] for i in stage_indices if 0 <= i < len(stages)]
    elif args.start_stage is not None or args.end_stage is not None:
        # 使用start-stage和end-stage
        start_idx = (args.start_stage - 1) if args.start_stage else 0
        end_idx = args.end_stage if args.end_stage else len(stages)
        start_idx = max(0, min(start_idx, len(stages) - 1))
        end_idx = max(1, min(end_idx, len(stages)))
        stages = stages[start_idx:end_idx]

    # 覆盖预热阶段可调参数
    if args.preheat_mix_end is not None:
        for st in stages:
            if st.name == 'wave_preheat':
                st.preheat_mix_end_step = int(args.preheat_mix_end)
                break
    if args.preheat_chunk_frames is not None:
        for st in stages:
            if st.name == 'wave_preheat':
                st.preheat_chunk_frames = int(args.preheat_chunk_frames)
                break

    print(f"📋 训练阶段: {[s.name for s in stages]}")

    # === Auto-resume: if starting from wave_preheat, try to load clean_baseline checkpoint ===
    try:
        if args.resume is None and len(stages) > 0 and stages[0].name == 'wave_preheat':
            # Search for any clean_baseline checkpoint saved by a prior run
            cand = []
            for p in output_dir.glob('stage_*_clean_baseline.pth'):
                try:
                    cand.append((p.stat().st_mtime, p))
                except Exception:
                    pass
            if cand:
                cand.sort(reverse=True)
                ckpt_path = cand[0][1]
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if 'encoder_state_dict' in ckpt:
                    encoder.load_state_dict(ckpt['encoder_state_dict'], strict=False)
                if 'decoder_state_dict' in ckpt:
                    decoder.load_state_dict(ckpt['decoder_state_dict'], strict=False)
                if 'wave_head_state_dict' in ckpt:
                    try:
                        wave_head.load_state_dict(ckpt['wave_head_state_dict'], strict=False)
                    except Exception:
                        pass
                print(f"🔁 已从上次 clean_baseline 检查点恢复: {ckpt_path.name}")
            else:
                print("ℹ️ 未找到 clean_baseline 检查点；建议先跑 Stage1 或使用 --resume")
    except Exception as e:
        print(f"⚠️ 自动恢复失败: {e}")

    # === 阶段二独立FARGAN训练分支 ===
    if args.stage2_fargan_only:
        print("🎯 进入阶段二独立FARGAN训练模式")

        # 验证必要参数
        if args.stage1_checkpoint is None:
            raise ValueError("阶段二模式需要指定 --stage1-checkpoint 参数")

        # 开始阶段二独立FARGAN训练
        result = train_stage2_fargan_only(
            frozen_encoder=encoder,
            frozen_decoder=decoder,
            fargan_wavehead=wave_head,
            train_loader=train_loader,
            device=device,
            args=args,
            checkpoint_dir=output_dir
        )

        print(f"🎉 阶段二独立FARGAN训练完成: {result}")
        return

    # 渐进式训练
    results = []
    for i, stage in enumerate(stages):
        result = train_progressive_stage(
            stage=stage,
            encoder=encoder,
            decoder=decoder,
            wave_head=wave_head,
            wave_loss=wave_loss,
            train_loader=train_loader,
            train_dataset=dataset,
            device=device,
            checkpoint_dir=output_dir,
            current_stage_index=i,
            total_stages=len(stages),
            checkpoint_every=args.checkpoint_every,
            feature_spec_type=args.feature_spec_type,
            decoder_type=args.decoder_type,
            disable_f0_loss=args.disable_f0_loss
        )
        results.append(result)

        # 如果阶段失败且不是最后一个阶段，询问是否继续
        if not result['passed'] and i < len(stages) - 1:
            print(f"\n⚠️ 阶段 {stage.name} 未通过验收条件")
            if not args.skip_passed:
                response = input("是否继续下一阶段? (y/n): ")
                if response.lower() != 'y':
                    print("🛑 训练提前终止")
                    break

    # 训练总结
    print(f"\n{'='*60}")
    print("🎉 渐进式训练完成")
    print(f"{'='*60}")

    passed_stages = sum(1 for r in results if r['passed'])
    print(f"✅ 通过阶段: {passed_stages}/{len(results)}")

    for i, (stage, result) in enumerate(zip(stages, results)):
        status = "✅" if result['passed'] else "❌"
        print(f"  {status} 阶段 {i+1}: {stage.name} (loss: {result['best_loss']:.6f})")

    # 最终音频验证与可视化
    if dataset is not None:
        try:
            encoder.eval()
            decoder.eval()
            wave_head.eval()

            final_models = {"final": (encoder, decoder, wave_head)}
            summary = integrate_audio_validation(
                dataset=dataset,
                trained_models=final_models,
                device=device,
                output_dir=str(output_dir)
            )
            print("🎧 最终音频验证已生成 (audio_validation/final)")
            if summary:
                print(summary)
        except Exception as e:
            print(f"⚠️ 最终音频验证失败: {e}")
        finally:
            encoder.train()
            decoder.train()
            wave_head.train()

    # 保存最终状态
    final_checkpoint = output_dir / "final_model.pth"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'wave_head_state_dict': wave_head.state_dict(),
        'training_results': results,
        'config': cfg.__dict__
    }, final_checkpoint)
    print(f"💾 最终模型保存至: {final_checkpoint}")


if __name__ == "__main__":
    main()
