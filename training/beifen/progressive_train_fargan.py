# -*- coding: utf-8 -*-
"""
AETHER-FARGAN渐进式训练脚本
适配原有的progressive_train_clean.py，支持FARGAN波形生成器
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

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    def _create_grad_scaler(enabled: bool) -> "_GradScaler":
        return _GradScaler(init_scale=256.0, growth_interval=200, enabled=enabled)
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

from models.aether_encoder_decoder import AETHEREncoder
from models.aether_fargan_decoder import AETHERFARGANDecoder
from training.losses import compute_layered_loss
from training.f0_losses import compute_enhanced_f0_loss
from training.fargan_losses import compute_fargan_comprehensive_loss
from utils.real_data_loader import create_aether_data_loader, AETHERRealDataset
from utils.audio_validation_generator import integrate_audio_validation
from models.utils import extract_acoustic_priors
from utils.feature_spec import get_default_feature_spec

# 全局常数
SAMPLE_RATE = 16000
FRAME_HOP_SAMPLES = 160  # 10 ms @ 16 kHz
FARGAN_FRAME_SIZE = 160  # FARGAN帧大小


@dataclass
class FARGANTrainStage:
    """FARGAN训练阶段配置"""
    name: str
    description: str

    steps: Optional[int] = None
    epochs: Optional[float] = None

    use_film: bool = False
    use_moe: bool = False
    apply_channel: bool = False

    # FARGAN特定配置
    enable_fargan_loss: bool = True
    fargan_loss_weight: float = 1.0
    feature_adapter_lr_scale: float = 2.0
    freeze_encoder: bool = False
    freeze_fargan: bool = False

    # 混合策略
    teacher_forcing_steps: int = 1000
    teacher_forcing_start_ratio: float = 1.0

    # 损失权重
    lambda_recon: float = 1.0
    lambda_f0: float = 1.5
    lambda_fargan: float = 1.0
    lambda_feature_align: float = 0.1

    learning_rate: float = 2e-4
    min_convergence_rate: float = 5.0
    max_final_loss: float = 2.0


def _finite_scalar(x: torch.Tensor, name: str, step: int) -> torch.Tensor:
    """确保标量有限"""
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


def _clean_tensor(x: torch.Tensor, clip: float = 1e4) -> torch.Tensor:
    """数值清洗"""
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    if clip is not None and clip > 0:
        x = x.clamp(min=-clip, max=clip)
    return x


def compute_feature_alignment_loss(
    aether_features: torch.Tensor,
    fargan_features: torch.Tensor,
    weight: float = 1.0
) -> torch.Tensor:
    """
    计算AETHER和FARGAN特征的对齐损失
    确保特征适配器保留关键信息

    Args:
        aether_features: [B, T, 48] AETHER特征
        fargan_features: [B, T, 36] FARGAN特征
        weight: 损失权重

    Returns:
        alignment_loss: 特征对齐损失
    """
    from models.feature_adapter import FARGANFeatureSpec

    spec_aether = get_default_feature_spec()
    spec_fargan = FARGANFeatureSpec()

    # 提取对应特征分量
    aether_ceps = spec_aether.extract_feature(aether_features, 'ceps')
    aether_f0 = spec_aether.extract_feature(aether_features, 'f0')
    aether_lpc = spec_aether.extract_feature(aether_features, 'lpc')

    fargan_ceps = spec_fargan.extract_feature(fargan_features, 'ceps')
    fargan_dnn_pitch = spec_fargan.extract_feature(fargan_features, 'dnn_pitch')
    fargan_lpc = spec_fargan.extract_feature(fargan_features, 'lpc')

    # 对齐损失
    ceps_loss = F.mse_loss(fargan_ceps, aether_ceps)
    f0_loss = F.mse_loss(fargan_dnn_pitch, aether_f0)  # 假设尺度相近
    lpc_loss = F.mse_loss(fargan_lpc, aether_lpc)

    total_loss = ceps_loss + f0_loss + lpc_loss
    return weight * total_loss


def train_fargan_stage(
    stage: FARGANTrainStage,
    encoder: nn.Module,
    decoder: AETHERFARGANDecoder,
    train_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path,
    current_stage_index: int,
    total_stages: int,
    checkpoint_every: int = 500
) -> Dict[str, Any]:
    """训练FARGAN阶段"""

    print(f"\n{'='*60}")
    print(f"🚀 开始FARGAN阶段 {current_stage_index+1}/{total_stages}: {stage.name}")
    print(f"📝 描述: {stage.description}")
    print(f"{'='*60}")

    # 计算训练步数
    total_batches = len(train_loader)
    if stage.steps is not None:
        actual_steps = stage.steps
    elif stage.epochs is not None:
        actual_steps = int(stage.epochs * total_batches)
    else:
        actual_steps = total_batches

    print(f"训练配置: {actual_steps} 步, 学习率: {stage.learning_rate}")
    print(f"FARGAN损失权重: {stage.lambda_fargan}, 特征对齐权重: {stage.lambda_feature_align}")

    # 设置模型训练状态
    encoder.train()
    decoder.train()

    # 冻结设置
    if stage.freeze_encoder:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        print("🔒 编码器已冻结")

    if stage.freeze_fargan:
        decoder.fargan_core.eval()
        for param in decoder.fargan_core.parameters():
            param.requires_grad = False
        print("🔒 FARGAN核心已冻结")

    # 创建优化器参数组
    param_groups = []

    if not stage.freeze_encoder:
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': stage.learning_rate,
                'name': 'encoder'
            })

    # 解码器参数 (分组处理)
    if hasattr(decoder, 'feature_adapter') and decoder.feature_adapter is not None:
        adapter_params = [p for p in decoder.feature_adapter.parameters() if p.requires_grad]
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': stage.learning_rate * stage.feature_adapter_lr_scale,
                'name': 'feature_adapter'
            })

    projector_params = [p for p in decoder.latent_projector.parameters() if p.requires_grad]
    if projector_params:
        param_groups.append({
            'params': projector_params,
            'lr': stage.learning_rate * 0.8,
            'name': 'latent_projector'
        })

    if not stage.freeze_fargan:
        fargan_params = [p for p in decoder.fargan_core.parameters() if p.requires_grad]
        if fargan_params:
            param_groups.append({
                'params': fargan_params,
                'lr': stage.learning_rate * 0.5,  # FARGAN用较小学习率
                'name': 'fargan_core'
            })

    period_params = [p for p in decoder.period_estimator.parameters() if p.requires_grad]
    if period_params:
        param_groups.append({
            'params': period_params,
            'lr': stage.learning_rate,
            'name': 'period_estimator'
        })

    if not param_groups:
        raise ValueError(f"阶段 {stage.name}: 没有可训练的参数")

    print(f"🔧 优化器参数组:")
    for group in param_groups:
        print(f"  {group['name']}: {len(group['params'])} 参数, lr={group['lr']:.2e}")

    optimizer = optim.AdamW(param_groups, weight_decay=1e-6)
    scaler = _create_grad_scaler(enabled=(device.type == 'cuda'))

    # 训练循环
    best_loss = float('inf')
    step = 0

    def batch_gen():
        while True:
            for batch in train_loader:
                yield batch

    batch_iter = batch_gen()

    for step in range(1, actual_steps + 1):
        batch = next(batch_iter)
        optimizer.zero_grad()

        # 数据预处理
        x_gpu = batch['x'].to(device, non_blocking=True)
        y_gpu = batch['y'].to(device, non_blocking=True)
        csi_dict = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.get('csi', {}).items()}

        x = _clean_tensor(x_gpu)
        y = _clean_tensor(y_gpu)
        original_audio = batch.get('audio')
        if original_audio is not None:
            original_audio = original_audio.to(device, non_blocking=True)

        # 教师强制比例计算
        if step <= stage.teacher_forcing_steps:
            teacher_ratio = stage.teacher_forcing_start_ratio * (1.0 - step / stage.teacher_forcing_steps)
        else:
            teacher_ratio = 0.0

        with _autocast_ctx(enabled=(device.type == 'cuda')):
            try:
                # 编码
                z, enc_logs = encoder(x, csi_dict=csi_dict, inference=False)

                # 解码为特征
                csi_dec = dict(csi_dict)
                try:
                    csi_dec["acoustic_priors"] = extract_acoustic_priors(x).detach()
                except Exception:
                    pass

                fargan_features, period = decoder._forward_features(z, csi_dec)

                # 特征域损失
                recon_loss = torch.zeros((), device=device)

                # 可选: 特征对齐损失
                if stage.lambda_feature_align > 0:
                    feature_align_loss = compute_feature_alignment_loss(
                        x, fargan_features, weight=stage.lambda_feature_align
                    )
                    recon_loss = recon_loss + feature_align_loss

                # F0损失 (在FARGAN特征上)
                f0_loss_val = torch.zeros((), device=device)
                if stage.lambda_f0 > 0:
                    try:
                        # 使用FARGAN特征计算F0损失
                        from models.feature_adapter import FARGANFeatureSpec
                        spec_fargan = FARGANFeatureSpec()

                        f0_pred = spec_fargan.extract_feature(fargan_features, 'dnn_pitch')
                        f0_target = get_default_feature_spec().extract_feature(y, 'f0')

                        # 简单MSE损失 (可以后续增强)
                        f0_loss_val = F.mse_loss(f0_pred, f0_target) * stage.lambda_f0
                    except Exception as e:
                        print(f"⚠️ F0损失计算失败: {e}")

                # 波形域损失
                wave_loss_val = torch.zeros((), device=device)
                fargan_loss_val = torch.zeros((), device=device)

                if stage.enable_fargan_loss and original_audio is not None:
                    # 生成波形
                    feature_frames = fargan_features.size(1)
                    target_len = feature_frames * FARGAN_FRAME_SIZE

                    # 准备原始音频
                    while original_audio.dim() > 2:
                        original_audio = original_audio.squeeze(1)
                    if original_audio.dim() == 1:
                        original_audio = original_audio.unsqueeze(0)

                    current_len = original_audio.size(-1)
                    if current_len < target_len:
                        original_audio = F.pad(original_audio, (0, target_len - current_len))
                    elif current_len > target_len:
                        original_audio = original_audio[..., :target_len]

                    # 教师强制
                    pre_audio = None
                    if teacher_ratio > 0 and random.random() < teacher_ratio:
                        pre_frames = max(1, int(teacher_ratio * 4))  # 最多4帧
                        pre_len = pre_frames * FARGAN_FRAME_SIZE
                        if original_audio.size(-1) >= pre_len:
                            pre_audio = original_audio[..., :pre_len]

                    # 生成音频
                    try:
                        fargan_features_out, audio_pred = decoder(
                            z, csi_dec, return_wave=True, target_len=target_len, pre_audio=pre_audio
                        )

                        if audio_pred.dim() == 3 and audio_pred.size(1) == 1:
                            audio_pred = audio_pred.squeeze(1)

                        # FARGAN损失
                        fargan_loss_val, fargan_loss_dict = compute_fargan_comprehensive_loss(
                            audio_pred, original_audio, period
                        )
                        fargan_loss_val = fargan_loss_val * stage.lambda_fargan

                    except Exception as e:
                        print(f"⚠️ FARGAN波形生成失败: {e}")
                        fargan_loss_val = torch.zeros((), device=device)

                # 总损失
                total_loss = (
                    _finite_scalar(stage.lambda_recon * recon_loss, "recon", step) +
                    _finite_scalar(f0_loss_val, "f0", step) +
                    _finite_scalar(fargan_loss_val, "fargan", step)
                )

            except Exception as e:
                print(f"❌ 前向传播失败 at step {step}: {e}")
                continue

        # 数值稳定性检查
        if not torch.isfinite(total_loss).all():
            print(f"⚠️ 步骤 {step}: 检测到损失异常 (NaN/Inf), 跳过此步")
            continue

        # 反向传播
        try:
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    max_norm=1.0
                )
                optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ 步骤 {step}: GPU内存不足, 跳过此步")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"⚠️ 步骤 {step}: 反向传播异常: {e}")
                continue

        # 记录损失
        current_loss = float(total_loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss

        # 中间检查点
        if step % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"fargan_stage_{current_stage_index}_{stage.name}_step_{step}.pth"
            torch.save({
                'stage_index': current_stage_index,
                'stage_name': stage.name,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'step': step,
                'best_loss': best_loss,
                'current_loss': current_loss,
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            print(f"  💾 检查点已保存: {checkpoint_path.name}")

        # 日志输出
        if step % 50 == 0 or step <= 10:
            fargan_loss_info = ""
            if 'fargan_loss_val' in locals() and fargan_loss_val.item() > 0:
                fargan_loss_info = f" fargan={fargan_loss_val.item():.6f}"

            f0_loss_info = ""
            if 'f0_loss_val' in locals() and f0_loss_val.item() > 0:
                f0_loss_info = f" f0={f0_loss_val.item():.6f}"

            print(f"  步骤 {step}/{actual_steps}: "
                  f"loss={current_loss:.6f}{f0_loss_info}{fargan_loss_info} "
                  f"best={best_loss:.6f} teacher={teacher_ratio:.3f}")

    # 阶段结束
    final_checkpoint = checkpoint_dir / f"fargan_stage_{current_stage_index}_{stage.name}_final.pth"
    torch.save({
        'stage_index': current_stage_index,
        'stage_name': stage.name,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'step': actual_steps,
        'best_loss': best_loss,
        'final_loss': current_loss,
    }, final_checkpoint)

    return {
        'stage_name': stage.name,
        'steps': actual_steps,
        'best_loss': best_loss,
        'final_loss': current_loss,
        'checkpoint_path': str(final_checkpoint)
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--device', type=str, default='auto', help='设备')

    args = parser.parse_args()

    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"🚀 使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    print("📂 加载数据...")
    train_loader, train_dataset = create_aether_data_loader(
        feature_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=100,
        num_workers=4
    )
    print(f"训练数据: {len(train_dataset)} 样本, {len(train_loader)} 批次")

    # 创建模型
    print("🏗️ 创建AETHER-FARGAN模型...")
    encoder = AETHEREncoder(
        d_in=48, d_model=128, dz=24,
        use_film=True, use_moe=True
    ).to(device)

    decoder = AETHERFARGANDecoder(
        dz=24, d_csi=32, enable_feature_adapter=True
    ).to(device)

    print(f"编码器参数: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"解码器参数: {sum(p.numel() for p in decoder.parameters()):,}")

    # 训练阶段配置
    stages = [
        FARGANTrainStage(
            name="adapter_warmup",
            description="特征适配器预热",
            steps=2000,
            freeze_fargan=True,
            lambda_feature_align=1.0,
            lambda_fargan=0.0,
            teacher_forcing_steps=1000,
            learning_rate=3e-4
        ),
        FARGANTrainStage(
            name="joint_training",
            description="联合训练",
            steps=5000,
            lambda_feature_align=0.1,
            lambda_fargan=1.0,
            teacher_forcing_steps=2000,
            learning_rate=2e-4
        ),
        FARGANTrainStage(
            name="end_to_end",
            description="端到端优化",
            steps=3000,
            lambda_feature_align=0.05,
            lambda_fargan=2.0,
            teacher_forcing_steps=500,
            learning_rate=1e-4
        )
    ]

    # 执行训练
    results = []
    for i, stage in enumerate(stages):
        print(f"\n{'='*80}")
        print(f"🎯 执行阶段 {i+1}/{len(stages)}: {stage.name}")
        print(f"{'='*80}")

        try:
            result = train_fargan_stage(
                stage, encoder, decoder, train_loader, device,
                checkpoint_dir, i, len(stages)
            )
            results.append(result)
            print(f"✅ 阶段 {stage.name} 完成: 最佳损失 {result['best_loss']:.6f}")

        except Exception as e:
            print(f"❌ 阶段 {stage.name} 失败: {e}")
            traceback.print_exc()
            break

    print(f"\n🎉 AETHER-FARGAN训练完成!")
    for result in results:
        print(f"  {result['stage_name']}: {result['best_loss']:.6f}")


if __name__ == "__main__":
    main()