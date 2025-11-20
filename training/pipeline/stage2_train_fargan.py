#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARGAN训练脚本
使用真实36维特征和音频数据训练FARGAN声码器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import time
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from models.fargan_decoder import FARGANDecoder
from training.fargan_losses import (
    compute_fargan_comprehensive_loss,
    compute_fargan_training_loss,
    compute_fargan_original_style_loss,
)


def create_small_dataset(
    feature_file: str,
    audio_file: str,
    output_dir: Path,
    sample_size: int = 100,
    sequence_length: int = 200
):
    """创建小样本数据集用于快速测试"""
    print(f"\n创建小样本数据集 (样本数: {sample_size})...")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始数据
    features = np.fromfile(feature_file, dtype=np.float32)
    features = features[:features.size // 36 * 36].reshape(-1, 36)

    audio = np.fromfile(audio_file, dtype=np.int16).astype(np.float32) / 32768.0

    # 计算需要的数据量
    feature_samples_needed = sample_size * sequence_length
    audio_samples_needed = sample_size * sequence_length * 160

    if len(features) < feature_samples_needed:
        print(f"警告: 特征数据不足，需要{feature_samples_needed}，实际{len(features)}")
        sample_size = len(features) // sequence_length

    if len(audio) < audio_samples_needed:
        print(f"警告: 音频数据不足，需要{audio_samples_needed}，实际{len(audio)}")
        sample_size = min(sample_size, len(audio) // (sequence_length * 160))

    # 重新计算实际数据量
    feature_samples_needed = sample_size * sequence_length
    audio_samples_needed = sample_size * sequence_length * 160

    # 截取数据
    small_features = features[:feature_samples_needed]
    small_audio = audio[:audio_samples_needed]

    # 保存小样本数据
    small_feature_file = output_dir / 'small_features.f32'
    small_audio_file = output_dir / 'small_audio.pcm'

    small_features.astype(np.float32).tofile(small_feature_file)
    (small_audio * 32767).astype(np.int16).tofile(small_audio_file)

    print(f"小样本数据集创建完成:")
    print(f"  样本数: {sample_size}")
    print(f"  特征文件: {small_feature_file} ({small_features.shape})")
    print(f"  音频文件: {small_audio_file} ({small_audio.shape})")
    print(f"  总时长: {len(small_audio) / 16000:.1f}秒")

    return str(small_feature_file), str(small_audio_file)


class FARGANDataset(Dataset):
    """FARGAN训练数据集"""

    def __init__(
        self,
        feature_file: str,
        audio_file: str,
        sequence_length: int = 200,
        frame_size: int = 160,
        max_samples: Optional[int] = None
    ):
        self.sequence_length = sequence_length
        self.frame_size = frame_size

        # 加载特征数据 (36维)
        features = np.fromfile(feature_file, dtype=np.float32)
        if features.size % 36 != 0:
            features = features[:features.size // 36 * 36]
        features = features.reshape(-1, 36)

        # 加载音频数据
        audio = np.fromfile(audio_file, dtype=np.int16).astype(np.float32) / 32768.0

        # 计算有效序列数量
        max_feature_sequences = features.shape[0] // sequence_length
        max_audio_sequences = len(audio) // (sequence_length * frame_size)
        max_sequences = min(max_feature_sequences, max_audio_sequences)

        if max_samples is not None:
            max_sequences = min(max_sequences, max_samples)

        # 截取对齐的数据
        self.num_sequences = max_sequences
        feature_samples = max_sequences * sequence_length
        audio_samples = max_sequences * sequence_length * frame_size

        self.features = features[:feature_samples].reshape(max_sequences, sequence_length, 36)
        self.audio = audio[:audio_samples].reshape(max_sequences, sequence_length * frame_size)

        print(f"数据集初始化完成:")
        print(f"  序列数量: {self.num_sequences}")
        print(f"  特征形状: {self.features.shape}")
        print(f"  音频形状: {self.audio.shape}")
        print(f"  每序列时长: {sequence_length * frame_size / 16000:.2f}秒")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # 预转换为tensor并缓存，避免重复转换
        features = torch.from_numpy(self.features[idx]).float()  # [T, 36]
        audio = torch.from_numpy(self.audio[idx]).float()        # [T*160]

        return features, audio


def create_data_loaders(
    feature_file: str,
    audio_file: str,
    batch_size: int = 8,
    sequence_length: int = 200,
    train_ratio: float = 0.9,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""

    # 创建完整数据集
    full_dataset = FARGANDataset(
        feature_file, audio_file,
        sequence_length=sequence_length,
        max_samples=max_samples
    )

    # 分割训练和验证集
    num_train = int(len(full_dataset) * train_ratio)
    num_val = len(full_dataset) - num_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [num_train, num_val]
    )

    # 创建数据加载器 - 优化加载速度
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # 增加worker数量
        pin_memory=True,
        persistent_workers=True,  # 保持worker存活
        prefetch_factor=4  # 预取更多batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"数据加载器创建完成:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  批大小: {batch_size}")

    return train_loader, val_loader


def train_epoch(
    model: FARGANDecoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    use_step_scheduler: bool = False,
    original_style_epochs: int = 0,
    ramp_epochs: int = 0,
) -> Dict[str, float]:
    """训练一个epoch"""

    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_components: Dict[str, float] = {}

    start_time = time.time()
    ema_loss: Optional[float] = None

    with tqdm(train_loader, desc=f"Train {epoch}", leave=False) as pbar:
        for batch_idx, (features, target_audio) in enumerate(pbar):
            features = features.to(device, non_blocking=True)
            target_audio = target_audio.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                nb_pre_frames = 2
                pre = target_audio[..., : nb_pre_frames * 160]

                period, pred_audio = model(features, pre=pre)
                pred_audio = pred_audio.squeeze(1)
                pred_audio = torch.cat([pre, pred_audio], dim=-1)

                min_len = min(pred_audio.size(-1), target_audio.size(-1))
                pred_audio = pred_audio[..., :min_len]
                target_audio = target_audio[..., :min_len]

                if original_style_epochs > 0:
                    if epoch <= original_style_epochs:
                        alpha = 0.0
                    else:
                        alpha = 1.0 if ramp_epochs <= 0 else min(1.0, (epoch - original_style_epochs) / float(ramp_epochs))
                    orig_loss, orig_dict = compute_fargan_original_style_loss(
                        pred_audio, target_audio, device=device,
                        frame_size=160, focus_start=nb_pre_frames * 160,
                    )
                    comp_loss, comp_dict = compute_fargan_training_loss(
                        pred_audio, target_audio, period, device=device
                    )
                    fargan_loss = (1.0 - alpha) * orig_loss + alpha * comp_loss
                    loss_dict = {**{f'orig_{k}': v for k, v in orig_dict.items()}, **comp_dict}
                else:
                    fargan_loss, loss_dict = compute_fargan_training_loss(
                        pred_audio, target_audio, period, device=device
                    )

                fargan_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                if use_step_scheduler and scheduler is not None:
                    scheduler.step()

                batch_size = features.size(0)
                loss_val = float(fargan_loss.item())
                total_loss += loss_val * batch_size
                total_samples += batch_size

                for key in ['train_l1', 'train_l2', 'train_energy', 'train_mr_stft',
                            'fargan_signal', 'fargan_continuity', 'fargan_pitch_consistency', 'fargan_subframe_alignment']:
                    if key in loss_dict:
                        loss_components[key] = loss_components.get(key, 0.0) + float(loss_dict[key].item()) * batch_size

                ema_loss = loss_val if ema_loss is None else 0.97 * ema_loss + 0.03 * loss_val
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}", ema=f"{ema_loss:.4f}",
                    l1=f"{float(loss_dict.get('train_l1', 0.0)):.4f}",
                    mr=f"{float(loss_dict.get('train_mr_stft', 0.0)):.4f}",
                    lr=f"{lr:.2e}"
                )

            except Exception as e:
                print(f"训练批次 {batch_idx} 失败: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    for key in list(loss_components.keys()):
        loss_components[key] /= total_samples

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} 训练完成: 平均损失={avg_loss:.6f}, 时间={epoch_time:.1f}s")

    return {'total_loss': avg_loss, **loss_components}


def validate_epoch(
    model: FARGANDecoder,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    original_style_epochs: int = 0,
    ramp_epochs: int = 0,
) -> Dict[str, float]:
    """验证一个epoch"""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_components: Dict[str, float] = {}

    with torch.no_grad():
        ema_v: Optional[float] = None
        with tqdm(val_loader, desc=f"Val {epoch}", leave=False) as pbar:
            for batch_idx, (features, target_audio) in enumerate(pbar):
                try:
                    features = features.to(device)
                    target_audio = target_audio.to(device)

                    period, pred_audio = model(features)
                    pred_audio = pred_audio.squeeze(1)

                    min_len = min(pred_audio.size(-1), target_audio.size(-1))
                    pred_audio = pred_audio[..., :min_len]
                    target_audio = target_audio[..., :min_len]

                    if original_style_epochs > 0:
                        if epoch <= original_style_epochs:
                            alpha = 0.0
                        else:
                            alpha = 1.0 if ramp_epochs <= 0 else min(1.0, (epoch - original_style_epochs) / float(ramp_epochs))
                        orig_loss, orig_dict = compute_fargan_original_style_loss(
                            pred_audio, target_audio, device=device,
                            frame_size=160, focus_start=0
                        )
                        comp_loss, comp_dict = compute_fargan_training_loss(
                            pred_audio, target_audio, period, device=device
                        )
                        fargan_loss = (1.0 - alpha) * orig_loss + alpha * comp_loss
                        loss_dict = {**{f'orig_{k}': v for k, v in orig_dict.items()}, **comp_dict}
                    else:
                        fargan_loss, loss_dict = compute_fargan_training_loss(
                            pred_audio, target_audio, period, device=device
                        )

                    batch_size = features.size(0)
                    v = float(fargan_loss.item())
                    total_loss += v * batch_size
                    total_samples += batch_size

                    for key, value in loss_dict.items():
                        loss_components[key] = loss_components.get(key, 0.0) + value.item() * batch_size

                    ema_v = v if ema_v is None else 0.97 * ema_v + 0.03 * v
                    pbar.set_postfix(loss=f"{v:.4f}", ema=f"{ema_v:.4f}")

                except Exception as e:
                    print(f"验证批次 {batch_idx} 失败: {e}")
                    continue

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    for key in list(loss_components.keys()):
        loss_components[key] /= total_samples

    print(f"Epoch {epoch} 验证完成: 平均损失={avg_loss:.6f}")
    return {'total_loss': avg_loss, **loss_components}


def evaluate_fargan_decoder(
    model: FARGANDecoder,
    feature_file: str,
    output_pcm: Path,
    device: torch.device,
    de_emphasis: float = 0.85
) -> dict:
    """使用FARGANDecoder对整段36维特征进行合成并导出PCM（类似原 test_fargan.py）。

    - 直接使用36维特征；周期在解码器内部由DNN pitch估计。
    - 对输出应用简单去加重滤波 y[n] = x[n] + a*y[n-1]（a=de_emphasis）。
    """
    model.eval()

    # 加载特征 [1, T, 36]
    feats = np.fromfile(feature_file, dtype=np.float32)
    feats = feats[: (feats.size // 36) * 36].reshape(1, -1, 36)

    feats_t = torch.from_numpy(feats).float().to(device)

    with torch.no_grad():
        # 让解码器自行推断周期
        _, audio = model(feats_t)
        wav = audio.squeeze().detach().cpu().numpy().astype(np.float32)

    # 简单去加重滤波（避免引入scipy依赖）
    if de_emphasis is not None and de_emphasis != 0.0:
        y = np.zeros_like(wav, dtype=np.float32)
        prev = 0.0
        a = float(de_emphasis)
        for i in range(len(wav)):
            prev = wav[i] + a * prev
            y[i] = prev
        wav = y

    # 导出PCM16
    output_pcm.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(wav, -0.99, 0.99) * 32768.0).astype(np.int16)
    pcm.tofile(str(output_pcm))

    return {
        'samples': int(pcm.size),
        'min': float(wav.min()) if wav.size else 0.0,
        'max': float(wav.max()) if wav.size else 0.0,
        'path': str(output_pcm),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    feature_file: str,
    output_pcm: str,
    device: torch.device
) -> dict:
    """加载训练生成的checkpoint并做一次整段合成评估（无打分，仅导出PCM）。

    兼容 final_version 保存结构（model_state_dict），若不存在则尝试 state_dict。
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    model = FARGANDecoder().to(device)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', None))
    if state_dict is None:
        raise RuntimeError('Checkpoint missing model state dict.')
    model.load_state_dict(state_dict, strict=False)

    return evaluate_fargan_decoder(model, feature_file, Path(output_pcm), device)


def save_checkpoint(
    model: FARGANDecoder,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: Path
):
    """保存检查点"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # 保存最新检查点
    latest_path = checkpoint_dir / 'latest_checkpoint.pth'
    torch.save(checkpoint, latest_path)

    # 保存最佳模型
    best_path = checkpoint_dir / 'best_model.pth'
    if not best_path.exists():
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型: {best_path}")
    else:
        best_checkpoint = torch.load(best_path)
        if val_loss < best_checkpoint['val_loss']:
            torch.save(checkpoint, best_path)
            print(f"更新最佳模型: {best_path} (验证损失: {val_loss:.6f})")


def plot_training_curves(train_losses: list, val_losses: list, output_dir: Path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')

    plt.title('FARGAN训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"训练曲线已保存: {output_dir / 'training_curves.png'}")


## 移除独立的测试环节，减少训练脚本内非必要流程


def main():
    parser = argparse.ArgumentParser(description='训练FARGAN声码器')
    parser.add_argument('--feature-file', type=str,
                        default='/home/bluestar/FARGAN/opus/data_cn/out_features.f32',
                        help='36维特征文件路径')
    parser.add_argument('--audio-file', type=str,
                        default='/home/bluestar/FARGAN/opus/data_cn/out_speech.pcm',
                        help='音频文件路径')
    parser.add_argument('--output-dir', type=str, default='./fargan_training_results',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--original-style-epochs', type=int, default=0,
                        help='前N个epoch使用原版风格（谱收敛+单帧sig_loss）')
    parser.add_argument('--ramp-epochs', type=int, default=0,
                        help='在此后N个epoch内线性过渡到综合损失')
    parser.add_argument('--orig-lr-decay', type=float, default=1e-4,
                        help='原版风格下每step的LambdaLR衰减系数')
    parser.add_argument('--sequence-length', type=int, default=200,
                        help='序列长度')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='最大训练样本数 (用于快速测试)')
    parser.add_argument('--create-small-dataset', action='store_true',
                        help='创建小样本数据集进行快速测试')
    parser.add_argument('--small-dataset-size', type=int, default=100,
                        help='小样本数据集大小')

    args = parser.parse_args()

    print("FARGAN声码器训练")
    print("=" * 50)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 0. 如果需要，创建小样本数据集
        feature_file = args.feature_file
        audio_file = args.audio_file

        if args.create_small_dataset:
            small_data_dir = output_dir / 'small_dataset'
            feature_file, audio_file = create_small_dataset(
                args.feature_file,
                args.audio_file,
                small_data_dir,
                sample_size=args.small_dataset_size,
                sequence_length=args.sequence_length
            )
            # 使用小数据集时不限制max_samples
            args.max_samples = None

        # 1. 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            feature_file,
            audio_file,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            max_samples=args.max_samples
        )

        # 2. 创建模型
        model = FARGANDecoder().to(device)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 3. 创建优化器/调度器（original-style 对齐）
        if args.original_style_epochs > 0:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,  # 不乘5
                weight_decay=1e-5,
                eps=1e-8,
                betas=(0.8, 0.95),
            )
            # 逐step LambdaLR：lr = lr0 / (1 + decay * step)
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda s: 1.0 / (1.0 + args.orig_lr_decay * float(s))
            )
            use_step_scheduler = True
        else:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.learning_rate * 5,
                weight_decay=1e-5,
                eps=1e-8,
                betas=(0.9, 0.999),
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=2
            )
            use_step_scheduler = False

        # 4. 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        print(f"\n开始训练 {args.epochs} 个epoch...")

        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

            # 训练
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, epoch,
                scheduler=scheduler if use_step_scheduler else None,
                use_step_scheduler=use_step_scheduler,
                original_style_epochs=args.original_style_epochs,
                ramp_epochs=args.ramp_epochs,
            )

            # 验证
            val_metrics = validate_epoch(
                model, val_loader, device, epoch,
                original_style_epochs=args.original_style_epochs,
                ramp_epochs=args.ramp_epochs,
            )

            # 记录损失
            train_losses.append(train_metrics['total_loss'])
            val_losses.append(val_metrics['total_loss'])

            # 更新学习率调度器
            if not use_step_scheduler:
                scheduler.step(val_metrics['total_loss'])

            # 保存检查点
            save_checkpoint(
                model, optimizer, epoch,
                train_metrics['total_loss'], val_metrics['total_loss'],
                output_dir / 'checkpoints'
            )

            # 移除阶段性测试

        # 5. 绘制训练曲线
        plot_training_curves(train_losses, val_losses, output_dir)

        # 6. 训练结束
        print(f"\n训练完成")

        # 保存训练配置 - 确保JSON序列化兼容
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'sequence_length': args.sequence_length,
            'best_val_loss': float(min(val_losses)),
            'final_train_loss': float(train_losses[-1]) if len(train_losses) else None,
            'final_val_loss': float(val_losses[-1]) if len(val_losses) else None,
        }

        with open(output_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n训练结果已保存到: {output_dir}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
