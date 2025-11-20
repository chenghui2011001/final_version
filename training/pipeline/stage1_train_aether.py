#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Train AETHER encoder/decoder for feature reconstruction.

This stage focuses purely on the 36‑dim FARGAN acoustic feature reconstruction
task and does not include waveform synthesis losses.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from contextlib import contextmanager

if __package__ is None or __package__ == "":
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "dnn").exists():
            sys.path.insert(0, str(parent))
            final_version_dir = current_file.parents[2]  # .../dnn/torch/final_version
            if str(final_version_dir) not in sys.path:
                sys.path.append(str(final_version_dir))
            break

from dnn.torch.final_version.models.aether_encoder_decoder import AETHEREncoder, AETHERDecoder
from dnn.torch.final_version.training.losses import compute_layered_loss, rate_loss
from dnn.torch.final_version.utils.real_data_loader import create_aether_data_loader, create_train_val_loaders
from dnn.torch.final_version.training.pipeline.stages import StageConfig, get_stage_config
import json
import numpy as np


def compute_detailed_feature_stats(y_hat: torch.Tensor, y_ref: torch.Tensor, feature_spec_type: str) -> Dict[str, float]:
    """计算详细的特征级别统计信息"""
    feature_stats = {}

    try:
        if feature_spec_type == "fargan":
            from models.feature_adapter import get_fargan_feature_spec
            spec = get_fargan_feature_spec()
            feature_names = ['ceps', 'dnn_pitch', 'frame_corr', 'lpc']
        else:
            from utils.feature_spec import get_default_feature_spec
            spec = get_default_feature_spec()
            feature_names = ['ceps', 'f0', 'voicing', 'lpc', 'enhanced', 'prosodic']

        for name in feature_names:
            try:
                # 获取特征切片
                if hasattr(spec, 'get_feature_slice'):
                    slice_obj = spec.get_feature_slice(name)
                    pred_feat = y_hat[..., slice_obj]
                    target_feat = y_ref[..., slice_obj]
                else:
                    continue  # 跳过不支持的特征

                # 计算详细指标
                feat_mae = torch.abs(pred_feat - target_feat).mean()
                feat_mse = torch.square(pred_feat - target_feat).mean()
                feat_rmse = torch.sqrt(feat_mse)

                # 相对误差
                target_norm = torch.norm(target_feat, dim=-1, keepdim=True) + 1e-8
                relative_error = torch.norm(pred_feat - target_feat, dim=-1, keepdim=True) / target_norm
                feat_rel_error = relative_error.mean()

                # 方差分析
                feat_var_pred = pred_feat.var()
                feat_var_target = target_feat.var()
                feat_var_ratio = feat_var_pred / (feat_var_target + 1e-8)

                # 皮尔逊相关系数
                try:
                    pred_flat = pred_feat.flatten().float()
                    target_flat = target_feat.flatten().float()
                    if len(pred_flat) > 1 and torch.std(pred_flat) > 1e-8 and torch.std(target_flat) > 1e-8:
                        correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                        feat_corr = float(correlation.item()) if not torch.isnan(correlation) else 0.0
                    else:
                        feat_corr = 0.0
                except:
                    feat_corr = 0.0

                # 保存统计结果
                feature_stats[f'{name}_mae'] = float(feat_mae.item())
                feature_stats[f'{name}_mse'] = float(feat_mse.item())
                feature_stats[f'{name}_rmse'] = float(feat_rmse.item())
                feature_stats[f'{name}_rel_error'] = float(feat_rel_error.item())
                feature_stats[f'{name}_var_pred'] = float(feat_var_pred.item())
                feature_stats[f'{name}_var_target'] = float(feat_var_target.item())
                feature_stats[f'{name}_var_ratio'] = float(feat_var_ratio.item())
                feature_stats[f'{name}_correlation'] = feat_corr

            except Exception as e:
                print(f"      Warning: Failed to compute stats for feature '{name}': {e}")
                continue

    except Exception as e:
        print(f"    Warning: Feature analysis failed: {e}")

    return feature_stats


def validate_epoch(encoder: nn.Module,
                   decoder: nn.Module,
                   device: torch.device,
                   autocast_mode: str,
                   feature_spec_type: str,
                   validation_batch: Dict,
                   epoch: int,
                   out_dir: Path,
                   current_step: int = 0,
                   focus_start_frames: int = 0,
                   save_audio: bool = False,
                   max_val_batches: int = 100) -> Dict[str, float]:
    """轻量级验证钩子：解码固定批次并生成特征/方差摘要"""
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # 在验证集上运行多个批次
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        total_rel_error = 0.0
        total_pred_var = 0.0
        total_target_var = 0.0
        total_samples = 0

        all_loss_details = {}
        stage = "unknown"

        # 保存第一个批次用于音频样本
        first_batch_data = None

        batch_count = 0
        for batch in validation_batch:
            if batch_count >= max_val_batches:
                break

            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            batch_size = x.size(0)

            # 构建CSI字典
            csi_dict = {
                'snr_db': torch.zeros(batch_size, device=device, dtype=x.dtype),
                'ber': torch.zeros(batch_size, device=device, dtype=x.dtype),
                'fading_onehot': torch.zeros(batch_size, 8, device=device, dtype=x.dtype)
            }
            csi_dict['fading_onehot'][:, 0] = 1.0

            # 前向传播
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(autocast_mode == "bf16")):
                z, _ = encoder(x, csi_dict)
                y_hat_full = decoder(z, csi_dict)

                # Teacher-forcing-like warm start: ignore first N frames in evaluation
                fs = max(0, int(focus_start_frames))
                if fs > 0 and y_hat_full.size(1) > fs:
                    y_hat = y_hat_full[:, fs:, :]
                    y_ref = y[:, fs:, :]
                else:
                    y_hat = y_hat_full
                    y_ref = y

                # 计算损失（使用与训练一致的current_step控制特征阶段）
                recon, loss_details, stage = compute_layered_loss(
                    y_hat, y_ref, current_step=current_step, feature_spec_type=feature_spec_type
                )

            # 计算详细指标
            mae = torch.abs(y_hat - y_ref).mean()
            mse = torch.square(y_hat - y_ref).mean()
            rmse = torch.sqrt(mse)

            # 相对误差
            target_norm = torch.norm(y_ref, dim=-1, keepdim=True) + 1e-8
            relative_error = torch.norm(y_hat - y_ref, dim=-1, keepdim=True) / target_norm
            rel_err_mean = relative_error.mean()

            # 特征方差分析
            pred_var = y_hat.var(dim=1).mean()
            target_var = y_ref.var(dim=1).mean()

            # 累加指标
            total_loss += recon.item() * batch_size
            total_mae += mae.item() * batch_size
            total_rmse += rmse.item() * batch_size
            total_rel_error += rel_err_mean.item() * batch_size
            total_pred_var += pred_var.item() * batch_size
            total_target_var += target_var.item() * batch_size
            total_samples += batch_size

            # 累加分层损失详情（只处理数值类型）
            for k, v in loss_details.items():
                if isinstance(v, (int, float)):  # 只处理数值类型
                    if k not in all_loss_details:
                        all_loss_details[k] = 0.0
                    all_loss_details[k] += v * batch_size
                elif k == 'stage':  # 保存阶段信息（非数值）
                    stage = v

            # 保存第一个批次用于音频样本
            if batch_count == 0:
                first_batch_data = {
                    'y_hat_full': y_hat_full,
                    'y': y,
                    'x': x
                }

            batch_count += 1

        # 计算平均值
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_mae = total_mae / total_samples
            avg_rmse = total_rmse / total_samples
            avg_rel_error = total_rel_error / total_samples
            avg_pred_var = total_pred_var / total_samples
            avg_target_var = total_target_var / total_samples
            var_ratio = avg_pred_var / (avg_target_var + 1e-8)

            # 平均分层损失详情
            for k in all_loss_details:
                all_loss_details[k] /= total_samples
        else:
            avg_loss = avg_mae = avg_rmse = avg_rel_error = 0.0
            avg_pred_var = avg_target_var = var_ratio = 0.0

        # 特征级别统计（使用第一个批次）
        feature_stats = {}
        if first_batch_data is not None:
            y_hat = first_batch_data['y_hat_full']
            y_ref = first_batch_data['y']

            if feature_spec_type == "fargan":
                from models.feature_adapter import get_fargan_feature_spec
                spec = get_fargan_feature_spec()
                feature_names = ['ceps', 'dnn_pitch', 'frame_corr', 'lpc']
            else:
                from utils.feature_spec import get_default_feature_spec
                spec = get_default_feature_spec()
                feature_names = ['ceps', 'f0', 'voicing', 'lpc', 'enhanced', 'prosodic']

            for name in feature_names:
                try:
                    pred_feat = spec.extract_feature(y_hat, name)
                    target_feat = spec.extract_feature(y_ref, name)

                    feat_mae = torch.abs(pred_feat - target_feat).mean()
                    feat_var_pred = pred_feat.var()
                    feat_var_target = target_feat.var()

                    feature_stats[f'{name}_mae'] = float(feat_mae.item())
                    feature_stats[f'{name}_var_pred'] = float(feat_var_pred.item())
                    feature_stats[f'{name}_var_target'] = float(feat_var_target.item())
                except:
                    continue

        # 组装验证指标
        val_metrics = {
            'val_loss': avg_loss,
            'val_mae': avg_mae,
            'val_rmse': avg_rmse,
            'val_rel_error': avg_rel_error,
            'val_pred_var': avg_pred_var,
            'val_target_var': avg_target_var,
            'val_var_ratio': var_ratio,
            'val_batches_processed': batch_count,
            'val_samples_total': total_samples,
            'stage': stage,
            **{f'val_loss_{k}': v for k, v in all_loss_details.items()},
            **{f'val_{k}': v for k, v in feature_stats.items()}
        }

        # 可选：保存音频片段用于听力测试
        if save_audio:
            try:
                # 创建音频输出目录
                audio_dir = out_dir / 'validation_audio'
                audio_dir.mkdir(exist_ok=True)

                # 保存重建特征和目标特征的对比样本（取第一个样本的前1秒）
                sample_len = min(400, y_hat_full.size(1))  # 最多1秒 (400帧 @ 50fps)
                y_hat_sample = y_hat_full[0, :sample_len].cpu().numpy()  # [T, 36]
                y_sample = y[0, :sample_len].cpu().numpy()         # [T, 36]

                # 保存特征文件（.f32格式用于后续音频合成）
                pred_feature_path = audio_dir / f'pred_features_epoch_{epoch}.f32'
                target_feature_path = audio_dir / f'target_features_epoch_{epoch}.f32'

                y_hat_sample.astype(np.float32).tofile(pred_feature_path)
                y_sample.astype(np.float32).tofile(target_feature_path)

                # 保存特征对比摘要
                feature_comparison = {
                    'epoch': epoch,
                    'sample_length_frames': int(sample_len),
                    'sample_length_seconds': float(sample_len / 50.0),  # 50fps
                    'pred_feature_stats': {
                        'mean': float(y_hat_sample.mean()),
                        'std': float(y_hat_sample.std()),
                        'min': float(y_hat_sample.min()),
                        'max': float(y_hat_sample.max())
                    },
                    'target_feature_stats': {
                        'mean': float(y_sample.mean()),
                        'std': float(y_sample.std()),
                        'min': float(y_sample.min()),
                        'max': float(y_sample.max())
                    },
                    'pred_feature_file': str(pred_feature_path.name),
                    'target_feature_file': str(target_feature_path.name),
                    'note': 'Use dump_data or fargan_demo to synthesize audio from these features'
                }

                feature_comp_path = audio_dir / f'feature_comparison_epoch_{epoch}.json'
                with open(feature_comp_path, 'w') as f:
                    json.dump(feature_comparison, f, indent=2)

                val_metrics['audio_samples_saved'] = True
                val_metrics['audio_sample_frames'] = int(sample_len)
                print(f"  Audio features saved: {pred_feature_path.name}, {target_feature_path.name}")

            except Exception as e:
                print(f"  Warning: Failed to save audio samples: {e}")
                val_metrics['audio_samples_saved'] = False
                val_metrics['audio_error'] = str(e)

        # 保存验证摘要
        val_summary_path = out_dir / f'validation_summary_epoch_{epoch}.json'
        with open(val_summary_path, 'w') as f:
            json.dump(val_metrics, f, indent=2)

        print(f"Validation Summary (Epoch {epoch}):")
        print(f"  Loss: {val_metrics['val_loss']:.6f}, MAE: {val_metrics['val_mae']:.6f}")
        print(f"  Rel Error: {val_metrics['val_rel_error']:.4f}, Var Ratio: {val_metrics['val_var_ratio']:.4f}")
        print(f"  Batches: {val_metrics['val_batches_processed']}, Samples: {val_metrics['val_samples_total']}")
        print(f"  Stage: {stage}")

    return val_metrics


def train_one_epoch(encoder: nn.Module,
                    decoder: nn.Module,
                    loader,
                    device: torch.device,
                    optimizer: optim.Optimizer,
                    autocast_mode: str,
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    epoch: int,
                    stage_cfg: StageConfig,
                    current_step: int,
                    feature_spec_type: str,
                    lambda_rate: float,
                    focus_start_frames: int = 0) -> Dict[str, float]:
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_items = 0
    step = current_step

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch in progress:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        # Stage 1 不引入JSCC，但保持CSI接口兼容，提供零填充的10维向量
        batch_bs = x.size(0)
        csi_dict = {
            'snr_db': torch.zeros(batch_bs, device=device, dtype=x.dtype),
            'ber': torch.zeros(batch_bs, device=device, dtype=x.dtype),
            'fading_onehot': torch.zeros(batch_bs, 8, device=device, dtype=x.dtype)
        }
        csi_dict['fading_onehot'][:, 0] = 1.0  # 标记“clean”信道

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(autocast_mode == "bf16")):
            z, _ = encoder(x, csi_dict)
            y_hat_full = decoder(z, csi_dict)

            # Teacher-forcing-like warm start: ignore first N frames for loss/metrics
            fs = max(0, int(focus_start_frames))
            if fs > 0 and y_hat_full.size(1) > fs:
                y_hat = y_hat_full[:, fs:, :]
                y_ref = y[:, fs:, :]
            else:
                y_hat = y_hat_full
                y_ref = y

            recon, loss_details, stage = compute_layered_loss(
                y_hat, y_ref, current_step=step, feature_spec_type=feature_spec_type
            )

            # 形状保持辅助损失（在关注区域内计算）
            # 1. 时间差分损失：mean(|Δŷ - Δy|)，权重≈0.05
            if y_hat.size(1) > 1:
                delta_y_hat = y_hat[:, 1:] - y_hat[:, :-1]  # [B, T-1, D]
                delta_y = y_ref[:, 1:] - y_ref[:, :-1]      # [B, T-1, D]
                temporal_diff_loss = torch.abs(delta_y_hat - delta_y).mean()
            else:
                temporal_diff_loss = torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)

            # 2. 方差匹配损失：mean(|Var_t - Var_ŷ|)，权重≈0.01
            var_y_hat = y_hat.var(dim=1, keepdim=True)  # [B, 1, D]
            var_y = y_ref.var(dim=1, keepdim=True)      # [B, 1, D]
            variance_match_loss = torch.abs(var_y_hat - var_y).mean()

            # 禁用rate_loss，专注于特征重建质量
            rloss = torch.zeros_like(recon)  # 禁用rate损失
            var_t = y_hat.float().var(dim=1).mean()
            anti_static = (1.0 / (var_t + 1e-3)).clamp(max=1e3)

            # 组合损失：主要重建损失 + 形状保持辅助损失 + anti_static
            loss = (recon +
                    0.05 * temporal_diff_loss +      # 时间差分损失权重
                    0.01 * variance_match_loss +     # 方差匹配损失权重
                    stage_cfg.anti_static_weight * anti_static)

        loss_value = float(loss.detach().item())

        # 计算MAE和相对误差指标
        with torch.no_grad():
            mae = torch.abs(y_hat - y_ref).mean()
            mse = torch.square(y_hat - y_ref).mean()
            rmse = torch.sqrt(mse)

            # 相对误差（避免除零）
            target_norm = torch.norm(y_ref, dim=-1, keepdim=True) + 1e-8
            relative_error = torch.norm(y_hat - y_ref, dim=-1, keepdim=True) / target_norm
            rel_err_mean = relative_error.mean()

            # 特征方差分析
            pred_var = y_hat.var(dim=1).mean()
            target_var = y_ref.var(dim=1).mean()

        # 创建详细指标字典
        metrics = {
            'loss': loss_value,
            'recon_loss': float(recon.item()),
            'temporal_diff_loss': float(temporal_diff_loss.item()),
            'variance_match_loss': float(variance_match_loss.item()),
            'anti_static': float(anti_static.item()),
            'mae': float(mae.item()),
            'rmse': float(rmse.item()),
            'rel_error': float(rel_err_mean.item()),
            'pred_var': float(pred_var.item()),
            'target_var': float(target_var.item()),
            'stage': stage,
            **{f'loss_{k}': v for k, v in loss_details.items()}
        }

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

        bs = x.size(0)
        total_loss += loss_value * bs
        total_items += bs
        step += 1

        # 更新进度条显示（每50步或最后一步）
        if step % 50 == 0 or step == len(loader):
            postfix = {
                'loss': f"{loss_value:.4f}",
                'recon': f"{metrics['recon_loss']:.4f}",
                'temp_diff': f"{metrics['temporal_diff_loss']:.4f}",
                'var_match': f"{metrics['variance_match_loss']:.4f}",
                'mae': f"{metrics['mae']:.4f}",
                'stage': stage
            }
            progress.set_postfix(postfix)

    avg = total_loss / max(1, total_items)
    return {
        'loss': avg,
        'steps': step - current_step,
        'last_metrics': metrics  # 返回最后一批次的详细指标
    }


def main() -> int:
    p = argparse.ArgumentParser(description='Stage 1: AETHER feature reconstruction training')
    p.add_argument('--features', type=str, required=True, help='路径到特征文件 (.f32)')
    p.add_argument('--pcm', type=str, required=True, help='路径到音频文件 (.pcm16)')
    p.add_argument('--output-dir', type=str, default='checkpoints_stage1', help='输出目录')
    p.add_argument('--device', type=str, default='auto', help='auto|cpu|cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seq-len', type=int, default=200, help='序列长度（第1阶段推荐200以保证稳定性，后续可提升至400/800）')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--feature-dims', type=int, default=36)
    p.add_argument('--feature-spec-type', type=str, default='fargan', choices=['fargan'])
    p.add_argument('--amp', type=str, default='bf16', choices=['none', 'bf16'], help='AMP 模式 (none 或 bf16)')
    p.add_argument('--preheat-frames', type=int, default=0, help='忽略前N帧计算损失/指标（类似教师强制热启动）')
    p.add_argument('--save-audio-samples', action='store_true', help='在验证时保存音频特征样本用于听力测试')
    p.add_argument('--validation-split', type=float, default=0.15, help='验证集比例 (0.0-1.0), 0.0表示不使用验证集')
    p.add_argument('--seed', type=int, default=42, help='随机种子用于可重现结果')
    p.add_argument('--early-stop', action='store_true', help='启用早停（基于StageConfig阈值）')
    p.add_argument('--early-stop-patience', type=int, default=5, help='早停耐心（epoch数）')
    p.add_argument('--cosine-lr', action='store_true', help='使用余弦学习率调度')
    p.add_argument('--warmup-epochs', type=int, default=2, help='学习率预热轮数')
    args = p.parse_args()
    stage_cfg = get_stage_config("stage1")

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device if args.device != 'auto' else 'cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子以保证可重现结果
    import random
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # 创建数据加载器（支持训练/验证分割）
    data_dir = str(Path(args.features).parent.parent) if 'lmr_export' in Path(args.features).parts else str(Path(args.features).parent)
    loader_kwargs = {
        'data_dir': data_dir,
        'sequence_length': args.seq_len,
        'batch_size': args.batch_size,
        'max_samples': None,
        'num_workers': max(1, min(4, args.num_workers)),
        'energy_selection': True,
        'test_mode': False,
        'feature_spec_type': args.feature_spec_type,
        'features_file': args.features,
        'audio_file': args.pcm
    }

    if args.validation_split > 0.0:
        # 使用训练/验证分割
        train_loader, val_loader, train_dataset, val_dataset = create_train_val_loaders(
            validation_split=args.validation_split,
            **loader_kwargs
        )
        print(f"Using train/val split: {args.validation_split:.1%} validation")
    else:
        # 不使用验证集，仅训练
        train_loader, train_dataset = create_aether_data_loader(**loader_kwargs)
        val_loader = None
        val_dataset = None
        print("Training without validation split")

    # Models - 第1阶段禁用MoE和FiLM，避免潜在特征z被过度正则化
    encoder = AETHEREncoder(
        d_in=args.feature_dims,
        d_csi=10,
        feature_spec_type=args.feature_spec_type,
        use_moe=stage_cfg.use_moe,  # False for stage1
        use_film=stage_cfg.use_film  # False for stage1
    ).to(device)
    encoder.quantize_latent = False
    decoder = AETHERDecoder(
        d_out=args.feature_dims,
        d_csi=10,
        enable_synth=False,
        feature_spec_type=args.feature_spec_type,
        use_film=stage_cfg.use_film  # False for stage1
    ).to(device)
    encoder.set_stage("A")

    optimizer = optim.AdamW(
        [{'params': encoder.parameters(), 'lr': stage_cfg.learning_rate},
         {'params': decoder.parameters(), 'lr': stage_cfg.learning_rate}],
        weight_decay=1e-6
    )
    scaler: Optional[torch.cuda.amp.GradScaler] = None

    # 学习率调度器设置
    scheduler = None
    if args.cosine_lr:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=stage_cfg.learning_rate * 0.1)
        print(f"Using cosine LR schedule: {stage_cfg.learning_rate} -> {stage_cfg.learning_rate * 0.1}")

    # 早停设置
    early_stop_counter = 0
    early_stop_best = float('inf')
    early_stop_threshold = stage_cfg.early_stop_loss if hasattr(stage_cfg, 'early_stop_loss') else 0.01
    print(f"Early stopping: enabled={args.early_stop}, patience={args.early_stop_patience}, threshold={early_stop_threshold}")

    best = float('inf')
    global_step = 0
    if device.type == 'cuda':
        requested_amp = args.amp
        if requested_amp == 'bf16' and not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 not supported on this GPU, disabling AMP.")
            amp_mode = 'none'
        else:
            amp_mode = requested_amp
    else:
        amp_mode = 'none'

    # 验证设置
    validation_freq = 1 if val_loader is not None else 0  # 每个epoch验证一次（如果有验证集）
    if val_loader is not None:
        print(f"Validation enabled: every epoch")
    else:
        print("Validation disabled: no validation split")

    # 创建训练日志文件
    train_log_path = out_dir / 'training_log.jsonl'
    train_log = open(train_log_path, 'w')

    try:
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(
                encoder,
                decoder,
                train_loader,
                device,
                optimizer,
                amp_mode,
                scaler,
                epoch,
                stage_cfg,
                current_step=global_step,
                feature_spec_type=args.feature_spec_type,
                lambda_rate=stage_cfg.lambda_rate,
                focus_start_frames=args.preheat_frames,
            )
            global_step += metrics.get('steps', 0)

            # 记录训练指标到日志
            train_record = {
                'epoch': epoch,
                'global_step': global_step,
                'train_loss': metrics['loss'],
                'train_steps': metrics.get('steps', 0)
            }

            # 添加详细的训练指标（如果可用）
            if 'last_metrics' in metrics:
                last = metrics['last_metrics']
                train_record.update({
                    'train_recon_loss': last.get('recon_loss', 0),
                    'train_mae': last.get('mae', 0),
                    'train_rmse': last.get('rmse', 0),
                    'train_rel_error': last.get('rel_error', 0),
                    'train_pred_var': last.get('pred_var', 0),
                    'train_target_var': last.get('target_var', 0),
                    'train_stage': last.get('stage', 'unknown')
                })
                # 添加分层损失详情
                for k, v in last.items():
                    if k.startswith('loss_'):
                        train_record[f'train_{k}'] = v

            print(f"Epoch {epoch}/{args.epochs} loss={metrics['loss']:.6f}")
            if 'last_metrics' in metrics:
                last = metrics['last_metrics']
                print(f"  MAE: {last.get('mae', 0):.4f}, Rel Error: {last.get('rel_error', 0):.4f}, Stage: {last.get('stage', 'unknown')}")

            # 学习率调度
            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if abs(old_lr - new_lr) > 1e-8:
                    print(f"  LR: {old_lr:.6f} -> {new_lr:.6f}")

            # 运行验证钩子（如果启用）
            if val_loader is not None and validation_freq > 0 and epoch % validation_freq == 0:
                print(f"\n--- Running validation hook (Epoch {epoch}) ---")
                try:
                    val_metrics = validate_epoch(
                        encoder, decoder, device, amp_mode,
                        args.feature_spec_type, val_loader, epoch, out_dir,
                        current_step=global_step,
                        focus_start_frames=args.preheat_frames,
                        save_audio=args.save_audio_samples,
                        max_val_batches=100
                    )
                    # 将验证指标添加到训练记录
                    train_record.update(val_metrics)
                    print(f"Validation completed. Loss: {val_metrics.get('val_loss', 'N/A'):.6f}")
                except Exception as e:
                    print(f"Warning: Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    train_record['val_error'] = str(e)
                print("--- Validation hook completed ---\n")

            # 写入训练日志
            train_log.write(json.dumps(train_record) + '\n')
            train_log.flush()

            # 使用验证损失（如果可用）或训练损失来决定最佳模型
            current_loss = train_record.get('val_loss', metrics['loss'])
            if current_loss < best:
                best = current_loss
                ckpt = {
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'epoch': epoch,
                    'loss': best,
                    'global_step': global_step,
                    'train_record': train_record,  # 保存训练记录以便后续分析
                    'validation_split': args.validation_split,
                    'seed': args.seed
                }
                torch.save(ckpt, out_dir / 'stage1_best.pth')
                loss_type = "validation" if 'val_loss' in train_record else "training"
                print(f"Saved best Stage1 checkpoint ({loss_type} loss: {best:.6f}): {out_dir / 'stage1_best.pth'}")
                early_stop_counter = 0  # 重置早停计数器

            # 早停检查
            if args.early_stop:
                if current_loss < early_stop_best:
                    early_stop_best = current_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # 检查是否达到早停条件
                if current_loss <= early_stop_threshold:
                    print(f"\nEarly stopping: Loss {current_loss:.6f} <= threshold {early_stop_threshold}")
                    break
                elif early_stop_counter >= args.early_stop_patience:
                    print(f"\nEarly stopping: No improvement for {args.early_stop_patience} epochs (best: {early_stop_best:.6f})")
                    break

                if early_stop_counter > 0:
                    print(f"  Early stop patience: {early_stop_counter}/{args.early_stop_patience}")

    finally:
        train_log.close()
        print(f"\nTraining completed. Final best loss: {best:.6f}")
        if args.early_stop:
            print(f"Early stopping stats: counter={early_stop_counter}, best={early_stop_best:.6f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
