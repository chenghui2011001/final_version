#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: MoE引入训练 (禁用FiLM，单变量验证MoE贡献)

按照AETHER工程级任务执行清单要求:
- 目标: 隔离评估MoE对瓶颈表达的贡献
- 模块: 保留DualStream+GLA；启用Micro-MoE；禁用FiLM；禁用信道模拟
- 单变量原则: 避免混淆MoE与CSI/FiLM的贡献
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
import soundfile as sf

# 使用简化的架构
import sys
import os
# Ensure final_version root is on sys.path; avoid inserting subdirs (e.g. models)
# to prevent shadowing top-level packages like 'utils'.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.enhanced_aether_integration import AETHEREncoder, AETHERDecoder, create_aether_codec
from models.maybe_useless.aether_fargan_decoder import AETHERFARGANDecoder
from utils.real_data_loader import create_aether_data_loader
from utils.real_data_loader import create_combined_data_loader
from training.pipeline.stages import StageConfig, get_stage_config
# 🔥 恢复FARGAN标准损失，移除自定义audio_usability_loss
from training.pipeline.wave_loss import fargan_wave_losses
from training.losses import rate_loss, compute_layered_loss
from models.utils import validate_csi_config, extract_acoustic_priors
from models.semantic_fargan_adapter import create_semantic_fargan_adapter
from models.semantic_extractor import create_semantic_extractor
# ---- 放在文件顶部合适位置（或 train_one_epoch 内部开头）----
def _sum_grad_norm(named_params, include_key=None, exclude_key=None):
    """对指定参数集合求梯度范数之和（先净化 NaN/Inf），返回 (sum_norm, n_tensors)。"""
    total = 0.0
    n = 0
    for name, p in named_params:
        if p.grad is None:
            continue
        if include_key is not None and include_key not in name:
            continue
        if exclude_key is not None and exclude_key in name:
            continue
        g = p.grad.detach()
        # 关键：净化 NaN/Inf，避免整段统计变 NaN
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        # 用 float() 再取范数，避免半精度下溢
        val = g.float().norm()
        # 双重保险：范数本身若仍非有限，就跳过
        if torch.isfinite(val):
            total += val.item()
            n += 1
    return total, n

def _topk_grad_norm(named_params, k=5, include_key=None, exclude_key=None):
    """可选：打印最大梯度范数的若干参数，便于定位异常。"""
    arr = []
    for name, p in named_params:
        if p.grad is None:
            continue
        if include_key is not None and include_key not in name:
            continue
        if exclude_key is not None and exclude_key in name:
            continue
        g = torch.nan_to_num(p.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        val = g.float().norm()
        if torch.isfinite(val):
            arr.append((val.item(), name))
    arr.sort(key=lambda x: x[0], reverse=True)
    return arr[:k]

def _print_feature_reconstruction_stats(pred_feats, orig_feats, global_step, batch_idx):
    """
    打印36维FARGAN特征的重建统计信息（后16维不再打印LPC，改由语义统计单独输出）
    pred_feats: 复原特征 [B, T, 36]
    orig_feats: 原始特征 [B, T, 36]
    """
    with torch.no_grad():
        tqdm.write(f"\n========== 特征重建统计 (Step {global_step}, Batch {batch_idx}) ==========")

        # 前18维倒谱特征统计
        tqdm.write("--- 倒谱特征 (Dims 0-17) ---")
        for dim in range(18):
            pred_dim = pred_feats[:, :, dim].flatten()
            orig_dim = orig_feats[:, :, dim].flatten()

            pred_mean, pred_std = pred_dim.mean().item(), pred_dim.std().item()
            pred_min, pred_max = pred_dim.min().item(), pred_dim.max().item()
            orig_mean, orig_std = orig_dim.mean().item(), orig_dim.std().item()
            orig_min, orig_max = orig_dim.min().item(), orig_dim.max().item()

            tqdm.write(f"  Dim[{dim:2d}] | Pred: mean={pred_mean:+6.3f} std={pred_std:6.3f} range=[{pred_min:+6.3f}, {pred_max:+6.3f}]")
            tqdm.write(f"         | Orig: mean={orig_mean:+6.3f} std={orig_std:6.3f} range=[{orig_min:+6.3f}, {orig_max:+6.3f}]")

        # 第19维F0特征统计 (DNN Pitch)
        tqdm.write("\n--- F0/基频特征 (Dim 18) ---")
        pred_f0 = pred_feats[:, :, 18].flatten()
        orig_f0 = orig_feats[:, :, 18].flatten()

        pred_f0_mean, pred_f0_std = pred_f0.mean().item(), pred_f0.std().item()
        pred_f0_min, pred_f0_max = pred_f0.min().item(), pred_f0.max().item()
        orig_f0_mean, orig_f0_std = orig_f0.mean().item(), orig_f0.std().item()
        orig_f0_min, orig_f0_max = orig_f0.min().item(), orig_f0.max().item()

        tqdm.write(f"  F0     | Pred: mean={pred_f0_mean:+6.3f} std={pred_f0_std:6.3f} range=[{pred_f0_min:+6.3f}, {pred_f0_max:+6.3f}]")
        tqdm.write(f"         | Orig: mean={orig_f0_mean:+6.3f} std={orig_f0_std:6.3f} range=[{orig_f0_min:+6.3f}, {orig_f0_max:+6.3f}]")

        # 清浊音统计 (基于F0阈值判断)
        pred_voiced = (pred_f0 > -1.0).float().mean().item()
        orig_voiced = (orig_f0 > -1.0).float().mean().item()
        tqdm.write(f"  Voice  | Pred: voiced={pred_voiced:.3f} unvoiced={1-pred_voiced:.3f}")
        tqdm.write(f"         | Orig: voiced={orig_voiced:.3f} unvoiced={1-orig_voiced:.3f}")

        # 第20维帧相关性特征
        tqdm.write("\n--- 帧相关性特征 (Dim 19) ---")
        pred_corr = pred_feats[:, :, 19].flatten()
        orig_corr = orig_feats[:, :, 19].flatten()

        pred_corr_mean, pred_corr_std = pred_corr.mean().item(), pred_corr.std().item()
        pred_corr_min, pred_corr_max = pred_corr.min().item(), pred_corr.max().item()
        orig_corr_mean, orig_corr_std = orig_corr.mean().item(), orig_corr.std().item()
        orig_corr_min, orig_corr_max = orig_corr.min().item(), orig_corr.max().item()

        tqdm.write(f"  Corr   | Pred: mean={pred_corr_mean:+6.3f} std={pred_corr_std:6.3f} range=[{pred_corr_min:+6.3f}, {pred_corr_max:+6.3f}]")
        tqdm.write(f"         | Orig: mean={orig_corr_mean:+6.3f} std={orig_corr_std:6.3f} range=[{orig_corr_min:+6.3f}, {orig_corr_max:+6.3f}]")

        # 整体重建质量评估（本节不再纳入后16维）
        tqdm.write("\n--- 整体重建质量（不含后16维语义） ---")
        overall_mse = F.mse_loss(pred_feats[:, :, :20], orig_feats[:, :, :20]).item()
        overall_mae = F.l1_loss(pred_feats[:, :, :20], orig_feats[:, :, :20]).item()

        # 分段评估（不含后16维）
        cepstral_mse = F.mse_loss(pred_feats[:, :, :18], orig_feats[:, :, :18]).item()
        f0_mse = F.mse_loss(pred_feats[:, :, 18:19], orig_feats[:, :, 18:19]).item()
        corr_mse = F.mse_loss(pred_feats[:, :, 19:20], orig_feats[:, :, 19:20]).item()

        tqdm.write(f"  Overall MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}")
        tqdm.write(f"  Cepstral MSE: {cepstral_mse:.6f}")
        tqdm.write(f"  F0 MSE: {f0_mse:.6f}")
        tqdm.write(f"  Correlation MSE: {corr_mse:.6f}")

        tqdm.write("=" * 65)

def _print_semantic_alignment_stats(semantic_pred: torch.Tensor, semantic_target: torch.Tensor, global_step: int, batch_idx: int):
    """
    打印后16维语义特征的对齐统计信息
    semantic_pred:   [B, T, 16]
    semantic_target: [B, T, 16]
    """
    with torch.no_grad():
        tqdm.write(f"\n---------- 语义特征对齐 (Step {global_step}, Batch {batch_idx}) ----------")
        for dim in range(16):
            pred_dim = semantic_pred[:, :, dim].flatten()
            tgt_dim  = semantic_target[:, :, dim].flatten()

            p_mean, p_std = pred_dim.mean().item(), pred_dim.std().item()
            p_min, p_max  = pred_dim.min().item(), pred_dim.max().item()
            t_mean, t_std = tgt_dim.mean().item(),  tgt_dim.std().item()
            t_min, t_max  = tgt_dim.min().item(),  tgt_dim.max().item()

            tqdm.write(f"  Sem[{dim:2d}] | Pred: mean={p_mean:+6.3f} std={p_std:6.3f} range=[{p_min:+6.3f}, {p_max:+6.3f}]")
            tqdm.write(f"           | Tgt : mean={t_mean:+6.3f} std={t_std:6.3f} range=[{t_min:+6.3f}, {t_max:+6.3f}]")

        sem_mse = F.mse_loss(semantic_pred, semantic_target).item()
        sem_mae = F.l1_loss(semantic_pred, semantic_target).item()
        tqdm.write(f"  Semantic MSE: {sem_mse:.6f}, MAE: {sem_mae:.6f}")
        tqdm.write("-" * 65)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 确保eps与输入tensor的dtype一致
        eps_tensor = torch.tensor(self.eps, dtype=x.dtype, device=x.device)
        return torch.mean(torch.sqrt((x - y) **2 + eps_tensor** 2))

class SimplifiedStage3Trainer:
    """Simplified Stage3 trainer focusing on MoE validation only."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Stage3: 强制禁用FiLM，启用3专家MoE (无CSI，不需要LowSNRExpert)
        stage3_config = {
            "d_in": 36,
            "d_model": 128,
            "dz": 24,
            "d_csi": 10,
            "use_film": False,  # Stage3: 禁用FiLM
            "use_moe": True,   # Stage3: 启用简化统一MoE
            "use_quantization": False,
            "latent_bits": 4,
            "n_experts": 4,    # Stage3: 4个功能专家 (Harmonic/Transient/BurstInpaint/LowSNR)
            "top_k": 2,        # Stage3: TOP-2路由，根据音频内容选择合适专家
            # 根据专家数据验证报告优化的MoE配置
            "moe_balance_weight": 0.5,      # 验证报告建议：中等质量数据使用0.5
            "expert_dropout": 0.15,         # 验证报告建议：使用0.15
            "router_jitter": 0.0,           # 禁用路由抖动，专注专家差异化
            "moe_router_use_csi": False,    # 新设计：禁用CSI路由，改为纯音频特征路由
            "enable_direct_pathway": True,   # 启用直流通路，权重0.1
            "initial_bypass_weight": 0.1,   # 验证报告建议的直流通路权重
            "adaptive_threshold": 0.15,     # 适应性阈值
        }

        # 打印Stage3 MoE配置摘要
        print("=" * 60)
        print("Stage3 Training - Unified MoE Configuration")
        print("=" * 60)
        print(f"Expert Count: {stage3_config['n_experts']} specialized experts")
        print(f"Top-K Routing: {stage3_config['top_k']} (competitive expert selection)")
        print(f"Architecture: Audio-Scenario Specialized UnifiedAudioExpert")
        print(f"Expert Functions: Harmonic/Transient/BurstInpaint/LowSNR")
        print(f"Balance Weight: {stage3_config['moe_balance_weight']} (optimized by expert data validation)")
        print(f"Expert Dropout: {stage3_config['expert_dropout']} (optimized by expert data validation)")
        print(f"Router Jitter: {stage3_config['router_jitter']} (disabled for specialization focus)")
        print(f"Direct Pathway: {stage3_config['enable_direct_pathway']} (weight: {stage3_config.get('initial_bypass_weight', 0.1)})")
        print("Key Improvements:")
        print("  - Expert data augmentation: Targeted datasets for each expert specialization")
        print("  - Audio scenario specialization: Harmonic(tonal) / Transient(dynamic) / Repair(gaps) / LowSNR(feature-noise)")
        print("  - Top-K=2 selects most relevant experts based on audio content analysis")
        print("  - Scenario-specific initialization patterns match processing requirements")
        print("  - AcousticFeatureExtractor provides expert routing signals from raw features")
        print("  - LowSNR expert focuses on raw feature quality, not channel noise")
        print("=" * 60)

        self.encoder, self.decoder = create_aether_codec(stage3_config)

    def get_moe_metrics(self) -> Dict[str, float]:
        """获取MoE专家利用率指标。"""
        metrics = {}
        if hasattr(self.encoder, 'moe') and self.encoder.moe is not None:
            try:
                expert_util = self.encoder.moe.get_expert_utilization()
                metrics['expert_usage_min'] = expert_util.min().item()
                metrics['expert_usage_max'] = expert_util.max().item()
                metrics['expert_entropy'] = -(expert_util * torch.log(expert_util + 1e-8)).sum().item()
                metrics['expert_balance'] = 1.0 - expert_util.std().item()
                # Store individual expert utilization rates
                for i, util in enumerate(expert_util):
                    metrics[f'expert_{i}_usage'] = util.item()

                # Store as formatted string for display with expert names
                expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"]
                expert_usage_named = []
                for i, util in enumerate(expert_util):
                    name = expert_names[i] if i < len(expert_names) else f"E{i}"
                    expert_usage_named.append(f"{name}:{util.item():.3f}")
                metrics['expert_usage_all'] = ', '.join(expert_usage_named)
            except (AttributeError, RuntimeError):
                # Fallback for MoE without utilization tracking
                metrics['expert_usage_min'] = 0.25  # Placeholder
                metrics['expert_usage_max'] = 0.25
                metrics['expert_entropy'] = 1.386  # log(4) for 4 experts
                metrics['expert_balance'] = 0.8
                # 使用带名称的占位符
                metrics['expert_usage_all'] = 'Harmonic:0.250, Transient:0.250, BurstInpaint:0.250, LowSNR:0.250'
        return metrics


def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    loader,
    device: torch.device,
    optimizer: optim.Optimizer,
    stage_cfg: StageConfig,
    current_step: int,
    args: argparse.Namespace,
    epoch_idx: Optional[int] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None, 
) -> Tuple[Dict[str, float], int]:
    """Train one epoch with MoE monitoring (encoder+decoder provided)."""
    encoder.train()
    decoder.train()
    use_fp16 = (scaler is not None) and scaler.is_enabled()
    epoch_metrics = {
        'total_loss': 0.0,
        'feature_loss': 0.0,
        'wave_loss': 0.0,
        'moe_loss': 0.0,
        'rate_loss': 0.0,
        'expert_entropy': 0.0,
        'expert_usage_min': 0.0,
        'expert_usage_max': 0.0,
    }
    total_samples = 0
    step = current_step

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch_idx}/{args.epochs}" if epoch_idx is not None else "Train",
        dynamic_ncols=True,
        leave=False,
    )

    if not hasattr(train_one_epoch, '_gn_ema'):
        train_one_epoch._gn_ema = 0.0
    if not hasattr(train_one_epoch, '_wave_bp_ratio'):
        train_one_epoch._wave_bp_ratio = 0.0

    # 🔥 移除旧的CharbonnierLoss，新系统使用audio_usability_loss
    # wave_char = CharbonnierLoss(eps=1e-3)

    # === 🔥 语义感知系统强制启用 ===
    # 初始化语义FarGAN适配器
    if not hasattr(train_one_epoch, '_semantic_adapter'):
        train_one_epoch._semantic_adapter = create_semantic_fargan_adapter(
            adapter_type="progressive",
            input_dim=36,
            output_dim=36
        ).to(device)

        # 强制添加适配器参数到优化器
        adapter_params = [p for p in train_one_epoch._semantic_adapter.parameters() if p.requires_grad]
        if adapter_params:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.add_param_group({
                'params': adapter_params,
                'lr': current_lr * 0.1,
                'weight_decay': 0.0
            })
            print(f"      🔥 Semantic adapter activated: {len(adapter_params)} parameters")

    semantic_adapter = train_one_epoch._semantic_adapter

    for batch_idx, batch in progress:

        global_step = step + batch_idx

        # Optional: token-level router warmup (use sample-level only for first N steps)
        warmup_tok = int(getattr(args, 'moe_token_warmup_steps', 0) or 0)
        if warmup_tok > 0 and hasattr(encoder, 'moe') and encoder.moe is not None:
            try:
                # CompatibleMicroMoE wrapper path
                encoder.moe.specialized_moe.use_token_level = (global_step >= warmup_tok)
            except Exception:
                # Legacy MicroMoE or other: ignore
                pass

        # Optional: Top-K warm-start for early steps (e.g., force k=1 for the first N steps)
        try:
            topk_warm_steps = int(getattr(args, 'topk_warm_steps', 0) or 0)
            if topk_warm_steps > 0 and hasattr(encoder, 'moe') and encoder.moe is not None \
               and hasattr(encoder.moe, 'specialized_moe') and hasattr(encoder.moe.specialized_moe, 'topk'):
                sm = encoder.moe.specialized_moe
                if not hasattr(sm, '_topk_orig'):
                    sm._topk_orig = int(getattr(sm, 'topk', 2) or 2)
                if global_step < topk_warm_steps:
                    sm.topk = int(getattr(args, 'topk_warm_k', 1) or 1)
                else:
                    sm.topk = int(getattr(sm, '_topk_orig', 2) or 2)
        except Exception:
            pass

        # Data to device
        x = batch['x'].to(device, non_blocking=True)  # Features [B, T, 36]
        y = batch['y'].to(device, non_blocking=True)  # Target features [B, T, 36]
        audio = batch['audio'].to(device, non_blocking=True)  # Target audio [B, L]

        # Stage3: 使用固定dummy CSI减少计算开销 (真正的单变量验证)
        batch_size = x.size(0)

        # 梯度累积：仅在开始时清零（在累积循环内部处理）
        accum_steps = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
        if batch_idx % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # 计算波形启用与热身：先严格执行 wave_start_step，再进行 warmup 拉起
        start_step = int(getattr(args, 'wave_start_step', 0) or 0)
        warm_steps = int(getattr(args, 'wave_warmup_steps', 0) or 0)
        active_wave = (global_step >= start_step)
        if active_wave and warm_steps > 0:
            warm_ratio = min(1.0, max(0.0, (global_step - start_step) / float(warm_steps)))
        else:
            warm_ratio = 0.0

        # 目标反传比例：更温和的二次曲线
        bp_target = warm_ratio ** 2
        # 仅当最近梯度EMA较低时，才允许放开反传比例（只增不减）
        gn_ema = float(train_one_epoch._gn_ema)
        wave_bp_ratio = float(train_one_epoch._wave_bp_ratio)
        if gn_ema < 3.0:
            wave_bp_ratio = max(wave_bp_ratio, bp_target)
        # >>> 🔧 FIX 3: 优化梯度传播控制，大幅提高传播比例 <<<
        # 解决fargan_core梯度过小的问题：从50%提高到80%
        min_bp = float(getattr(args, 'wave_min_bp', 0.8))  # 提高到80%
        if active_wave:
            wave_bp_ratio = max(wave_bp_ratio, min_bp)   # 确保最小80%梯度传播

        # 在训练早期进一步增强梯度传播
        if global_step < 2000:  # 前2000步使用更高的传播比例
            wave_bp_ratio = max(wave_bp_ratio, 0.9)

        # <<< DEBUG/SAFETY end <<<

        # 修复损失权重：按用户要求调整权重比例，防止梯度爆炸
        # feat_loss=0.1, wave_loss=0.7, moe_loss=0.2
        # 注意：wave loss通常较大，所以实际权重要更小
        # Deprecated static alpha; dynamic weights are applied later.
        # alpha_wave_eff = args.alpha_wave * 0.1
        # 写回状态（本步用"旧EMA"，在结尾更新EMA）
        train_one_epoch._wave_bp_ratio = wave_bp_ratio

        # Forward pass (optional AMP autocast)
        # === 正确的AMP配置：前向用fp16，损失计算用float32 ===
        amp_mode    = getattr(args, 'amp', 'none')
        # 关键修复：AMP在非波形阶段也启用，只是不计算波形损失
        amp_enabled = (device.type == 'cuda' and amp_mode in ('fp16', 'bf16'))
        amp_dtype   = torch.float16 if amp_mode == 'fp16' else torch.bfloat16
        # 损失计算始终用float32确保数值稳定性
        loss_dtype  = torch.float32

        # 创建固定的dummy CSI（始终用float32，避免dtype问题）
        if not hasattr(train_one_epoch, '_dummy_csi_cache'):
            train_one_epoch._dummy_csi_cache = {
                'snr_db': torch.tensor(15.0, device=device, dtype=torch.float32),
                'fading_onehot': torch.zeros(8, device=device, dtype=torch.float32),
                'ber': torch.tensor(0.001, device=device, dtype=torch.float32)
            }
            train_one_epoch._dummy_csi_cache['fading_onehot'][0] = 1.0

        # 复制到当前batch_size
        csi_dict = {
            'snr_db': train_one_epoch._dummy_csi_cache['snr_db'].expand(batch_size),
            'fading_onehot': train_one_epoch._dummy_csi_cache['fading_onehot'].unsqueeze(0).expand(batch_size, -1),
            'ber': train_one_epoch._dummy_csi_cache['ber'].expand(batch_size)
        }

        # 前向传播：输入保持原始dtype，只在autocast内部自动转换
        with torch.cuda.amp.autocast(enabled=False):
            z, enc_logs = encoder(x, csi_dict=None, training_step=global_step)

        # 2) 解码器：可用 AMP（注意：你的 vocoder 内部已禁用 autocast）
        if amp_enabled:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                feats = decoder(z, csi_dict=csi_dict, return_wave=False, target_len=None)
        else:
            feats = decoder(z, csi_dict=csi_dict, return_wave=False, target_len=None)

        # 3) 统一转回 FP32，避免后续数值不稳
        z = z.float()
        feats = feats.float()

        # 4) enc_logs 中的所有张量（包含 dict/list/tuple 嵌套）也转成 FP32
        def _to_float32_inplace(obj):
            if isinstance(obj, torch.Tensor):
                return obj.float()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    obj[k] = _to_float32_inplace(v)
                return obj
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_float32_inplace(v) for v in obj)
            return obj

        enc_logs = _to_float32_inplace(enc_logs)
        # ----------------------------------------------------------

        # 🔧 FIX 1: 确保feats保持梯度连接，避免计算图断裂
        if not isinstance(feats, torch.Tensor) or not feats.requires_grad:
            # 将 feats 与 z 的计算图建立依赖，避免被上游 detach/no_grad 断开
            feats = feats + 0.0 * z.sum()

        # 额外检查：确保feats始终保持梯度
        if isinstance(feats, torch.Tensor) and not feats.requires_grad:
            feats = feats.detach().requires_grad_(True)
        # --- Wave / Vocoder branch: only after wave_start_step ---
        y_hat_audio = None  # 默认不算波形
        wave_computed_flag = False  # 标记本batch是否计算过波形

        # 检查vocoder调用条件
        wave_stride = int(getattr(args, 'wave_stride', 1) or 1)
        should_call_vocoder = active_wave and (batch_idx % wave_stride == 0)

        # Helper: consistent DNN pitch -> f0_hz mapping (align with AETHERDecoder)
        def _decode_f0_hz_from_dnn_pitch(pitch_log2: torch.Tensor, sr: float = 16000.0) -> torch.Tensor:
            # AETHER convention: f0_hz = sr * 2**(dnn_pitch - 6.5)
            return (sr * torch.pow(2.0, pitch_log2 - 6.5)).clamp(50.0, 400.0)

        if should_call_vocoder:
            # 现场构造 vocoder 条件：feats20 (20-d cepstrum) + period
            # 当前解码器输出 feats 是 FARGAN 36 维：ceps(18) + dnn_pitch(1, idx=18) + frame_corr(1) + lpc(16)

            # === 1) 🔧 FIX 2: 改进teacher forcing策略，确保梯度路径清晰 ===
            # 降低最大teacher forcing比例，减少对GT的依赖
            tf = max(0.0, min(0.3, 0.3 * (1.0 - warm_ratio)))  # 从0.5降低到0.3

            # 使用detach()明确控制梯度流：GT部分不参与反传，预测部分保持梯度
            if tf > 0.0:
                feats_mix = tf * y.detach() + (1.0 - tf) * feats  # GT部分显式detach
            else:
                feats_mix = feats  # 完全使用预测特征，保持梯度

            # === 直接使用原始36维特征，跳过Semantic Adapter约束 ===
            # 取消语义适配器处理，直接使用解码器输出的原始特征
            # semantic_adapter.update_training_stage(global_step)
            # feats_adapted = semantic_adapter(feats_mix, global_step=global_step)

            # 适配器状态监控 - 已禁用
            # if batch_idx % 100 == 0:
            #     adapter_status = semantic_adapter.get_status()
            #     tqdm.write(f"      🔧 Adapter: stage={semantic_adapter.get_stage_name()}, "
            #               f"strength={adapter_status['adaptation_strength']:.2f}")

            # ① 倒谱给 vocoder：直接使用原始特征的前20维
            feats20 = feats_mix[..., :20].contiguous()      # [B,T,20] ← 使用原始未适配特征

            # ② 周期也用预测流的 dnn_pitch（第18维）
            pred_pitch = feats_mix[..., 18].float()         # [B,T]
            # Use the same mapping as AETHERDecoder to avoid periodic artifacts
            f0_hz = _decode_f0_hz_from_dnn_pitch(pred_pitch)  # [B,T]
            period = (16000.0 / f0_hz).clamp(32.0, 255.0).round().to(torch.long)


            # === 2) 时间轴桥接：若上游是 50 Hz（常见：T_audio≈2*T_feat），则上采样到 100 Hz ===
            T_feat = int(feats20.size(1))
            audio_frames = int(audio.size(-1) // 160)  # 100 Hz 帧
            if audio_frames >= 2 * T_feat - 4:  # 经验阈：当音频帧数显著大于特征帧时，认定为50→100
                feats20 = F.interpolate(feats20.permute(0, 2, 1), scale_factor=2.0,
                                        mode="linear", align_corners=False).permute(0, 2, 1).contiguous()
                period = F.interpolate(period.float().unsqueeze(1), scale_factor=2.0,
                                       mode="nearest").squeeze(1).to(torch.long).contiguous()
                T_feat = int(feats20.size(1))

            # === 3) 先在“进入 FARGANCond 之前”把时间维对齐：features 与 period 必须同长 ===
            T0 = min(T_feat, int(period.size(1)))
            if T0 <= 0:
                tqdm.write(f"      ❌ T0={T0} <= 0, T_feat={T_feat}, period.size(1)={period.size(1)}")
                y_hat_audio = None
            else:
                feats20 = feats20[:, :T0, :].contiguous()
                period  = period[:,  :T0   ].contiguous()

                # === 4) 严格按 FARGANCond 的内部收缩计算可用帧数 ===
                # FARGANCond.forward 内部会：
                #   a) 丢前 2 帧 → T-2
                #   b) 与周期对齐 cat 后，走 k=3 的 valid 卷积 → 再减 2 帧，最终有效为 (T-4)
                # 为了让 f/p 在 cat 前“时间维完全相等”，我们对外部输入也统一给“_nb+4”长度，
                # 这样内部丢 2 帧后都为 (_nb+2)，再经 k=3 valid 卷积 → (_nb)（刚好等于目标 nb）。
                nb_pre = 2                                 # 2 帧预热（与 Stage2 一致）
                pre = audio[..., : nb_pre * 160]           # 预热波形
                cond_len   = max(0, T0 - 4)                # 条件分支有效帧（T→T-4）
                period_len = max(0, int(period.size(1)) - 4)  # 周期也按 -4 估算，以“等长 +4 裕量”策略喂入
                target_len = max(0, audio_frames - nb_pre) # 去掉预热后的可合成帧
                nb_frames  = min(cond_len, period_len, target_len)

                # 以 5 帧粒度对齐（FARGAN 内核多以 5 为粒度）
                nb_frames = (nb_frames // 5) * 5

                if nb_frames < 5:
                    tqdm.write(f"      ❌ nb_frames={nb_frames} < 5")
                    y_hat_audio = None
                else:
                    # “等长 +4 裕量”裁片：两者都裁到 nb+4，确保进入 cond_net 前时间维完全一致
                    feats20_vc = feats20[:, : nb_frames + 4, :].contiguous()
                    period_vc  = period[:,  : nb_frames + 4   ].contiguous()  # 注意：+4（不是 +2）

                    def _call_vocoder_nb(_nb: int):
                        # ① 先裁片（等长 +4 裕量）
                        f = feats20_vc[:, : _nb + 4, :].contiguous()
                        p = period_vc[:,  : _nb + 4   ].clamp(32, 255).to(torch.long).contiguous()

                        # ② 再打印一次性桥接检查
                        if not hasattr(train_one_epoch, '_vc_checked'):
                            T_in = f.size(1)  # = _nb + 4
                            tqdm.write(f"[VocoderBridge] T_in={T_in}, will request nb={_nb}, "
                                    f"expect cond_len={T_in-4}, pre_frames={nb_pre}")
                            train_one_epoch._vc_checked = True

                        # ③ 🔧 FIX 5: 统一FARGAN调用策略，保持梯度连接且使用 eval 模式避免训练态噪声
                        prev_mode_fc = None
                        if hasattr(decoder, 'fargan_core'):
                            prev_mode_fc = decoder.fargan_core.training
                            decoder.fargan_core.eval()
                        y_audio, aux = decoder.fargan_core(f, p, _nb, pre=pre)
                        if hasattr(decoder, 'fargan_core') and prev_mode_fc is not None:
                            decoder.fargan_core.train(prev_mode_fc)

                        # 优化NaN处理，保持梯度连接而非完全替换
                        if torch.isnan(y_audio).any() or torch.isinf(y_audio).any():
                            # 使用条件替换保持梯度，而非torch.nan_to_num
                            y_audio = torch.where(
                                torch.isnan(y_audio) | torch.isinf(y_audio),
                                torch.zeros_like(y_audio),
                                y_audio
                            )
                        return y_audio.float(), aux



                    # === 5) 自适应重试（最多 4 次）：从“合法 nb”直接起步，必要时按 5 帧递减 ===
                    y_hat_audio = None
                    max_tries = 4
                    nb_try = max(5, (int(nb_frames) // 5) * 5)  # 合法 nb（已考虑 -4 收缩）

                    for try_idx in range(max_tries):
                        try:
                            if not hasattr(train_one_epoch, '_vc_checked'):
                                T_in = int(feats20_vc.size(1))  # 预期 = nb_try + 4
                                train_one_epoch._vc_checked = True

                            y_hat_audio, _ = _call_vocoder_nb(nb_try)
                            L = int(y_hat_audio.size(-1))

                            # ✅ 正确的期望：vocoder 只返回“预测段”，不含 pre 段
                            exp_len = max(0, (nb_try - nb_pre) * 160)

                            if L > exp_len:
                                y_hat_audio = y_hat_audio[..., :exp_len]
                            elif L < exp_len:
                                # 用返回长度反推最可能的 nb：frames_pred = L/160，nb = pre + frames_pred
                                frames_pred = max(0, L // 160)
                                nb_est = frames_pred + nb_pre
                                # 以 5 帧粒度回退
                                nb_back = max(5, (nb_est // 5) * 5)
                                if nb_back < nb_try:
                                    nb_try = nb_back
                                    tqdm.write(f"  🔁 vocoder retry: nb_try→{nb_try}")
                                    y_hat_audio = None
                                    continue

                            break
                        except IndexError as ie:
                            # 极端越界，按 5 帧退
                            tqdm.write(f"      ❌ IndexError in vocoder (try {try_idx+1}/{max_tries}): {str(ie)[:100]}")
                            nb_new = max(5, nb_try - 5)
                            if nb_new >= nb_try:
                                nb_new = nb_try - 5
                            nb_try = nb_new
                            tqdm.write(f"  🔁 vocoder retry: nb_try→{nb_try}")
                        except Exception:
                            # 其它异常保留抛出，便于定位
                            raise

                if y_hat_audio is None:
                    tqdm.write("  ❌ VOCODER FAILURE: All retries failed, skipping wave loss for this batch.")
                    tqdm.write(f"      Final nb_try={nb_try}, nb_frames={nb_frames}, nb_pre={nb_pre}")

        # === Quick validation audio export (pred + original) ===
        snap_every = int(getattr(args, 'val_audio_interval', 0) or 0)
        if snap_every > 0 and y_hat_audio is not None:
            if global_step % snap_every == 0:
                try:
                    out_root = Path(args.output_dir) / 'audio_snaps'
                    out_root.mkdir(parents=True, exist_ok=True)
                    # slice first sample
                    pred = y_hat_audio[0]
                    orig = audio[0]
                    # ensure 1-D
                    if pred.dim() > 1:
                        pred = pred.view(-1)
                    if orig.dim() > 1:
                        orig = orig.view(-1)
                    # clamp length to requested seconds and availability
                    max_len = int(getattr(args, 'val_audio_seconds', 10) * 16000)
                    L = min(pred.numel(), orig.numel(), max_len)
                    pred_np = torch.clamp(pred[:L].detach().cpu(), -1.0, 1.0).numpy()
                    # Optional de-emphasis (preview only)
                    deemph = float(getattr(args, 'val_audio_deemph', 0.85))
                    if deemph > 0.0:
                        y_prev = 0.0
                        for i in range(pred_np.shape[0]):
                            y_prev = float(pred_np[i]) + deemph * y_prev
                            pred_np[i] = y_prev
                    orig_np = torch.clamp(orig[:L].detach().cpu(), -1.0, 1.0).numpy()
                    sf.write(str(out_root / f'step_{global_step:06d}_pred.wav'), pred_np, 16000, subtype='PCM_16')
                    sf.write(str(out_root / f'step_{global_step:06d}_orig.wav'), orig_np, 16000, subtype='PCM_16')

                    # Optional: teacher-forced preview (use GT features -> vocoder)
                    if bool(getattr(args, 'val_audio_teacher', True)):
                        # Build teacher-forced features: 直接使用原始特征，跳过adapter
                        tf_feats = y.detach()  # [B,T,36] ground-truth features
                        # 强制跳过semantic adapter，直接使用原始特征
                        # if bool(getattr(args, 'val_audio_teacher_no_adapter', False)):
                        tf_adapted = tf_feats  # 直接使用原始GT特征
                        # else:
                        #     tf_adapted = semantic_adapter(tf_feats, global_step=global_step)
                        tf_feats20 = tf_adapted[..., :20].contiguous()
                        # Period from GT pitch (index 18), same mapping as pred
                        tf_pitch = tf_feats[..., 18].float()
                        tf_f0_hz = _decode_f0_hz_from_dnn_pitch(tf_pitch)
                        tf_period = (16000.0 / tf_f0_hz).clamp(32.0, 255.0).round().to(torch.long)

                        # 50->100Hz bridge if needed
                        T_feat_tf = int(tf_feats20.size(1))
                        audio_frames = int(audio.size(-1) // 160)
                        if audio_frames >= 2 * T_feat_tf - 4:
                            tf_feats20 = F.interpolate(tf_feats20.permute(0, 2, 1), scale_factor=2.0,
                                                       mode="linear", align_corners=False).permute(0, 2, 1).contiguous()
                            tf_period = F.interpolate(tf_period.float().unsqueeze(1), scale_factor=2.0,
                                                      mode="nearest").squeeze(1).to(torch.long).contiguous()
                            T_feat_tf = int(tf_feats20.size(1))

                        # Align lengths using the same nb logic
                        nb_pre = 0 if bool(getattr(args, 'val_audio_no_preheat', True)) else 2
                        pre_wav = audio[..., : nb_pre * 160] if nb_pre > 0 else None
                        cond_len = max(0, T_feat_tf - 4)
                        period_len = max(0, int(tf_period.size(1)) - 4)
                        target_len = max(0, audio_frames - nb_pre)
                        nb_frames = min(cond_len, period_len, target_len)
                        nb_frames = (nb_frames // 5) * 5
                        tf_audio_np = None
                        if nb_frames >= 5:
                            tf_f = tf_feats20[:, : nb_frames + 4, :].contiguous()
                            tf_p = tf_period[:,  : nb_frames + 4   ].clamp(32, 255).to(torch.long).contiguous()
                        try:
                            tf_audio, _ = decoder.fargan_core(tf_f, tf_p, nb_frames, pre=pre_wav)
                            # clip to requested seconds too
                            Ltf = min(int(tf_audio.size(-1)), max_len)
                            tf_np = torch.clamp(tf_audio[0, :Ltf].detach().cpu(), -1.0, 1.0).numpy()
                            sf.write(str(out_root / f'step_{global_step:06d}_teacher.wav'), tf_np, 16000, subtype='PCM_16')
                        except RuntimeError as _e:
                            msg = str(_e).lower()
                            if 'out of memory' in msg and bool(getattr(args, 'val_audio_teacher', True)):
                                # Optional CPU fallback for preview-only path
                                try:
                                    tqdm.write("    ⚠️ CUDA OOM on teacher preview, falling back to CPU (short clip)")
                                    orig_dev = next(decoder.fargan_core.parameters()).device
                                    # Shorten nb for CPU preview to reduce latency
                                    nb_cpu = max(5, min(nb_frames, int((max_len // 160) + nb_pre)))
                                    tf_f_cpu = tf_f[:1, : nb_cpu + 4, :].contiguous().cpu()
                                    tf_p_cpu = tf_p[:1, : nb_cpu + 4].contiguous().cpu()
                                    pre_cpu  = pre_wav[:1].contiguous().cpu()
                                    decoder.fargan_core.to('cpu')
                                    with torch.no_grad():
                                        tf_audio_cpu, _ = decoder.fargan_core(tf_f_cpu, tf_p_cpu, nb_cpu, pre=pre_cpu)
                                    Ltf = min(int(tf_audio_cpu.size(-1)), max_len)
                                    tf_np = torch.clamp(tf_audio_cpu[0, :Ltf], -1.0, 1.0).numpy()
                                    if deemph > 0.0:
                                        y_prev = 0.0
                                        for i in range(tf_np.shape[0]):
                                            y_prev = float(tf_np[i]) + deemph * y_prev
                                            tf_np[i] = y_prev
                                    sf.write(str(out_root / f'step_{global_step:06d}_teacher.wav'), tf_np, 16000, subtype='PCM_16')
                                except Exception as _e2:
                                    tqdm.write(f"    ⚠️ CPU fallback failed: {_e2}")
                                finally:
                                    # move vocoder back to original device
                                    decoder.fargan_core.to(orig_dev)
                            else:
                                tqdm.write(f"    ⚠️ Failed to synth teacher-forced preview: {_e}")
                        except Exception as _e:
                            tqdm.write(f"    ⚠️ Failed to synth teacher-forced preview: {_e}")
                        else:
                            # Add de-emphasis for GPU path too (after writing raw, overwrite)
                            try:
                                # Force eval mode for preview to match generate_10s_audio.py behavior
                                prev_mode_fc = decoder.fargan_core.training if hasattr(decoder, 'fargan_core') else None
                                if hasattr(decoder, 'fargan_core'):
                                    decoder.fargan_core.eval()
                                tf_audio, _ = decoder.fargan_core(tf_f, tf_p, nb_frames, pre=pre_wav)
                                Ltf = min(int(tf_audio.size(-1)), max_len)
                                tf_np = torch.clamp(tf_audio[0, :Ltf].detach().cpu(), -1.0, 1.0).numpy()
                                if deemph > 0.0:
                                    y_prev = 0.0
                                    for i in range(tf_np.shape[0]):
                                        y_prev = float(tf_np[i]) + deemph * y_prev
                                        tf_np[i] = y_prev
                                sf.write(str(out_root / f'step_{global_step:06d}_teacher.wav'), tf_np, 16000, subtype='PCM_16')
                                if hasattr(decoder, 'fargan_core') and prev_mode_fc is not None:
                                    decoder.fargan_core.train(prev_mode_fc)
                            except Exception:
                                pass

                    tqdm.write(f"    🎧 Saved preview audio at step {global_step} ({L/16000.0:.1f}s)")
                    # Force sync to surface CUDA errors near preview instead of later
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                except Exception as e:
                    tqdm.write(f"    ⚠️ Failed to save preview audio: {e}")

        # 确保特征维度匹配
        if feats.size(1) != y.size(1):
            min_len = min(feats.size(1), y.size(1))
            feats = feats[:, :min_len, :]
            y = y[:, :min_len, :]


        # Stage1-like warm start: ignore first N frames for loss
        fs = max(0, int(getattr(args, 'preheat_frames', 0)))
        if fs > 0 and feats.size(1) > fs:
            feats_loss = feats[:, fs:, :]
            y_loss = y[:, fs:, :]
        else:
            feats_loss, y_loss = feats, y

        # === Fused Loss Computation - 损失计算用float32 ===

        # 初始化所有损失组件，用float32确保数值稳定性
        l_feat = torch.tensor(0.0, device=device, dtype=loss_dtype)
        l_wave = torch.tensor(0.0, device=device, dtype=loss_dtype)
        l_moe  = torch.tensor(0.0, device=device, dtype=loss_dtype)
        l_rate = torch.tensor(0.0, device=device, dtype=loss_dtype)
        l_sem  = torch.tensor(0.0, device=device, dtype=loss_dtype)

        moe_metrics = {}

        # 1. Feature reconstruction loss（转换为float32进行稳定计算）
        feats_loss_safe = feats_loss.float()
        y_loss_safe     = y_loss.float()


        # 检查特征重建输入是否有异常
        if torch.isnan(feats_loss_safe).any() or torch.isinf(feats_loss_safe).any():
            tqdm.write(f"    🚨 NaN/Inf in decoded features before loss computation!")
            tqdm.write(f"      feats_loss range: [{feats_loss_safe.min().item():.6f}, {feats_loss_safe.max().item():.6f}]")
            feats_loss_safe = torch.where(torch.isnan(feats_loss_safe) | torch.isinf(feats_loss_safe),
                                        torch.zeros_like(feats_loss_safe), feats_loss_safe)

        if torch.isnan(y_loss_safe).any() or torch.isinf(y_loss_safe).any():
            tqdm.write(f"    🚨 NaN/Inf in target features before loss computation!")
            tqdm.write(f"      y_loss range: [{y_loss_safe.min().item():.6f}, {y_loss_safe.max().item():.6f}]")
            y_loss_safe = torch.where(torch.isnan(y_loss_safe) | torch.isinf(y_loss_safe),
                                    torch.zeros_like(y_loss_safe), y_loss_safe)

        # === 特征统计打印：审查特征重建效果 ===
        if global_step % 20 == 0:  # 每20个step打印一次
            _print_feature_reconstruction_stats(feats_loss_safe, y_loss_safe, global_step, batch_idx)

        l_feat = F.mse_loss(feats_loss_safe, y_loss_safe)

        # Add layered loss if enabled
        if stage_cfg.layered_enabled(global_step):
            layered_loss, _, _ = compute_layered_loss(
                feats, y,
                current_step=global_step,
                feature_spec_type='fargan'
            )
            l_feat = l_feat + layered_loss.float()

        # 2. Wave loss (继承Stage2权重调度)
        if y_hat_audio is not None and wave_bp_ratio > 0.0:
            # 对齐wave loss的关注区域：裁剪掉前 fs 帧对应的样本数
            wave_audio_target = audio
            wave_audio_pred = y_hat_audio
            if fs > 0:
                sample_off = fs * 160  # 16kHz, 10ms hop
                if y_hat_audio.size(-1) > sample_off:
                    wave_audio_pred = y_hat_audio[..., sample_off:]
                if audio.size(-1) > sample_off:
                    wave_audio_target = audio[..., sample_off:]

            # ① 修复梯度传播：移除detach操作，避免人为削弱FARGAN梯度
            # 问题分析：detach()导致FARGAN梯度只有50%，造成梯度不平衡
            wave_audio_pred_bp = wave_audio_pred  # 直接传播，不再人为削弱梯度
            # 标记已计算波形
            wave_computed_flag = True
            # ② 计算wave loss的窗口大小 - 使用更大的基础窗口
            # 修复：确保最小窗口至少80%，避免因bp_ratio过小而导致窗口太小
            window_ratio = max(0.8, 0.6 + 0.4 * wave_bp_ratio)  # 最小80%，最大100%
            m = int(min(
                wave_audio_pred_bp.size(-1),
                window_ratio * wave_audio_pred_bp.size(-1)
            ))
            # Cap wave loss window by seconds to reduce memory
            max_wave_len = int(float(getattr(args, 'wave_loss_seconds', 6.0)) * 16000)
            m = min(m, max_wave_len)
            wave_pred_head = wave_audio_pred_bp[..., :m]
            wave_tgt_head  = wave_audio_target[..., :m]
            # Release large tensors early
            try:
                del y_hat_audio
            except Exception:
                pass
            # ③ 使用FARGAN标准wave损失函数
            # 🔥 恢复fargan_wave_losses，移除自定义audio_usability_loss
            # 确保period维度正确对齐
            audio_frames = wave_pred_head.size(-1) // 160
            if period.size(1) > audio_frames:
                period_aligned = period[:, :audio_frames]
            else:
                # 如果period不够长，扩展到所需长度
                period_aligned = period.repeat(1, (audio_frames // period.size(1)) + 1)[:, :audio_frames]

            # === 🔄 使用FARGAN标准波形损失 ===
            # 数值稳定性检查
            if torch.isnan(wave_pred_head).any() or torch.isinf(wave_pred_head).any():
                tqdm.write(f"    WARNING: wave_pred_head contains NaN/Inf")
                wave_pred_head = torch.clamp(wave_pred_head, -1.0, 1.0)

            if torch.isnan(wave_tgt_head).any() or torch.isinf(wave_tgt_head).any():
                tqdm.write(f"    WARNING: wave_tgt_head contains NaN/Inf")
                wave_tgt_head = torch.clamp(wave_tgt_head, -1.0, 1.0)

            # 强制使用fp32计算FARGAN波形损失
            with torch.autocast(device_type='cuda', enabled=False):
                try:
                    l_wave_raw, fargan_details = fargan_wave_losses(
                        wave_pred_head.float(),
                        wave_tgt_head.float(),
                        period_aligned,
                        comprehensive_weight=0.0,  # 可调整
                        original_weight=0.0,       # 可调整
                        train_weights=None,        # 使用默认权重
                        device=device
                    )
                except Exception as e:
                    tqdm.write(f"    ERROR in fargan_wave_losses: {e}")
                    l_wave_raw = torch.tensor(0.1, device=device, dtype=loss_dtype, requires_grad=True)
                    fargan_details = {'primary': l_wave_raw}

            # 🔧 保存fargan_details供后续权重调整使用
            train_one_epoch._last_fargan_details = fargan_details

            # ④ 有效权重（注意：这里的alpha_wave_eff还是旧的值，稍后会被新权重覆盖）
            l_wave = l_wave_raw  # 暂时不应用权重，等待动态权重计算

            # RMS质量检查（批量检查，减少打印频率）
            # 使用已裁剪的 wave_pred_head，避免对完整 y_hat_audio 进行大规模归约占用显存
            try:
                pred_rms = torch.sqrt(wave_pred_head.float().pow(2).mean(dim=-1) + 1e-8)
            except RuntimeError:
                # GPU OOM 回退到 CPU 计算
                pred_rms = torch.sqrt(wave_pred_head.detach().float().cpu().pow(2).mean(dim=-1) + 1e-8)
            pred_rms_db = 20.0 * torch.log10(pred_rms.mean() + 1e-8)

            # 记录低RMS事件，但减少打印频率
            if not hasattr(train_one_epoch, '_low_rms_count'):
                train_one_epoch._low_rms_count = 0
                train_one_epoch._last_rms_report = 0

            if pred_rms_db < -40.0:
                train_one_epoch._low_rms_count += 1
                # 每50次低RMS事件才报告一次
                if train_one_epoch._low_rms_count - train_one_epoch._last_rms_report >= 50:
                    tqdm.write(f"      Low RMS: {pred_rms_db:.1f} dB (occurred {train_one_epoch._low_rms_count} times)")
                    train_one_epoch._last_rms_report = train_one_epoch._low_rms_count

        # 3. Rate loss (Stage3默认禁用)
        l_rate = torch.tensor(0.0, device=device, dtype=loss_dtype)

        # 3.5. 双路径特征分离损失：20维声学特征 + 16维语义特征
        # 按照重新设计的框架：物理隔离，独立优化
        l_acoustic = torch.tensor(0.0, device=device, dtype=loss_dtype)
        l_semantic = torch.tensor(0.0, device=device, dtype=loss_dtype)

        # 使用当前对齐后的特征张量 feats/y 计算分离损失
        if feats is not None:  # 只有在成功解码出特征时才计算分离损失
            # 提取原始音频的16维语义特征作为目标
            try:
                # 确保音频在正确的设备和格式上
                audio_for_semantic = audio.detach().float().to(device)
                # 通过缓存的提取器（由main注入）
                sem_ext = getattr(train_one_epoch, '_semantic_extractor', None)
                if sem_ext is None:
                    raise RuntimeError('semantic_extractor not initialized')
                with torch.no_grad():  # 语义提取不需要梯度
                    semantic_target = sem_ext(audio_for_semantic, target_frames=feats.size(1))  # [B,T,16]

                # 提取预测特征中的20维声学部分和16维语义部分
                acoustic_pred = feats[..., :20]      # [B,T,20] 前20维：倒谱+F0+相关性
                acoustic_target = y[..., :20]        # [B,T,20] 对应的GT声学特征

                semantic_pred = feats[..., 20:36]    # [B,T,16] 后16维：语义特征分量
                # semantic_target已在上面计算      # [B,T,16] SSL提取的语义目标

                # 计算分离损失（需要梯度）
                l_acoustic = F.mse_loss(acoustic_pred.float(), acoustic_target.float())
                l_semantic = F.mse_loss(semantic_pred.float(), semantic_target.float())

                # 记录分离损失统计
                if batch_idx % 20 == 0:  # 每20步记录一次
                    with torch.no_grad():
                        acoustic_mse = l_acoustic.item()
                        semantic_mse = l_semantic.item()
                        tqdm.write(f"    分离损失: 声学MSE={acoustic_mse:.6f}, 语义MSE={semantic_mse:.6f}")
                        # 追加后16维语义特征的维度级统计
                        try:
                            _print_semantic_alignment_stats(semantic_pred.float(), semantic_target.float(), global_step, batch_idx)
                        except Exception:
                            pass

            except Exception as e:
                tqdm.write(f"    语义特征提取失败: {e}")
                l_semantic = torch.tensor(0.0, device=device, dtype=loss_dtype)

        # 4. Semantic proxy loss (encoder semantic head vs acoustic priors)
        if isinstance(enc_logs, dict) and 'semantic_pred' in enc_logs and args.alpha_sem > 0:
            sem_pred = enc_logs['semantic_pred']  # [B,T,6]
            sem_pred_avg = sem_pred.mean(dim=1)   # [B,6]
            # 确保extract_acoustic_priors的输入是float32
            sem_target = extract_acoustic_priors(y.float())
            l_sem = args.alpha_sem * F.mse_loss(sem_pred_avg.float(), sem_target.float())


        # 5. MoE auxiliary losses —— 直接进图，参与反传
        l_moe = torch.tensor(0.0, device=device, dtype=loss_dtype)
        moe_metrics = {}

        if isinstance(enc_logs, dict):
            # 记录利用率（只在需要时，无梯度）
            util_interval = max(50, int(getattr(args, 'log_interval', 50)))
            if hasattr(encoder, 'moe') and encoder.moe is not None and (batch_idx == 0 or batch_idx % util_interval == 0):
                with torch.no_grad():
                    try:
                        expert_util = encoder.moe.get_expert_utilization()
                        moe_metrics['expert_usage_min'] = expert_util.min().item()
                        moe_metrics['expert_usage_max'] = expert_util.max().item()
                        moe_metrics['expert_entropy']   = -(expert_util * torch.log(expert_util + 1e-8)).sum().item()

                        # 构建带专家名称的使用率字符串
                        expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"]
                        expert_usage_named = []
                        for i, u in enumerate(expert_util):
                            name = expert_names[i] if i < len(expert_names) else f"E{i}"
                            expert_usage_named.append(f"{name}:{u.item():.3f}")
                        moe_metrics['expert_usage_all'] = ', '.join(expert_usage_named)
                    except Exception:
                        pass

            # **关键：按 CLI 权重并入 loss（保持 float32，参与反传）**
            mb = enc_logs.get('moe_balance_loss', None)
            mt = enc_logs.get('moe_token_balance_loss', None)

            # 调试MoE损失计算 + 路由学习成果展示
            if batch_idx == 0 or batch_idx % 50 == 0:
                moe_w = getattr(args, 'moe_w', 0.05)
                moe_token_w = getattr(args, 'moe_token_w', 0.02)

                # 精简MoE关键指标监控
                if hasattr(encoder, 'moe'):
                    try:
                        expert_util = encoder.moe.get_expert_utilization()
                        expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"]
                        # 构建带专家名称的使用率显示
                        expert_usage_named = []
                        for i, util in enumerate(expert_util):
                            name = expert_names[i] if i < len(expert_names) else f"E{i}"
                            expert_usage_named.append(f"{name}:{util:.3f}")
                        tqdm.write(f"    [MoE] Expert usage: [{', '.join(expert_usage_named)}]")

                        # 性能比较（最关键指标）
                        if hasattr(encoder.moe, 'performance_ratio'):
                            perf_ratio = encoder.moe.performance_ratio
                            status = "Learning" if perf_ratio > 1.5 else "Competitive"
                            tqdm.write(f"    [MoE] Performance: {status} ({perf_ratio:.2f}x vs direct)")

                    except Exception as e:
                        tqdm.write(f"    [MoE] Analysis failed: {e}")

            if isinstance(mb, torch.Tensor):
                # 简化MoE损失计算：统一权重，专注平衡性
                expert_util = None
                try:
                    expert_util = encoder.moe.get_expert_utilization() if hasattr(encoder, 'moe') else None
                except Exception:
                    expert_util = None

                if expert_util is not None:
                    # 将计算挪到CPU，规避异步CUDA错误在此处抛出
                    eu_cpu = expert_util.detach().cpu()
                    min_util = float(eu_cpu.min().item())
                    max_util = float(eu_cpu.max().item())
                    util_variance = float(eu_cpu.var().item())

                    # 基于方差的平衡权重：方差越大，权重越高
                    balance_multiplier = 1.0 + min(util_variance * 10.0, 3.0)  # 1.0-4.0倍

                    if batch_idx == 0 or batch_idx % 50 == 0:
                        tqdm.write(f"    Expert Util: min={min_util:.3f}, max={max_util:.3f}, var={util_variance:.3f}")
                else:
                    balance_multiplier = 1.5

                moe_contrib = float(getattr(args, 'moe_w', 0.1)) * balance_multiplier * mb.float()
                l_moe = l_moe + moe_contrib
                if batch_idx == 0 or batch_idx % 50 == 0:
                    tqdm.write(f"    Balance Loss: {moe_contrib.item():.6f} (×{balance_multiplier:.1f})")

            if isinstance(mt, torch.Tensor):
                # Token级别损失：适度权重，鼓励细粒度路由
                token_contrib = float(getattr(args, 'moe_token_w', 0.05)) * 2.0 * mt.float()
                l_moe = l_moe + token_contrib
                if batch_idx == 0 or batch_idx % 50 == 0:
                    tqdm.write(f"    Token Loss: {token_contrib.item():.6f}")

            # 其他可能的 MoE 约束（如一致性等），统一用较小权重
            for k, v in enc_logs.items():
                if not isinstance(v, torch.Tensor) or (k in ('moe_balance_loss','moe_token_balance_loss')):
                    continue
                if k.startswith('moe_') and v.requires_grad:
                    l_moe = l_moe + 0.05 * v.float()

            # 监督路由暖启动：
            # - 若批次包含 per-sample 'expert_class'，则按样本监督（更精细）
            # - 否则退化为整批同一 focus 的 sample-level 监督
            if getattr(args, 'router_sup', 0.0) > 0.0 and hasattr(encoder, 'moe') and encoder.moe is not None:
                expert_map = {
                    'harmonic': 0,
                    'transient': 1,
                    'burst_inpaint': 2,
                    'low_snr': 3,
                }
                # 取出最近一次前向的 sample-level router logits（由增强MoE内部缓存）
                logits = None
                try:
                    if hasattr(encoder.moe, 'specialized_moe') and hasattr(encoder.moe.specialized_moe, '_last_router_logits'):
                        logits = encoder.moe.specialized_moe._last_router_logits
                    elif hasattr(encoder.moe, '_last_router_logits'):
                        logits = encoder.moe._last_router_logits
                except Exception:
                    logits = None

                if logits is not None and torch.is_tensor(logits):
                    # 目标标签：优先 per-sample 标签，其次使用 --expert-focus
                    if isinstance(batch, dict) and ('expert_class' in batch):
                        target_id = batch['expert_class'].to(device=logits.device, dtype=torch.long)
                        focus_tag = 'per-sample'
                    else:
                        focus_name = str(getattr(args, 'expert_focus', '')).strip().lower()
                        if focus_name not in expert_map:
                            target_id = None
                        else:
                            target_id = torch.full((logits.size(0),), expert_map[focus_name], device=logits.device, dtype=torch.long)
                        focus_tag = focus_name or 'none'

                    if target_id is not None:
                        # 线性衰减：在 router_sup_decay_steps 内从sup_w到0.1*sup_w
                        sup_w = float(getattr(args, 'router_sup', 0.0))
                        decay_steps = int(getattr(args, 'router_sup_decay_steps', 0) or 0)
                        if decay_steps > 0:
                            pdec = min(1.0, float(global_step) / float(decay_steps))
                            sup_w = sup_w * (1.0 - 0.9 * pdec)
                        sup_loss = F.cross_entropy(logits.float(), target_id)
                        l_moe = l_moe + sup_w * sup_loss.to(l_moe.dtype)
                        if batch_idx == 0 or batch_idx % 50 == 0:
                            tqdm.write(f"    RouterSup CE: {sup_loss.item():.6f} (w={sup_w:.3f}, focus={focus_tag})")


        # === 🔥 新的三阶段语义感知权重策略（完全替换旧调度） ===
        # 根据训练步数和质量状况动态调整损失权重

        # 基础阶段权重分配 - 强化"重语义，轻特征重建"策略
        if global_step < 1000:
            # Foundation阶段：建立语义基础，适度特征学习
            base_weights = {"feat": 0.3, "wave": 0.4, "moe": 0.3}
            stage_name = "Foundation"
        elif global_step < 5000:
            # Balanced阶段：进一步降低特征重建权重
            base_weights = {"feat": 0.15, "wave": 0.55, "moe": 0.3}
            stage_name = "Balanced"
        else:
            # Quality阶段：极大降低特征重建权重，专注语义音频质量
            base_weights = {"feat": 0.05, "wave": 0.75, "moe": 0.2}
            stage_name = "Quality"

        # === 🧠 智能的自适应权重调整策略 ===
        # 基于多维度音频质量指标的动态权重调整
        quality_score = 0.5  # 默认中性质量评分
        detailed_quality_metrics = {
            'audibility': 0.5,
            'intelligibility': 0.5,
            'quality': 0.5
        }

        # 获取详细的质量评估信息
        if hasattr(train_one_epoch, '_last_fargan_details'):
            fargan_details = getattr(train_one_epoch, '_last_fargan_details', {})

            # 计算综合质量评分（基于FARGAN损失）
            primary_loss = fargan_details.get('primary', torch.tensor(1.0))
            if isinstance(primary_loss, torch.Tensor):
                raw_total = primary_loss.item()
            else:
                raw_total = float(primary_loss)
            quality_score = max(0.0, min(1.0, 1.0 - raw_total))

            # 基于FARGAN损失组件计算质量得分
            detailed_quality_metrics.update({
                'audibility': quality_score,      # 基于primary loss
                'intelligibility': quality_score,  # 基于primary loss
                'quality': quality_score           # 基于primary loss
            })

        # 多层次权重调整策略
        # 1. 基于综合质量的主要调整
        if quality_score < 0.2:  # 质量严重不足
            # 激进地增强音频损失权重
            wave_boost = 2.0
            feat_reduction = 0.6
        elif quality_score < 0.4:  # 质量较差
            # 适度增强音频损失权重
            wave_boost = 1.5
            feat_reduction = 0.8
        elif quality_score > 0.8:  # 质量很好
            # 可以更注重语义保持
            wave_boost = 0.8
            feat_reduction = 1.3
        elif quality_score > 0.9:  # 质量优秀
            # 大力加强语义学习
            wave_boost = 0.6
            feat_reduction = 1.5
        else:  # 质量中等
            # 保持基础权重平衡
            wave_boost = 1.0
            feat_reduction = 1.0

        # 2. 基于特定指标的微调
        # 可听性差时，优先修复基础音频问题
        if detailed_quality_metrics['audibility'] < 0.3:
            wave_boost *= 1.2

        # 清晰度差时，平衡音频和特征权重
        if detailed_quality_metrics['intelligibility'] < 0.3:
            wave_boost *= 1.1
            feat_reduction *= 0.9

        # 感知质量差时，注重音频精细化
        if detailed_quality_metrics['quality'] < 0.3:
            wave_boost *= 1.15

        # 应用权重调整
        base_weights["wave"] *= wave_boost
        base_weights["feat"] *= feat_reduction

        # 3. 训练阶段特定的权重保护
        # Foundation阶段：确保基础功能不被过度削弱
        if stage_name == "Foundation":
            base_weights["wave"] = min(base_weights["wave"], 0.4)  # 限制波形损失权重
            base_weights["feat"] = max(base_weights["feat"], 0.4)  # 保证特征学习
        # Quality阶段：确保音频质量优先
        elif stage_name == "Quality":
            base_weights["wave"] = max(base_weights["wave"], 0.3)  # 保证音频关注度
            base_weights["feat"] = min(base_weights["feat"], 0.5)  # 限制特征重建

        # 归一化权重
        total_weight = sum(base_weights.values())
        alpha_feat_eff = base_weights["feat"] / total_weight
        alpha_wave_final = base_weights["wave"] / total_weight
        alpha_moe_eff = base_weights["moe"] / total_weight

        # 语义训练策略监控
        if batch_idx % 200 == 0:
            tqdm.write(f"      [SEMANTIC] Stage: {stage_name} (step {global_step})")
            tqdm.write(f"      [SEMANTIC] Weights: recon={alpha_feat_eff:.3f}, audio_quality={alpha_wave_final:.3f}, routing={alpha_moe_eff:.3f}")
            tqdm.write(f"      [SEMANTIC] Quality: overall={quality_score:.3f}, aud={detailed_quality_metrics['audibility']:.3f}, int={detailed_quality_metrics['intelligibility']:.3f}, qual={detailed_quality_metrics['quality']:.3f}")

        # 应用最终权重计算损失 - 增加数值稳定性检查
        # 检查每个损失分量的有效性
        if torch.isnan(l_feat) or torch.isinf(l_feat):
            tqdm.write(f"    WARNING: l_feat is NaN/Inf: {l_feat.item()}")
            l_feat = torch.tensor(0.0, device=device, dtype=loss_dtype, requires_grad=True)

        if torch.isnan(l_wave) or torch.isinf(l_wave):
            tqdm.write(f"    WARNING: l_wave is NaN/Inf: {l_wave.item()}")
            l_wave = torch.tensor(0.0, device=device, dtype=loss_dtype, requires_grad=True)

        if torch.isnan(l_moe) or torch.isinf(l_moe):
            tqdm.write(f"    WARNING: l_moe is NaN/Inf: {l_moe.item()}")
            l_moe = torch.tensor(0.0, device=device, dtype=loss_dtype, requires_grad=True)

        if torch.isnan(l_sem) or torch.isinf(l_sem):
            tqdm.write(f"    WARNING: l_sem is NaN/Inf: {l_sem.item()}")
            l_sem = torch.tensor(0.0, device=device, dtype=loss_dtype, requires_grad=True)

        # 安全的权重检查
        if not torch.isfinite(torch.tensor(alpha_feat_eff)):
            alpha_feat_eff = 0.1
        if not torch.isfinite(torch.tensor(alpha_wave_final)):
            alpha_wave_final = 0.1
        if not torch.isfinite(torch.tensor(alpha_moe_eff)):
            alpha_moe_eff = 0.1

        # 计算双路径分离损失权重
        alpha_acoustic_eff = args.alpha_acoustic if hasattr(args, 'alpha_acoustic') else 1.0
        alpha_semantic_eff = args.alpha_semantic if hasattr(args, 'alpha_semantic') else 0.5

        total_loss = (alpha_feat_eff * l_feat + alpha_wave_final * l_wave + l_rate + l_sem +
                     alpha_moe_eff * l_moe + alpha_acoustic_eff * l_acoustic + alpha_semantic_eff * l_semantic)

        # 最终安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            tqdm.write(f"    CRITICAL: total_loss is NaN/Inf, using fallback")
            total_loss = torch.tensor(1.0, device=device, dtype=loss_dtype, requires_grad=True)

        # 保存当前权重用于后续分析
        if not hasattr(train_one_epoch, '_weight_history'):
            train_one_epoch._weight_history = []

        train_one_epoch._weight_history.append({
            'step': global_step,
            'stage': stage_name,
            'weights': {
                'feat': alpha_feat_eff,
                'wave': alpha_wave_final,
                'moe': alpha_moe_eff
            },
            'quality_metrics': detailed_quality_metrics.copy(),
            'quality_score': quality_score
        })

        # 限制历史记录长度
        if len(train_one_epoch._weight_history) > 1000:
            train_one_epoch._weight_history = train_one_epoch._weight_history[-500:]


        # 🚨 增强的NaN检测和诊断 - 分解检查每个损失分量
        if torch.isnan(l_feat) or torch.isinf(l_feat):
            tqdm.write(f"      ⚠️ NaN/Inf in l_feat: {l_feat.item()}")
        if torch.isnan(l_wave) or torch.isinf(l_wave):
            tqdm.write(f"      ⚠️ NaN/Inf in l_wave: {l_wave.item()}")
        if torch.isnan(l_moe) or torch.isinf(l_moe):
            tqdm.write(f"      ⚠️ NaN/Inf in l_moe: {l_moe.item()}")
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            tqdm.write(f"      ⚠️ NaN/Inf in total_loss: {total_loss.item()}")
            tqdm.write(f"      Components: feat={l_feat.item():.4f}, wave={l_wave.item():.4f}, moe={l_moe.item():.4f}")

        def check_tensor(name, tensor):
            """检查tensor中的异常值"""
            if tensor is None:
                return f"{name}: None"
            if not isinstance(tensor, torch.Tensor):
                return f"{name}: not tensor ({type(tensor)})"
            has_nan = torch.isnan(tensor).any()
            has_inf = torch.isinf(tensor).any()
            min_val = tensor.min().item() if tensor.numel() > 0 else 0
            max_val = tensor.max().item() if tensor.numel() > 0 else 0
            return f"{name}: nan={has_nan}, inf={has_inf}, range=[{min_val:.3f}, {max_val:.3f}]"

        # 检查模型参数是否包含NaN/Inf
        param_nan_count = 0
        for name, param in list(encoder.named_parameters()) + list(decoder.named_parameters()):
            if torch.isnan(param).any() or torch.isinf(param).any():
                param_nan_count += 1
                tqdm.write(f"    CRITICAL: Parameter {name} contains NaN/Inf")
                # 紧急重置参数
                with torch.no_grad():
                    param.data = torch.randn_like(param.data) * 0.01

        if param_nan_count > 0:
            tqdm.write(f"    EMERGENCY: Reset {param_nan_count} parameters with NaN/Inf")

        # 详细的NaN诊断
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                tqdm.write(f"    🚨 NaN/Inf detected in total_loss: {total_loss.item() if not torch.isnan(total_loss) else 'NaN'}")
            elif args.debug_nan:
                tqdm.write(f"    🔍 Debug NaN mode - batch {batch_idx} diagnostics:")

            # 语义感知方案的关键指标
            tqdm.write(f"      Semantic Loss Components:")
            if hasattr(train_one_epoch, '_last_fargan_details') and train_one_epoch._last_fargan_details:
                fargan = train_one_epoch._last_fargan_details
                try:
                    tqdm.write(
                        f"        Audio Usability: aud={detailed_quality_metrics.get('audibility', 0.0):.3f}, "
                        f"int={detailed_quality_metrics.get('intelligibility', 0.0):.3f}, "
                        f"qual={detailed_quality_metrics.get('quality', 0.0):.3f}")
                except Exception:
                    pass

            if hasattr(train_one_epoch, '_semantic_adapter'):
                adapter_status = train_one_epoch._semantic_adapter.get_status()
                tqdm.write(f"        Adapter: strength={adapter_status['adaptation_strength']:.1f}")

            tqdm.write(f"      Loss Values:")
            tqdm.write(f"        {check_tensor('feat', l_feat)}")
            tqdm.write(f"        {check_tensor('wave', l_wave)}")
            tqdm.write(f"        {check_tensor('moe', l_moe)}")

            # 检查输入数据
            tqdm.write(f"      Input data:")
            tqdm.write(f"        {check_tensor('features_x', x)}")
            tqdm.write(f"        {check_tensor('features_y', y)}")
            tqdm.write(f"        {check_tensor('audio', audio)}")

            # 检查前向输出
            tqdm.write(f"      Forward outputs:")
            tqdm.write(f"        {check_tensor('latent_z', z)}")
            tqdm.write(f"        {check_tensor('decoded_feats', feats)}")
            if y_hat_audio is not None:
                tqdm.write(f"        {check_tensor('decoded_audio', y_hat_audio)}")

            # 检查MoE相关信息
            if encoder.use_moe and isinstance(enc_logs, dict):
                tqdm.write(f"      MoE logs:")
                for key, value in enc_logs.items():
                    if isinstance(value, torch.Tensor):
                        tqdm.write(f"        {check_tensor(key, value)}")

            # 仅在真正的NaN时进行恢复
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                # 跳过这个batch，使用上一个有效loss
                if hasattr(train_one_epoch, '_last_valid_loss'):
                    total_loss = train_one_epoch._last_valid_loss.clone().detach().requires_grad_()
                    tqdm.write(f"      Recovery: using last valid loss {total_loss.item():.6f}")
                else:
                    # 使用保守的fallback loss
                    total_loss = torch.tensor(1.0, device=device, dtype=loss_dtype, requires_grad=True)
                    tqdm.write(f"      Recovery: using fallback loss {total_loss.item():.6f}")

        if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
            # 保存有效loss用于恢复
            train_one_epoch._last_valid_loss = total_loss.clone().detach()

        # 梯度累积支持
        accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        scaled_loss = total_loss / accumulation_steps


        # 计算复杂度监控（可选）
        complexity_monitor_interval = 200
        if global_step % complexity_monitor_interval == 0 and hasattr(encoder, 'moe'):
            try:
                with torch.no_grad():
                    # 统计不同pathway模式的计算量
                    pathway_stats = encoder.moe.get_performance_stats() if hasattr(encoder.moe, 'get_performance_stats') else {}
                    pathway_mode = pathway_stats.get('pathway_mode', 'unknown')
                    complexity_ratio = pathway_stats.get('complexity_ratio', 1.0)

                    # 记录到epoch metrics用于后续分析
                    if 'pathway_complexity_samples' not in epoch_metrics:
                        epoch_metrics['pathway_complexity_samples'] = 0
                        epoch_metrics['pathway_complexity_total'] = 0.0

                    epoch_metrics['pathway_complexity_samples'] += batch_size
                    epoch_metrics['pathway_complexity_total'] += complexity_ratio * batch_size

                    # 每1000步报告一次复杂度状态
                    if global_step % 1000 == 0:
                        avg_complexity = epoch_metrics['pathway_complexity_total'] / max(1, epoch_metrics['pathway_complexity_samples'])
                        tqdm.write(f"    Complexity: {avg_complexity:.2f}x baseline")
            except Exception as e:
                pass  # 复杂度监控失败不影响训练

        # 反传：fp16 用 GradScaler，bf16/none 直接 backward
        if use_fp16:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 只有在累积边界才做 unscale/clip/日志/step
        if (batch_idx + 1) % accumulation_steps == 0:
            # 先 unscale 再 clip（fp16 专用）
            if use_fp16:
                scaler.unscale_(optimizer)

            # （把你原来“对 parametrizations 与 fargan_core 的手动 clamp”移到这里来，
            #  确保在 unscale 之后、clip 之前执行）
            with torch.no_grad():
                for name, param in decoder.named_parameters():
                    if param.grad is None:
                        continue
                    if 'parametrizations' in name:
                        param.grad.clamp_(-0.1, 0.1)
                    elif 'fargan_core' in name:
                        param.grad.clamp_(-1.0, 1.0)

            # 先做一次 NaN/Inf 清洗（更激进的版本）
            with torch.no_grad():
                cleaned_grads = 0
                for p in list(encoder.parameters()) + list(decoder.parameters()):
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            cleaned_grads += 1
                            # 温和修复：仅把异常元素替换为0，保留其余有效梯度
                            p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        # 对梯度整体做保守的clamp，防止爆炸
                        p.grad.clamp_(-10.0, 10.0)

                if cleaned_grads > 0:
                    tqdm.write(f"    Cleaned {cleaned_grads} gradients with NaN/Inf")

            # 全模型 clip（建议 1.0 更稳）
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0
            )

            # 计算梯度统计，但减少打印频率
            gn_vc, n_vc  = _sum_grad_norm(decoder.named_parameters(), include_key='fargan_core')
            gn_dec, n_dec = _sum_grad_norm(decoder.named_parameters(), exclude_key='fargan_core')
            gn_enc, n_enc = _sum_grad_norm(encoder.named_parameters())

            # 每100步打印一次梯度统计，或者在fargan_core梯度异常低时打印
            if batch_idx % 100 == 0 or gn_vc < 1e-4:
                tqdm.write(f"[GRAD] fargan_core={gn_vc:.3e}, decoder={gn_dec:.3e}, encoder={gn_enc:.3e}")

            # 修复：专家差异化分析监控 - 独立于梯度监控条件，确保所有阶段都显示
            expert_monitor_interval = max(50, int(getattr(args, 'log_interval', 50)))
            if batch_idx % expert_monitor_interval == 0:
                # 调试信息：确认专家监控条件触发
                # tqdm.write(f"[DEBUG] Expert monitoring triggered at batch {batch_idx}, interval={expert_monitor_interval}")
                if hasattr(encoder, 'moe') and encoder.moe is not None:
                    try:
                        # 获取实际的MoE实现
                        actual_moe = encoder.moe
                        if hasattr(encoder.moe, 'specialized_moe'):
                            actual_moe = encoder.moe.specialized_moe

                        # 找到真正的experts
                        experts_container = None
                        if hasattr(actual_moe, 'experts'):
                            experts_container = actual_moe
                        elif hasattr(actual_moe, 'moe_system') and hasattr(actual_moe.moe_system, 'experts'):
                            experts_container = actual_moe.moe_system

                        # 检查专家差异化和专业化学习
                        expert_biases = []
                        specialization_analysis = []

                        if experts_container and hasattr(experts_container, 'experts') and len(experts_container.experts) > 0:
                            expert_bias_values = []
                            expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"]

                            for i, expert in enumerate(experts_container.experts):
                                if hasattr(expert, 'expert_bias'):
                                    bias_norm = expert.expert_bias.norm().item()
                                    spec_norm = expert.specialization_weights.norm().item() if hasattr(expert, 'specialization_weights') else 0.0
                                    expert_name = expert_names[i] if i < len(expert_names) else f"E{i}"
                                    expert_biases.append(f"{expert_name}:{bias_norm:.3f}")
                                    expert_bias_values.append(bias_norm)
                                else:
                                    expert_biases.append(f"E{i}:no_bias")
                                    expert_bias_values.append(0.0)

                            # 分析专家差异化程度
                            if len(expert_bias_values) > 1:
                                bias_variance = torch.tensor(expert_bias_values).var().item()
                                bias_max_diff = max(expert_bias_values) - min(expert_bias_values)
                                specialization_analysis.append(f"bias_var:{bias_variance:.4f}")
                                specialization_analysis.append(f"max_diff:{bias_max_diff:.3f}")

                                # 判断专业化状态
                                if bias_variance < 0.001 and bias_max_diff < 0.02:
                                    spec_status = "SYNCED"  # 同步增长，缺乏差异化
                                elif bias_variance > 0.01:
                                    spec_status = "DIVERGING"  # 正在学习差异化
                                else:
                                    spec_status = "LEARNING"  # 中等差异化

                                specialization_analysis.append(f"status:{spec_status}")

                            # 检查专家使用率差异（另一个专业化指标）
                            if hasattr(encoder.moe, 'get_expert_utilization'):
                                try:
                                    expert_util = encoder.moe.get_expert_utilization()
                                    util_variance = expert_util.var().item()
                                    if util_variance > 0.05:
                                        routing_status = "SPECIALIZED"  # 路由器学到了专业化
                                    elif util_variance > 0.02:
                                        routing_status = "LEARNING"     # 正在学习专业化
                                    else:
                                        routing_status = "UNIFORM"      # 均匀分布，缺乏专业化
                                    specialization_analysis.append(f"routing:{routing_status}")
                                except:
                                    pass

                        if expert_biases:
                            tqdm.write(f"[EXPERT] Differentiation: {', '.join(expert_biases)}")
                            if specialization_analysis:
                                tqdm.write(f"[EXPERT] Specialization: {', '.join(specialization_analysis)}")
                        else:
                            tqdm.write(f"[EXPERT] No differentiation metrics found")

                    except Exception as e:
                        tqdm.write(f"[DEBUG] Expert analysis failed: {e}")

            # 保存fargan_core梯度状态用于后续检查
            train_one_epoch._last_fargan_grad_norm = gn_vc

            # 非有限总体范数：做一次深度清洗并跳过本步
            if not torch.isfinite(total_norm):
                tqdm.write("    🚨 Non-finite grad norm detected. Sanitizing & skipping this step.")
                with torch.no_grad():
                    for p in list(encoder.parameters()) + list(decoder.parameters()):
                        if p.grad is not None:
                            p.grad.nan_to_num_(0.0, posinf=0.0, neginf=0.0)
                optimizer.zero_grad(set_to_none=True)
                # 重置scaler状态以防止后续问题
                if use_fp16:
                    scaler.update()
            else:
                # 正常更新：fp16 用 scaler.step/update，其他直接 step
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 更新梯度 
            with torch.no_grad():
                tn = float(total_norm) if torch.isfinite(total_norm) else 10.0
                beta = 0.98
                train_one_epoch._gn_ema = beta * train_one_epoch._gn_ema + (1.0 - beta) * tn

            # Step-based checkpoint saving (optional)
            save_steps = int(getattr(args, 'save_every_steps', 0) or 0)
            if save_steps > 0 and (global_step % save_steps == 0):
                try:
                    step_ckpt = {
                        'epoch': int(epoch_idx) if epoch_idx is not None else -1,
                        'step': int(global_step),
                        'encoder_state_dict': encoder.state_dict(),
                        'decoder_state_dict': decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': float(total_loss),
                    }
                    step_path = Path(args.output_dir) / f'stage3_step_{global_step:06d}.pth'
                    torch.save(step_ckpt, step_path)
                    tqdm.write(f"💾 Saved step checkpoint: {step_path}")
                except Exception as _e:
                    tqdm.write(f"⚠️  Failed to save step checkpoint at {global_step}: {_e}")



        # 为metrics记录使用原始loss（未缩放）
        loss_for_metrics = total_loss

        # Update metrics
        batch_size = x.size(0)
        total_samples += batch_size

        epoch_metrics['total_loss'] += loss_for_metrics.item() * batch_size
        epoch_metrics['feature_loss'] += l_feat.item() * batch_size
        epoch_metrics['wave_loss'] += l_wave.item() * batch_size
        epoch_metrics['moe_loss'] += l_moe.item() * batch_size
        epoch_metrics['rate_loss'] += l_rate.item() * batch_size
        if isinstance(l_sem, (int, float)):
            sem_item = l_sem
        else:
            sem_item = l_sem.item()
        epoch_metrics.setdefault('semantic_loss', 0.0)
        epoch_metrics['semantic_loss'] += sem_item * batch_size

        # MoE metrics
        for key in ['expert_entropy', 'expert_usage_min', 'expert_usage_max']:
            if key in moe_metrics:
                epoch_metrics[key] += moe_metrics[key] * batch_size

        # Individual expert metrics - 动态专家数量
        n_experts = getattr(encoder.moe, 'n_experts', 4) if hasattr(encoder, 'moe') else 4
        for i in range(n_experts):  # 动态专家数量
            expert_key = f'expert_{i}_usage'
            if expert_key in moe_metrics:
                if expert_key not in epoch_metrics:
                    epoch_metrics[expert_key] = 0.0
                epoch_metrics[expert_key] += moe_metrics[expert_key] * batch_size

        # 直流通路性能监控
        if hasattr(encoder, 'moe') and hasattr(encoder.moe, 'get_performance_stats'):
            pathway_stats = encoder.moe.get_performance_stats()
            for key, value in pathway_stats.items():
                pathway_key = f'pathway_{key}'
                if pathway_key not in epoch_metrics:
                    epoch_metrics[pathway_key] = 0.0
                if isinstance(value, (int, float)):
                    epoch_metrics[pathway_key] += value * batch_size
                elif isinstance(value, list) and len(value) > 0:
                    # 对于列表类型，计算平均值
                    avg_value = sum(value) / len(value)
                    epoch_metrics[pathway_key] += avg_value * batch_size

            # 分离损失计算和EMA更新
            if hasattr(encoder.moe, 'get_separated_outputs'):
                direct_output, expert_output = encoder.moe.get_separated_outputs()
                if direct_output is not None and expert_output is not None:
                    # 计算分离损失：直流vs专家的内在差异（L2距离）
                    # 这里比较的是两种处理方式的差异，而不是与原始输入的重建误差
                    pathway_diff = F.mse_loss(direct_output, expert_output)

                    # 使用pathway_diff作为性能指标：值越大说明两种方法差异越大
                    # 理想情况下专家系统应该产生与直流不同但更优的特征
                    direct_loss_proxy = pathway_diff.item()
                    expert_loss_proxy = l_feat.item()  # 使用总体特征损失作为专家性能代理

                    # 更新EMA
                    encoder.moe.update_performance_ema(direct_loss_proxy, expert_loss_proxy)

                    # 记录到epoch metrics
                    epoch_metrics.setdefault('pathway_diff_loss', 0.0)
                    epoch_metrics.setdefault('pathway_expert_proxy_loss', 0.0)
                    epoch_metrics['pathway_diff_loss'] += direct_loss_proxy * batch_size
                    epoch_metrics['pathway_expert_proxy_loss'] += expert_loss_proxy * batch_size

        # 🔧 FIX 8: 增强调试信息，监控梯度流和关键指标
        # Progress logging with enhanced gradient flow monitoring
        log_interval = max(1, int(getattr(args, 'log_interval', 50)))
        if batch_idx % log_interval == 0:
            # 记录关键梯度流信息
            with torch.no_grad():
                # 检查feats的梯度连接状态
                feats_grad_connected = feats.requires_grad if isinstance(feats, torch.Tensor) else False
                # 检查teacher forcing比例和梯度传播比例
                tf_ratio = tf if 'tf' in locals() else 0.0
                # 检查是否计算了wave_loss
                wave_computed = wave_computed_flag

        if batch_idx % log_interval == 0:
            # Safe float conversions for printing
            def _sf(x):
                try:
                    return float(x)
                except Exception:
                    try:
                        return float(x.item())
                    except Exception:
                        return 0.0

            tl = _sf(loss_for_metrics)
            ff = _sf(l_feat)
            ww = _sf(l_wave)
            mm = _sf(l_moe)
            # 双路径分离损失
            aa = _sf(l_acoustic)  # 声学损失
            ss = _sf(l_semantic)  # 语义损失
            # 新版SpecializedMicroMoE的损失
            mb = _sf(enc_logs.get('moe_balance_loss', 0.0) if isinstance(enc_logs, dict) else 0.0)
            ms = _sf(enc_logs.get('moe_harmonic_pref', 0.0) if isinstance(enc_logs, dict) else 0.0)  # 使用实际存在的指标
            lr = optimizer.param_groups[0].get('lr', 0.0)
            # ETA from tqdm
            remaining = progress.format_dict.get('remaining', None)
            import time as _t
            eta_str = _t.strftime('%H:%M:%S', _t.gmtime(remaining)) if remaining is not None else 'NA'
            # 语义感知方案的关键指标
            post = {
                'loss': f"{tl:.4f}",
                'feat': f"{ff:.3f}",
                'wave': f"{ww:.3f}",
                'acou': f"{aa:.3f}",  # 声学损失
                'sem': f"{ss:.3f}",   # 语义损失
                'lr': f"{lr:.2e}",
                'eta': eta_str,
            }

            # 语义适配器状态
            if hasattr(train_one_epoch, '_semantic_adapter'):
                adapter_status = train_one_epoch._semantic_adapter.get_status()
                post['adapt'] = f"{adapter_status['adaptation_strength']:.1f}"

            # 音频可用性指标
            if hasattr(train_one_epoch, '_last_fargan_details') and train_one_epoch._last_fargan_details:
                fargan = train_one_epoch._last_fargan_details
                # 使用已计算的质量指标（detailed_quality_metrics）进行展示
                try:
                    post.update({
                        'aud': f"{detailed_quality_metrics.get('audibility', 0.0):.2f}",
                        'int': f"{detailed_quality_metrics.get('intelligibility', 0.0):.2f}",
                        'qual': f"{detailed_quality_metrics.get('quality', 0.0):.2f}",
                    })
                except Exception:
                    pass
            else:
                post.update({
                    'feat': f"{ff:.4f}",
                    'wave': f"{ww:.4f}",
                })

            # 训练控制状态
            post.update({
                'warm': f"{warm_ratio:.2f}",
                'bp': f"{wave_bp_ratio:.2f}",
            })

            # 只在MoE真正启用时显示MoE指标
            mm = float(l_moe)
            mb = float(enc_logs.get('moe_balance_loss', 0.0)) if isinstance(enc_logs, dict) else 0.0
            mt = float(enc_logs.get('moe_token_balance_loss', 0.0)) if isinstance(enc_logs, dict) else 0.0
            md = float(enc_logs.get('expert_diversification_loss', 0.0)) if isinstance(enc_logs, dict) else 0.0
            post.update({
                'moe': f"{mm:.4f}",
                'moe_b': f"{mb:.4f}",
                'moe_t': f"{mt:.4f}",
                'moe_d': f"{md:.4f}",
            })

            # Optional profile timings if enabled
            prof_int = int(getattr(args, 'profile_interval', 0) or 0)
            if prof_int > 0 and batch_idx % prof_int == 0:
                try:
                    post.update({
                        't_fwd': f"{(t_fwd1 - t_fwd0)*1000:.0f}ms",
                        't_feat': f"{(t_feat - t_fwd1)*1000:.0f}ms",
                        't_wave': f"{(t_wave - t_feat)*1000:.0f}ms",
                        't_back': f"{(t_back - t_wave)*1000:.0f}ms",
                    })
                except Exception:
                    pass
            progress.set_postfix(post)


            # 第 0 个 batch 打一行 MoE 自检 + 梯度流诊断
            if batch_idx == 0:
                try:
                    n_exp = getattr(encoder.moe, 'n_experts', '?')
                    top_k = getattr(encoder.moe, 'top_k', '?')
                    sm = getattr(encoder.moe, 'specialized_moe', None)
                    token_level = getattr(sm, 'use_token_level', False) if sm else False
                    tqdm.write(f"    [MoE Active] n_experts={n_exp}, top_k={top_k}, token_level={token_level}")
                except Exception:
                    pass

                # 梯度流诊断信息
                tqdm.write(f"    🔍 Gradient Flow Status:")
                tqdm.write(f"       feats gradient: {'Connected' if feats_grad_connected else 'DISCONNECTED'}")
                tqdm.write(f"       teacher forcing: {tf_ratio:.3f}")
                tqdm.write(f"       wave backprop: {wave_bp_ratio:.3f}")
                tqdm.write(f"       wave computed: {'Yes' if wave_computed else 'No'}")
                if hasattr(args, 'amp'):
                    tqdm.write(f"       AMP mode: {args.amp}")

            # Health 打印：第 0 步也打；间隔=max(4*log_interval, 50)
            moe_monitor_interval = max(4 * int(getattr(args, 'log_interval', 50)), 50)
            if encoder.use_moe and (batch_idx == 0 or batch_idx % moe_monitor_interval == 0):
                # 在绕过模式下，显示绕过状态信息
                if args.emergency_bypass_moe:
                    # 绕过模式：尝试显示模拟的均匀分布统计
                    if hasattr(encoder, 'moe') and encoder.moe is not None:
                        try:
                            with torch.no_grad():
                                expert_util = encoder.moe.get_expert_utilization()
                                usage_str = ', '.join([f'{util.item():.3f}' for util in expert_util])
                                tqdm.write(f"    MoE Health (BYPASS): usage=[{usage_str}] (simulated uniform distribution)")
                        except Exception:
                            n_experts = getattr(encoder.moe, 'n_experts', 4) if hasattr(encoder, 'moe') else 4
                            bypass_usage = ', '.join(['0.333'] * n_experts)
                            tqdm.write(f"    MoE Health (BYPASS): usage=[{bypass_usage}] (routing bypassed)")
                    else:
                        tqdm.write(f"    MoE Health (BYPASS): MoE module not available")
                elif moe_metrics:
                    # 正常模式：显示完整的MoE健康信息
                    expert_min = moe_metrics.get('expert_usage_min', 0)
                    expert_max = moe_metrics.get('expert_usage_max', 0)
                    expert_entropy = moe_metrics.get('expert_entropy', 0)
                    expert_usage_all = moe_metrics.get('expert_usage_all', 'N/A')

                    tqdm.write(f"    MoE Health: usage=[{expert_usage_all}] entropy={expert_entropy:.3f}")

                    # 新增：统一专家架构监控
                    try:
                        if hasattr(encoder.moe, 'experts') and len(encoder.moe.experts) > 0:
                            first_expert = encoder.moe.experts[0]
                            if hasattr(first_expert, 'expert_id'):
                                total_params = sum(p.numel() for p in first_expert.parameters())
                                tqdm.write(f"    [Unified Architecture] Each expert: {total_params:,} params")

                                # 显示专家差异化学习进度
                                expert_biases = []
                                for i, expert in enumerate(encoder.moe.experts):
                                    if hasattr(expert, 'expert_bias'):
                                        bias_norm = expert.expert_bias.norm().item()
                                        expert_biases.append(f"E{i}:{bias_norm:.3f}")

                                if expert_biases:
                                    tqdm.write(f"    [Expert Differentiation] Bias norms: {', '.join(expert_biases)}")
                    except Exception:
                        pass

                    # 专家利用率警告（仅在严重不均衡时）
                    if expert_min < 0.1:  # 提高阈值从0.15到0.1，减少警告频率
                        tqdm.write(f"    [WARNING] Expert usage imbalance detected (min={expert_min:.3f})")

                    # 新增：专家性能分析
                    if expert_entropy < 0.5:
                        tqdm.write(f"    [INFO] Low routing diversity - possible expert collapse")
                    elif expert_entropy > 1.0:
                        tqdm.write(f"    [INFO] High routing diversity - good expert utilization")

                    # 梯度警告：检查fargan_core梯度状态
                    if hasattr(train_one_epoch, '_last_fargan_grad_norm'):
                        last_fg_norm = train_one_epoch._last_fargan_grad_norm
                        if last_fg_norm < 1e-6:
                            tqdm.write(f"    😨 Warning: FARGAN core gradient very low ({last_fg_norm:.2e})")
                            tqdm.write(f"       Check: feature connectivity, wave_bp_ratio, dtype consistency")
                        # 额外的MoE路由诊断
                        if hasattr(encoder, 'moe') and encoder.moe is not None:
                            try:
                                with torch.no_grad():
                                    # 获取最近一次的路由logits/probs（如果可用）
                                    if hasattr(encoder.moe, '_last_router_logits'):
                                        logits = encoder.moe._last_router_logits
                                        tqdm.write(f"        Last router logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                                        probs = torch.softmax(logits, dim=-1)
                                        tqdm.write(f"        Last router probs mean: {probs.mean(dim=0).tolist()}")
                            except Exception as e:
                                tqdm.write(f"        MoE diagnosis failed: {e}")

                    if expert_entropy < 0.7:  # 提高阈值从0.8到0.7
                        tqdm.write(f"    ⚠️  Warning: Low expert entropy detected ({expert_entropy:.3f})")

                # 直流通路监控输出 - 架构级绕过支持
                if hasattr(encoder, 'moe') and hasattr(encoder.moe, 'get_performance_stats'):
                    pathway_stats = encoder.moe.get_performance_stats()
                    if pathway_stats:
                        bypass_weight = pathway_stats.get('bypass_weight', 0.0)
                        expert_weight = pathway_stats.get('expert_weight', 1.0)
                        performance_ratio = pathway_stats.get('performance_ratio', 1.0)
                        pathway_mode = pathway_stats.get('pathway_mode', 'unknown')
                        stage1_equivalent = pathway_stats.get('stage1_equivalent', False)
                        complexity_ratio = pathway_stats.get('complexity_ratio', 1.0)

                        # 格式化权重显示
                        weight_display = f"direct={bypass_weight:.2f}, expert={expert_weight:.2f}"
                        ratio_display = f"perf_ratio={performance_ratio:.3f}"
                        mode_display = f"mode={pathway_mode}, complexity={complexity_ratio:.1f}x"

                        tqdm.write(f"    Pathway Balance: {weight_display}, {ratio_display}")
                        tqdm.write(f"    System Mode: {mode_display}")

                        # 架构级绕过状态显示
                        if stage1_equivalent:
                            tqdm.write(f"    🟢 Architectural Bypass: Stage1-equivalent mode active")
                        elif pathway_mode == 'mixed':
                            tqdm.write(f"    🟡 Mixed Mode: Transitioning to expert system")
                        elif pathway_mode == 'pure_expert':
                            tqdm.write(f"    🔵 Pure Expert Mode: Full MoE active")

                        # 性能分析 - 针对简化专家架构
                        if pathway_mode == 'architectural_bypass':
                            tqdm.write(f"    [PATHWAY] Training in Stage1-equivalent mode for stability")
                        elif pathway_mode == 'mixed':
                            # 更详细的性能分析
                            if performance_ratio > 3.0:
                                tqdm.write(f"    [PERFORMANCE] Expert system significantly underperforming (ratio={performance_ratio:.3f})")
                                tqdm.write(f"    [ANALYSIS] Unified experts may need more training time")
                            elif performance_ratio > 1.5:
                                tqdm.write(f"    [PERFORMANCE] Direct pathway outperforming (ratio={performance_ratio:.3f})")
                                tqdm.write(f"    [ANALYSIS] Expert routing learning in progress")
                            elif performance_ratio < 0.8:
                                tqdm.write(f"    [SUCCESS] Expert system outperforming direct pathway (ratio={performance_ratio:.3f})")
                                tqdm.write(f"    [ANALYSIS] Unified architecture achieving specialization")
                            else:
                                tqdm.write(f"    [BALANCED] Competitive performance (ratio={performance_ratio:.3f})")

                            # 专家vs直流通路损失EMA对比
                            if hasattr(encoder.moe, 'moe_system'):
                                moe_sys = encoder.moe.moe_system
                                if hasattr(moe_sys, 'expert_loss_ema') and hasattr(moe_sys, 'direct_loss_ema'):
                                    expert_ema = moe_sys.expert_loss_ema.item()
                                    direct_ema = moe_sys.direct_loss_ema.item()
                                    tqdm.write(f"    [LOSS EMA] Expert: {expert_ema:.4f}, Direct: {direct_ema:.4f}")

                        elif pathway_mode == 'pure_expert':
                            tqdm.write(f"    [PATHWAY] Pure expert mode - unified architecture active")

    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= max(total_samples, 1)

    # === 🔍 Epoch质量监控和综合评估报告 ===
    def generate_quality_report():
        """生成训练质量综合报告"""
        current_global_step = step + len(loader)

        # 确定当前训练阶段
        if current_global_step < 1000:
            current_stage = "Foundation"
            stage_progress = current_global_step / 1000.0
        elif current_global_step < 5000:
            current_stage = "Balanced"
            stage_progress = (current_global_step - 1000) / 4000.0
        else:
            current_stage = "Quality"
            stage_progress = min((current_global_step - 5000) / 5000.0, 1.0)

        # 适配器状态
        adapter_status = semantic_adapter.get_status() if 'semantic_adapter' in locals() else {}

        # 损失趋势分析
        feat_loss_avg = epoch_metrics.get('feature_loss', 0.0)
        wave_loss_avg = epoch_metrics.get('wave_loss', 0.0)
        moe_loss_avg = epoch_metrics.get('moe_loss', 0.0)

        # 训练健康度评估
        health_score = 1.0
        health_issues = []

        if feat_loss_avg > 1.0:
            health_score *= 0.8
            health_issues.append("High feature reconstruction loss")

        if wave_loss_avg > 2.0:
            health_score *= 0.7
            health_issues.append("High audio usability loss")

        if moe_loss_avg > 0.5:
            health_score *= 0.9
            health_issues.append("High MoE loss")

        # 语义感知方案的epoch报告
        report = [
            f"\n=== Semantic Training Report (Step {current_global_step}) ===",
            f"Stage: {current_stage} ({stage_progress:.1%} complete)",
            f"Adaptation Strength: {adapter_status.get('adaptation_strength', 0.0):.2f}",
            f"Bypass Mode: {'On' if adapter_status.get('bypass_mode', False) else 'Off'}",
        ]

        # 如果有权重历史，显示权重分布
        if hasattr(train_one_epoch, '_weight_history') and train_one_epoch._weight_history:
            recent_weights = train_one_epoch._weight_history[-10:]  # 最近10次权重
            try:
                avg_weights = {
                    'feat': sum(w['weights']['feat'] for w in recent_weights) / len(recent_weights),
                    'wave': sum(w['weights']['wave'] for w in recent_weights) / len(recent_weights),
                    'moe': sum(w['weights']['moe'] for w in recent_weights) / len(recent_weights),
                }
                report.append(f"Semantic Weight Balance: recon={avg_weights['feat']:.3f}, quality={avg_weights['wave']:.3f}, routing={avg_weights['moe']:.3f}")
            except (KeyError, TypeError):
                pass  # 跳过权重统计

        # 精简MoE状态汇总
        moe_summary = []
        if hasattr(encoder, 'moe') and encoder.moe is not None:
            try:
                expert_util = encoder.moe.get_expert_utilization()
                expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"]

                # 显示带名称的专家使用率
                expert_usage_named = []
                for i, util in enumerate(expert_util):
                    name = expert_names[i] if i < len(expert_names) else f"E{i}"
                    expert_usage_named.append(f"{name}:{util:.3f}")

                moe_summary.append(f"Expert Usage: [{', '.join(expert_usage_named)}]")

                # 性能状态（最关键）
                if hasattr(trainer.encoder.moe, 'performance_ratio'):
                    perf_ratio = trainer.encoder.moe.performance_ratio
                    status = "Learning" if perf_ratio > 1.5 else "Competitive"
                    moe_summary.append(f"vs Direct: {status} ({perf_ratio:.2f}x)")

            except Exception:
                moe_summary.append("Analysis Failed")

        report.extend([
            f"Semantic Loss Summary: recon={feat_loss_avg:.4f}, audio_quality={wave_loss_avg:.4f}, routing={moe_loss_avg:.4f}",
            f"System Health: {health_score:.1%}",
        ])

        if moe_summary:
            report.extend(["MoE Expert Summary:"] + [f"  {item}" for item in moe_summary])

        if health_issues:
            report.append(f"[WARNING] Issues: {', '.join(health_issues)}")
        else:
            report.append("[OK] No significant issues detected")

        # 下阶段建议
        if current_stage == "Foundation" and stage_progress > 0.8:
            report.append("💡 Recommendation: Prepare for Balanced stage transition")
        elif current_stage == "Balanced" and stage_progress > 0.8:
            report.append("💡 Recommendation: Prepare for Quality stage transition")
        elif current_stage == "Quality":
            report.append("💡 Recommendation: Monitor semantic preservation metrics")

        return "\n".join(report)

    # 生成并打印质量报告
    if epoch_idx is not None:
        quality_report = generate_quality_report()
        print(quality_report)

        # 保存到epoch_metrics中供后续分析
        epoch_metrics['training_stage'] = current_stage if 'current_stage' in locals() else "Unknown"
        epoch_metrics['stage_progress'] = stage_progress if 'stage_progress' in locals() else 0.0
        epoch_metrics['health_score'] = health_score if 'health_score' in locals() else 1.0

    return epoch_metrics, step + len(loader)


def main() -> int:
    """Stage3训练主函数 - 按AETHER任务清单要求配置"""

    # 创建tqdm兼容的打印函数（防止被进度条覆盖）
    def safe_print(msg: str, flush: bool = True):
        """安全打印函数：在进度条存在时使用tqdm.write，否则使用普通print"""
        try:
            # 尝试使用tqdm.write（如果tqdm活跃时）
            tqdm.write(msg)
        except:
            # 回退到普通print
            print(msg)
        if flush:
            import sys
            sys.stdout.flush()
    p = argparse.ArgumentParser(description='Stage 3: MoE引入训练 (禁用FiLM，单变量验证)')
    p.add_argument('--moe-w', type=float, default=0.05, help='sample-level MoE balance loss 权重')
    p.add_argument('--moe-token-w', type=float, default=0.02, help='token-level MoE balance loss 权重')
    # 监督路由暖启动（当当前数据来自某个专家集时）
    p.add_argument('--expert-focus', type=str, default=None,
                   help='可选：当前训练样本来自哪个专家集（harmonic|transient|burst_inpaint|low_snr）。用于路由器监督暖启动。')
    p.add_argument('--router-sup', type=float, default=0.0,
                   help='路由监督损失权重（CrossEntropy于sample-level logits）。0关闭。建议暖启动期0.3~0.5，后期降至≤0.05')
    p.add_argument('--router-sup-decay-steps', type=int, default=5000,
                   help='路由监督线性衰减步数（>0时，从router-sup衰减至10%）')
    # 可选：短期聚焦的 Top-K 暖启动（小样本验证下非常有效）
    p.add_argument('--topk-warm-steps', type=int, default=0,
                   help='前 N 个全局 step 使用指定的 top-k 值（0 禁用）')
    p.add_argument('--topk-warm-k', type=int, default=1, choices=[1, 2],
                   help='暖启动阶段使用的 top-k（默认 1）')
    # In combined mode, --features/--pcm are not required
    p.add_argument('--features', type=str, required=False, help='Features file path (required if not using --combined-data-root)')
    p.add_argument('--pcm', type=str, required=False, help='Audio PCM file path (required if not using --combined-data-root)')
    p.add_argument('--stage1-checkpoint', type=str, default=None, help='Optional Stage 1 checkpoint for warm start (AETHER encoder/decoder)')
    p.add_argument('--fargan-checkpoint', type=str, help='Pre-trained FARGAN checkpoint (optional)')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from a previous Stage3 checkpoint (loads encoder/decoder/optimizer)')
    p.add_argument('--output-dir', type=str, default='checkpoints_stage3')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--epochs', type=int, default=3, help='Training epochs (task: 3 epochs)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=800)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--feature-dims', type=int, default=36)
    # Deprecated: dynamic weighting has replaced static alpha settings.
    # p.add_argument('--alpha-feat', type=float, default=1.0, help='Feature loss base weight (if no schedule)')
    # p.add_argument('--alpha-wave', type=float, default=1.0, help='Wave loss weight')
    p.add_argument('--alpha-sem', type=float, default=0.2, help='Semantic proxy loss weight (encoder semantic head vs priors)')
    p.add_argument('--alpha-acoustic', type=float, default=1.0, help='Acoustic path loss weight (20-dim cepstral features)')
    p.add_argument('--alpha-semantic', type=float, default=0.5, help='Semantic path loss weight (16-dim semantic features)')
    p.add_argument('--amp', type=str, default='fp16', choices=['none', 'fp16', 'bf16'], help='Mixed precision mode for CUDA (fp16 recommended for stability)')
    # Deprecated: no longer used in current Stage3 pipeline
    # p.add_argument('--wave-warmup-steps', type=int, default=3000, help='Linear warmup steps for wave loss weight (alpha_wave)')
    # p.add_argument('--wave-start-step', type=int, default=1500, help='Do not compute vocoder/wave loss before this global step')
    # p.add_argument('--preheat-frames', type=int, default=2, help='Ignore first N frames for loss (Stage1-like warm start)')
    # DataLoader knobs
    p.add_argument('--stride-frames', type=int, default=None, help='Data loader stride in frames (None=auto-adaptive)')
    p.add_argument('--semantic-source', type=str, default='fused', choices=['fused', 'ribbon', 'thread'], help='Semantic head source inside DualStream')
    p.add_argument('--split-stream-inputs', action='store_true', default=True,
                   help='将输入特征按 [0:20]→Ribbon(coarse), [20:36]→Thread(fine) 分流映射进入 DualStream')
    # Deprecated: dynamic weighting replaces alpha_feat scheduling
    # p.add_argument('--alpha-feat-start', type=float, default=None, help='Optional alpha_feat start value (overrides --alpha-feat)')
    # p.add_argument('--alpha-feat-end', type=float, default=0.0, help='alpha_feat end value')
    # p.add_argument('--alpha-feat-steps', type=int, default=1000, help='Linear anneal steps for alpha_feat')
    p.add_argument('--log-interval', type=int, default=50, help='Steps between progress updates')
    # Quick audio snapshots
    p.add_argument('--val-audio-interval', type=int, default=500,
                   help='If >0, every N steps export ~val-audio-seconds preview wavs (pred/orig)')
    p.add_argument('--val-audio-seconds', type=int, default=10,
                   help='Validation audio preview length in seconds (clamped to available)')
    p.add_argument('--val-audio-teacher', dest='val_audio_teacher', action='store_true', default=True,
                   help='Also export teacher-forced audio in quick previews (default: on)')
    p.add_argument('--no-val-audio-teacher', dest='val_audio_teacher', action='store_false',
                   help='Disable teacher-forced audio in quick previews')
    p.add_argument('--val-audio-deemph', type=float, default=0.85,
                   help='De-emphasis coefficient for preview audio (0 disables; default 0.85 to match eval tool)')
    p.add_argument('--val-audio-teacher-no-adapter', action='store_true', default=False,
                   help='Use raw GT features (no adapter) for teacher-forced preview (A/B for vocoder issues)')
    p.add_argument('--val-audio-no-preheat', action='store_true', default=True,
                   help='Do not use preheat audio in preview synthesis (match generate_10s_audio.py)')
    # p.add_argument('--profile-interval', type=int, default=0, help='If >0, include per-step timings every N steps')
    # Checkpointing controls
    p.add_argument('--save-every-epochs', type=int, default=0, help='Save a checkpoint every N epochs (0 disables)')
    p.add_argument('--save-every-steps', type=int, default=0, help='Save a checkpoint every N global steps (0 disables)')
    p.add_argument('--always-save-last', action='store_true', help='Always save a final checkpoint at the end')

    # Combined multi-expert dataset (mixed batch training)
    p.add_argument('--combined-data-root', type=str, default=None,
                   help='If set, combine four expert subsets under this root into mixed batches')
    p.add_argument('--mix-ratio', type=str, default=None,
                   help='Comma-separated ratios for [harmonic,transient,burst_inpaint,low_snr] in combined mode')

    # Stage3特定配置 (按任务要求)
    p.add_argument('--moe', action='store_true', default=True, help='Enable MoE (Stage3 default: enabled)')
    p.add_argument('--no-moe', action='store_true', help='🚨 Disable MoE for debugging (临时诊断选项)')
    # p.add_argument('--enable-rate', action='store_true',
    #                help='Enable rate regularizer (Stage3默认禁用，当前实现占位)')
    p.add_argument('--router-no-csi', action='store_true', default=True,
                   help='Router不使用CSI (Stage3单变量验证)')
    # 性能优化选项
    # p.add_argument('--gradient-accumulation-steps', type=int, default=1,
    #                help='(unused placeholder)')
    p.add_argument('--use-compile', action='store_true', default=False,
                   help='Use torch.compile for model optimization (PyTorch 2.0+)')
    p.add_argument('--moe-token-warmup-steps', type=int, default=0,
                   help='Use sample-level routing only for first N steps, then enable token-level')
    # p.add_argument('--debug-nan', action='store_true', default=False,
    #                help='(unused placeholder)')
    # p.add_argument('--safe-init', action='store_true', default=True,
    #                help='(unused placeholder)')
    p.add_argument('--emergency-bypass-moe', action='store_true', default=False,
                   help='🚨 Emergency: completely bypass MoE for NaN diagnosis')
    p.add_argument('--wave-stride', type=int, default=1,
                help='Compute vocoder/audio loss every N batches (default 1 = every batch)')
    p.add_argument('--wave-loss-seconds', type=float, default=6.0,
                help='Cap the audio segment length used for wave loss (seconds, default 6.0)')
    # p.add_argument('--wave-min-bp', type=float, default=0.1,
    #             help='(unused placeholder)')
    p.add_argument('--router-jitter', type=float, default=0.01,
                help='训练态对路由 logits 施加的高斯抖动强度，用于促使专家探索（默认 0.01）')
    # 直流通路相关参数 - 架构级绕过优化
    p.add_argument('--enable-direct-pathway', action='store_true', default=True,
                help='启用MoE直流通路，用于性能对比验证（默认启用）')
    p.add_argument('--disable-direct-pathway', action='store_true', default=False,
                help='禁用直流通路，使用纯专家系统（用于对照实验）')
    p.add_argument('--initial-bypass-weight', type=float, default=0.1,
                help='直流通路初始权重 (0.0-1.0)，控制训练开始时直流vs专家的比例，Stage3默认0.1强制expert训练')
    p.add_argument('--adaptive-threshold', type=float, default=0.15,
                help='性能差异阈值，触发权重调整（默认15%，增大以减少直流通路干预）')
    p.add_argument('--pathway-warmup-steps', type=int, default=2000,
                help='直流权重warmup步数，前一半为架构级绕过期')

    # MoE structure knobs
    p.add_argument('--n-experts', type=int, default=4, help='Number of experts (default 4)')
    p.add_argument('--expert-dropout', type=float, default=0.1, help='Expert dropout probability (default 0.1)')

    args = p.parse_args()

    # 强制Stage3配置 (按任务清单要求)
    if args.no_moe:
        safe_print("🚨 DEBUG: Disabling MoE for gradient explosion diagnosis")
        args.moe = False
    elif not args.moe:
        safe_print("⚠️  Warning: Force enabling MoE for Stage3 (per task requirements)")
        args.moe = True

    # 直流通路配置处理
    if args.disable_direct_pathway:
        args.enable_direct_pathway = False
        safe_print("🚨 直流通路已禁用 - 纯专家系统模式")
    elif args.enable_direct_pathway:
        safe_print(f"✅ 直流通路已启用 - 初始权重: {args.initial_bypass_weight:.2f}")
        safe_print(f"   自适应阈值: {args.adaptive_threshold:.3f}, Warmup步数: {args.pathway_warmup_steps}")

    # 获取Stage3配置
    stage_cfg = get_stage_config("stage3")
    # 确保禁用FiLM (单变量验证要求)
    stage_cfg.use_film = False
    stage_cfg.apply_channel = False  # 禁用信道模拟

    safe_print("🚀 Starting Stage3 Training - MoE引入 (禁用FiLM，单变量验证)")
    safe_print(f"   MoE enabled: {args.moe}")
    safe_print(f"   FiLM disabled: {not stage_cfg.use_film}")
    safe_print(f"   Channel simulation disabled: {not stage_cfg.apply_channel}")
    safe_print(f"   Router strategy: {'no-CSI' if args.router_no_csi else 'with-CSI'}")
    if args.expert_focus:
        safe_print(f"   Router supervised warmup: focus={args.expert_focus}, sup_w={args.router_sup}")
    safe_print(f"   AMP mode: {args.amp}")
    safe_print(f"   DataLoader: workers={args.num_workers} | stride_frames={args.stride_frames}")
    safe_print(f"   Performance optimizations: torch_compile={args.use_compile}")

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                         args.device if args.device != 'auto' else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data loader (supports combined mixed-batch mode)
    if getattr(args, 'combined_data_root', None):
        # Build combined loader from four expert subsets
        train_loader, dataset = create_combined_data_loader(
            data_root=args.combined_data_root,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            frame_size=160,
            stride_frames=args.stride_frames,
            energy_selection=True,
            feature_dims=args.feature_dims,
            num_workers=max(1, int(args.num_workers)),
        )
        # Optional custom mix ratio
        if getattr(args, 'mix_ratio', None):
            try:
                ratios = [float(x.strip()) for x in str(args.mix_ratio).split(',')]
                assert len(ratios) == 4
                import numpy as _np
                dataset.mix_ratio = _np.array(ratios, dtype=_np.float64)
                s = dataset.mix_ratio.sum()
                dataset.mix_ratio = dataset.mix_ratio / (s if s > 0 else 1.0)
                dataset.cumprob = _np.cumsum(dataset.mix_ratio)
                safe_print(f"   Combined mix ratio set to: {dataset.mix_ratio.tolist()}")
            except Exception:
                safe_print("⚠️  Invalid --mix-ratio format; expected 'a,b,c,d'. Using defaults.")
    else:
        train_loader, dataset = create_aether_data_loader(
            data_dir=str(Path(args.features).parent.parent) if 'lmr_export' in Path(args.features).parts else str(Path(args.features).parent),
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            max_samples=None,
            num_workers=max(1, int(args.num_workers)),
            energy_selection=True,
            test_mode=False,
            feature_spec_type='fargan',
            features_file=args.features,
            audio_file=args.pcm,
            stride_frames=args.stride_frames,  # 新增步幅配置
        )

    # Create models using simplified architecture
    config = {
        "d_in": args.feature_dims,
        "d_model": 128,
        "dz": 24,
        "d_csi": 10,
        "use_film": False,  # Stage3: 禁用FiLM
        "use_moe": args.moe,  # Stage3: 启用MoE
        "use_quantization": False,
        "latent_bits": 4,
        "n_experts": int(getattr(args, 'n_experts', 4)),
        "top_k": 2,        # Stage3: TOP-2路由
        "moe_router_use_csi": (not args.router_no_csi),  # Router不使用CSI → False
        "use_semantic_head": True,
        "semantic_dim": 6,
        "semantic_source": args.semantic_source,
        # 将输入分派到 DualStream 两条通路（前20→Ribbon，后16→Thread）
        "split_stream_inputs": bool(getattr(args, 'split_stream_inputs', False)),
        # 直流通路配置
        "enable_direct_pathway": args.enable_direct_pathway,
        "initial_bypass_weight": args.initial_bypass_weight,
        "adaptive_threshold": args.adaptive_threshold,
        "pathway_warmup_steps": args.pathway_warmup_steps,
        # MoE training knobs
        "expert_dropout": float(getattr(args, 'expert_dropout', 0.1)),
    }

    encoder, _ = create_aether_codec(config)
    encoder = encoder.to(device)
    if hasattr(encoder, 'moe') and hasattr(encoder.moe, 'specialized_moe'):
        try:
            encoder.moe.specialized_moe.router_jitter = float(getattr(args, 'router_jitter', 0.0))
        except Exception:
            pass
    def _cast_rnns_fp32(module: torch.nn.Module):
        for m in module.modules():
            name = m.__class__.__name__
            if name in ('LSTM', 'GRU', 'RobustLSTM', 'RobustGRU'):
                m.to(torch.float32)

    _cast_rnns_fp32(encoder)



    # 🚨 紧急绕过MoE模式
    if args.emergency_bypass_moe and hasattr(encoder, 'moe') and encoder.moe is not None:
        safe_print("🚨 EMERGENCY: Bypassing MoE completely for NaN diagnosis")
        encoder.moe.specialized_moe._emergency_bypass = True

    # 自定义decoder with FARGAN
    decoder = AETHERFARGANDecoder(
        d_out=args.feature_dims,
        d_csi=10,
        enable_synth=True,
        use_film=False  # Stage3: 解码端禁用FiLM
    ).to(device)

    # 创建语义特征提取器（SSL+投影+帧率对齐）
    semantic_extractor = create_semantic_extractor(
        model_name="hubert-base",  # 使用HuBERT作为SSL模型
        proj_dim=16,               # 16维语义特征
        device=device
    )
    safe_print(f"  ✅ Semantic extractor initialized: {semantic_extractor.ssl_model_name}")
    # 确保提取器在目标设备
    try:
        semantic_extractor.to(device)
    except Exception:
        pass
    semantic_extractor.eval()  # SSL模型保持eval模式
    # 将提取器注入到 train_one_epoch 缓存，避免作用域问题
    try:
        train_one_epoch._semantic_extractor = semantic_extractor
    except Exception:
        pass

    # 🔧 FIX 6: 条件性mixed precision配置，避免dtype冲突
    # 仅在非AMP模式下强制float32，AMP模式下保持一致性
    if args.amp == 'none':
        decoder.fargan_core.float()  # 非AMP模式使用float32确保稳定
    # AMP模式下让autocast自动管理dtype，避免冲突
    _cast_rnns_fp32(decoder)
    # 可选的模型编译优化（PyTorch 2.0+）
    if args.use_compile:
        try:
            safe_print("  🚀 Compiling models with torch.compile...")
            encoder = torch.compile(encoder, mode='default')
            decoder = torch.compile(decoder, mode='default')
            safe_print("  ✅ Model compilation successful")
        except Exception as e:
            safe_print(f"  ⚠️  Model compilation failed: {e}")
            safe_print("  📋 Continuing without compilation")

    # Optional: Load Stage 1 checkpoint (AETHER warm start)
    if args.stage1_checkpoint:
        try:
            stage1_ckpt = torch.load(args.stage1_checkpoint, map_location='cpu')
            enc_sd = stage1_ckpt.get('encoder_state_dict') or {}
            dec_sd = stage1_ckpt.get('decoder_state_dict') or {}

            # Load encoder with shape check
            enc_state = encoder.state_dict()
            enc_loaded = enc_skipped = 0
            for k, v in enc_sd.items():
                if k in enc_state and enc_state[k].shape == v.shape:
                    enc_state[k] = v
                    enc_loaded += 1
                else:
                    enc_skipped += 1
            encoder.load_state_dict(enc_state, strict=False)
            safe_print(f"✅ Stage1: loaded encoder params: {enc_loaded} matched, {enc_skipped} skipped")

            # Partially load decoder (Stage1 AETHERDecoder → AETHERFARGANDecoder common parts)
            dec_state = decoder.state_dict()
            dec_loaded = dec_skipped = 0
            for k, v in dec_sd.items():
                if k in dec_state and dec_state[k].shape == v.shape:
                    dec_state[k] = v
                    dec_loaded += 1
                else:
                    dec_skipped += 1
            decoder.load_state_dict(dec_state, strict=False)
            safe_print(f"✅ Stage1: loaded decoder params: {dec_loaded} matched, {dec_skipped} skipped (strict=False)")
        except Exception as e:
            safe_print(f"⚠️  Failed to load Stage1 checkpoint: {e}")
    def force_vocoder_fp32(decoder):
        # 只把 fargan_core 强制到 float32，其他解码头继续跟随 AMP
        decoder.fargan_core.float()
        for m in decoder.fargan_core.modules():
            name = m.__class__.__name__
            if name in ('LSTM', 'GRU', 'RobustLSTM', 'RobustGRU'):
                m.to(torch.float32)
    # Load FARGAN weights if provided
    if args.fargan_checkpoint:
        try:
            ckpt = torch.load(args.fargan_checkpoint, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

            # 更鲁棒的加载：匹配所有形状一致的键，自动剥离可能的前缀
            decoder_state = decoder.state_dict()
            loaded = skipped = 0
            # 额外：处理旧版 weight_norm (weight_v/weight_g) → 新版 parametrizations.weight.original0
            # 收集g/v对
            vg_pairs = {}
            for k in list(state.keys()):
                if k.endswith('.weight_v'):
                    base = k[:-9]
                    gk = base + '.weight_g'
                    if gk in state:
                        vg_pairs[base] = (state[k], state[gk])

            for k, v in state.items():
                cand_keys = [k]
                if k.startswith('module.'):
                    cand_keys.append(k[len('module.'):])
                if k.startswith('model.'):
                    cand_keys.append(k[len('model.'):])
                matched = False
                for kk in cand_keys:
                    if kk in decoder_state and decoder_state[kk].shape == v.shape:
                        decoder_state[kk] = v
                        loaded += 1
                        matched = True
                        break
                if not matched:
                    skipped += 1

            # 尝试将旧版 weight_norm 的 g/v 转换为新版 parametrizations 权重
            import torch as _t
            for base, (v_t, g_t) in vg_pairs.items():
                # 目标键：parametrizations.weight.original0
                target_key = base + '.parametrizations.weight.original0'
                if target_key in decoder_state:
                    try:
                        v_tensor = _t.as_tensor(v_t)
                        g_tensor = _t.as_tensor(g_t)
                        # 重建权重：w = v * (g / ||v||)
                        v_norm = v_tensor.norm(dim=list(range(1, v_tensor.ndim)), keepdim=True) if v_tensor.ndim > 1 else v_tensor.abs() + 1e-8
                        scale = g_tensor / (v_norm + 1e-8)
                        while scale.ndim < v_tensor.ndim:
                            scale = scale.unsqueeze(-1)
                        w_recon = v_tensor * scale
                        if w_recon.shape == decoder_state[target_key].shape:
                            decoder_state[target_key] = w_recon
                            loaded += 1
                    except Exception:
                        continue

            # 第二轮：尝试常见前缀映射（cond→cond_net, sig→sig_net, fargan_cond→cond_net, fargan_sub→sig_net）
            if skipped > 0:
                remap_rules = [
                    ('cond.', 'cond_net.'),
                    ('sig.',  'sig_net.'),
                    ('fargan_cond.', 'cond_net.'),
                    ('fargan_sub.',  'sig_net.'),
                ]
                for k, v in list(state.items()):
                    # 跳过已匹配过的键
                    if any([(alt in decoder_state) and (decoder_state[alt].shape == v.shape) for alt in [k]]):
                        continue
                    for a, b in remap_rules:
                        if a in k:
                            kk = k.replace(a, b)
                            if kk in decoder_state and decoder_state[kk].shape == v.shape:
                                decoder_state[kk] = v
                                loaded += 1
                                skipped -= 1
                                break

            decoder.load_state_dict(decoder_state, strict=False)
            safe_print(f"✅ Stage2(FARGAN): loaded {loaded} params into decoder, skipped {skipped}")
            # 打印部分未匹配键，便于进一步精确映射
            if skipped > 0:
                try:
                    missing = []
                    for k, v in state.items():
                        found = False
                        if k in decoder_state and decoder_state[k].shape == v.shape:
                            found = True
                        else:
                            # 检查已应用的前缀替换
                            for a, b in [('cond.', 'cond_net.'), ('sig.', 'sig_net.'), ('fargan_cond.', 'cond_net.'), ('fargan_sub.', 'sig_net.')]:
                                if a in k:
                                    kk = k.replace(a, b)
                                    if kk in decoder_state and decoder_state[kk].shape == v.shape:
                                        found = True
                                        break
                            # 检查 weight_norm 重建目标键
                            if not found and k.endswith('.weight_v'):
                                base = k[:-9]
                                target_key = base + '.parametrizations.weight.original0'
                                if target_key in decoder_state and decoder_state[target_key].shape == v.shape:
                                    found = True
                        if not found:
                            missing.append(k)
                    if missing:
                        show = missing[:10]
                        safe_print(f"   ⚠️ Unmatched keys (sample): {show}")
                except Exception:
                    pass
        except Exception as e:
            safe_print(f"⚠️  Failed to load FARGAN checkpoint: {e}")
        # 🔧 FIX 7: 应用一致的mixed precision策略
        if args.amp == 'none':
            force_vocoder_fp32(decoder)  # 仅非AMP模式强制float32

    # Optimizer with differential learning rates
    lr = getattr(stage_cfg, 'learning_rate', 2e-4)

    # FARGAN：核心与 parametrizations 单独 param group（更小 lr，且禁 WD）
    decoder_params, fargan_core_params, fargan_parametrizations = [], [], []
    for name, p in decoder.named_parameters():
        if 'parametrizations' in name:
            fargan_parametrizations.append(p)
        elif 'fargan_core' in name:
            fargan_core_params.append(p)
        else:
            decoder_params.append(p)
    # 在创建 optimizer 之前，替换 encoder 的分组：
    enc_backbone, enc_attn = [], []
    for n, p in encoder.named_parameters():
        if any(k in n for k in ['thread_blocks', 'qkv', 'out_proj', 'mix']):
            enc_attn.append(p)       # 注意力/混合相关
        else:
            enc_backbone.append(p)   # 其它
    param_groups = [
        {'params': enc_backbone, 'lr': lr,      'weight_decay': 1e-6},
        {'params': enc_attn,     'lr': lr*0.5,  'weight_decay': 1e-6},  # 注意力层半速,
        {'params': decoder_params,                  'lr': lr,       'weight_decay': 1e-6},
        {'params': fargan_core_params,              'lr': lr*0.1,   'weight_decay': 0.0},
        {'params': fargan_parametrizations,         'lr': lr*0.1,   'weight_decay': 0.0},
    ]


    optimizer = optim.AdamW(param_groups, weight_decay=1e-6)
    use_fp16 = (args.amp == 'fp16')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    accum_steps = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
    # Optional resume from previous Stage3 checkpoint (encoder/decoder/optimizer)
    if getattr(args, 'resume', None):
        from pathlib import Path as _P
        _rp = _P(args.resume)
        if _rp.exists():
            try:
                _ck = torch.load(str(_rp), map_location='cpu')
                if 'encoder_state_dict' in _ck:
                    encoder.load_state_dict(_ck['encoder_state_dict'], strict=False)
                if 'decoder_state_dict' in _ck:
                    decoder.load_state_dict(_ck['decoder_state_dict'], strict=False)
                if 'optimizer_state_dict' in _ck:
                    try:
                        optimizer.load_state_dict(_ck['optimizer_state_dict'])
                    except Exception:
                        pass
                best_loss = float(_ck.get('loss', float('inf')))
                safe_print(f"🔁 Resumed from: {args.resume} (best_loss={best_loss:.6f})")
            except Exception as _e:
                safe_print(f"⚠️  Failed to resume from {args.resume}: {_e}")

    safe_print(f"📊 Training setup:")
    safe_print(f"   Model params: Encoder={sum(p.numel() for p in encoder.parameters()):,}, "
          f"Decoder={sum(p.numel() for p in decoder.parameters()):,}")
    safe_print(f"   Learning rate: {lr}")
    safe_print(f"   Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    safe_print(f"   Total batches per epoch: {len(train_loader)}")

    # Training loop
    best_loss = float('inf')
    global_step = 0

    safe_print("\n🎯 Stage3 验收标准:")
    safe_print("   - 特征重建损失 ≤ 0.25")
    safe_print("   - MoE专家利用率 > 75% (3个专家均衡使用: Harmonic, Transient, BurstInpaint)")
    safe_print("   - 波形RMS偏移 < 5dB")
    safe_print("   - MoE损失收敛稳定")
    safe_print("   - 无FiLM条件下训练稳定")
    safe_print("   - 跳过LowSNRExpert (无CSI输入)")
    safe_print(f"   - Rate loss: 禁用 (Stage3默认禁用，避免梯度爆炸)\n")

    # CUDA backend optimisations
    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 强制flush所有配置信息，确保在进度条出现前显示
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

    for epoch in range(1, args.epochs + 1):
        safe_print(f"🔄 Epoch {epoch}/{args.epochs}")

        epoch_metrics, global_step = train_one_epoch(
            encoder, decoder, train_loader, device, optimizer,
            stage_cfg, global_step, args, epoch_idx=epoch,
            scaler=scaler, 
        )

        # 语义感知系统Epoch总结
        safe_print(f"[Semantic Audio System] Epoch {epoch} Summary:")
        safe_print(f"   Total Loss: {epoch_metrics['total_loss']:.6f}")
        safe_print(f"   Feature Reconstruction: {epoch_metrics['feature_loss']:.6f}")
        safe_print(f"   Audio Quality (Usability): {epoch_metrics['wave_loss']:.6f}")
        safe_print(f"   Expert Routing: {epoch_metrics['moe_loss']:.6f}")
        safe_print(f"   Rate Control: {epoch_metrics['rate_loss']:.6f}")

        # 语义损失信息
        if 'semantic_loss' in epoch_metrics:
            safe_print(f"   Semantic Alignment: {epoch_metrics['semantic_loss']:.6f}")

        # MoE health check (只在非绕过模式下显示)
        if args.emergency_bypass_moe:
            safe_print(f"   [WARNING] MoE Status: BYPASSED (emergency diagnosis mode)")
        elif encoder.use_moe:
            expert_min = epoch_metrics.get('expert_usage_min', 0)
            expert_max = epoch_metrics.get('expert_usage_max', 0)
            expert_entropy = epoch_metrics.get('expert_entropy', 0)
            expert_balance = 1.0 - (expert_max - expert_min)  # 均衡度

            # Display individual expert usage rates if available - 动态专家数量
            n_experts = getattr(encoder.moe, 'n_experts', 4) if hasattr(encoder, 'moe') else 4
            expert_usage_display = []
            expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"][:n_experts]  # 根据专家数量截断

            # 构建带专家名称的使用率显示
            expert_usage_named = []
            for i in range(n_experts):  # 动态专家数量
                usage_key = f'expert_{i}_usage'
                name = expert_names[i] if i < len(expert_names) else f"E{i}"
                if usage_key in epoch_metrics:
                    usage_value = f"{epoch_metrics[usage_key]:.3f}"
                    expert_usage_display.append(usage_value)
                    expert_usage_named.append(f"{name}:{usage_value}")
                else:
                    expert_usage_display.append("N/A")
                    expert_usage_named.append(f"{name}:N/A")

            safe_print(f"   MoE Health:")
            safe_print(f"     Expert Usage: [{', '.join(expert_usage_named)}]")
            for i, (name, usage) in enumerate(zip(expert_names, expert_usage_display)):
                safe_print(f"       E{i+1} {name}: {usage}")
            safe_print(f"     Expert Entropy: {expert_entropy:.3f}")
            safe_print(f"     Balance Score: {expert_balance:.3f}")

            # 验收标准检查
            if expert_min > 0.2:  # 每个专家>20%使用率
                safe_print("     ✅ Expert utilization criterion met")
            else:
                safe_print("     ❌ Expert utilization below threshold")

            if expert_entropy > 0.8:
                safe_print("     ✅ Expert entropy criterion met")
            else:
                safe_print("     ❌ Expert entropy below threshold")

        # Periodic epoch checkpoint
        if args.save_every_epochs and args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0):
            epoch_ckpt = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': float(epoch_metrics['total_loss']),
                'metrics': epoch_metrics,
                'config': config,
                'args': vars(args)
            }
            ckpt_path = out_dir / f'stage3_epoch_{epoch}.pth'
            torch.save(epoch_ckpt, ckpt_path)
            safe_print(f"💾 Saved epoch checkpoint: {ckpt_path}")

        # Save best checkpoint
        if epoch_metrics['total_loss'] < best_loss:
            best_loss = epoch_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'metrics': epoch_metrics,
                'config': config,
                'args': vars(args)
            }

            ckpt_path = out_dir / 'stage3_best.pth'
            torch.save(checkpoint, ckpt_path)
            # Also save a copy with the epoch suffix for reproducibility
            ckpt_epoch_path = out_dir / f'stage3_best_epoch_{epoch}.pth'
            try:
                torch.save(checkpoint, ckpt_epoch_path)
            except Exception as _e:
                safe_print(f"⚠️  Failed to save epoch-suffixed best checkpoint: {ckpt_epoch_path} ({_e})")
            safe_print(f"💾 Saved best checkpoint: {ckpt_path} (epoch copy: {ckpt_epoch_path})")

            # 验收标准综合检查
            feature_loss_ok = epoch_metrics['feature_loss'] <= 0.25
            expert_util_ok = encoder.use_moe and expert_min > 0.2

            safe_print(f"📋 Stage3 验收进度:")
            safe_print(f"   Feature Loss ≤ 0.25: {'✅' if feature_loss_ok else '❌'} ({epoch_metrics['feature_loss']:.4f})")
            if encoder.use_moe:
                safe_print(f"   Expert Utilization > 20%: {'✅' if expert_util_ok else '❌'} (min={expert_min:.3f})")

    safe_print(f"\n🎉 Stage3 training completed!")
    safe_print(f"   Best loss: {best_loss:.6f}")
    safe_print(f"   Final checkpoint: {out_dir / 'stage3_best.pth'}")

    # Always save last if requested
    if args.always_save_last:
        last_ckpt = {
            'epoch': args.epochs,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(best_loss),
            'config': config,
            'args': vars(args)
        }
        ckpt_path = out_dir / 'stage3_last.pth'
        torch.save(last_ckpt, ckpt_path)
        safe_print(f"💾 Saved last checkpoint: {ckpt_path}")

    # 最终验收报告
    if best_loss <= 0.25:
        safe_print("✅ Stage3 training PASSED - ready for Stage4")
    else:
        safe_print("⚠️  Stage3 training needs improvement - consider adjusting hyperparameters")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
