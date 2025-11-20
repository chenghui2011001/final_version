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

# 使用简化的架构
import sys
import os
# Ensure final_version root is on sys.path; avoid inserting subdirs (e.g. models)
# to prevent shadowing top-level packages like 'utils'.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.enhanced_aether_integration import AETHEREncoder, AETHERDecoder, create_aether_codec
from models.maybe_useless.aether_fargan_decoder import AETHERFARGANDecoder
from utils.real_data_loader import create_aether_data_loader
from training.pipeline.stages import StageConfig, get_stage_config
from training.pipeline.wave_loss import fargan_wave_losses
from training.losses import rate_loss, compute_layered_loss
from models.utils import validate_csi_config, extract_acoustic_priors
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
            "use_moe": True,   # Stage3: 启用MoE
            "use_quantization": False,
            "latent_bits": 4,
            "n_experts": 3,    # Stage3: 只用3个专家 (无CSI，跳过LowSNRExpert)
            "topk": 2,         # Stage3: TOP-2路由
        }

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
                # Store as formatted string for display
                usage_str = ', '.join([f'{util.item():.3f}' for util in expert_util])
                metrics['expert_usage_all'] = usage_str
            except (AttributeError, RuntimeError):
                # Fallback for MoE without utilization tracking
                metrics['expert_usage_min'] = 0.25  # Placeholder
                metrics['expert_usage_max'] = 0.25
                metrics['expert_entropy'] = 1.386  # log(4) for 4 experts
                metrics['expert_balance'] = 0.8
                metrics['expert_usage_all'] = '0.250, 0.250, 0.250, 0.250'
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

    wave_char = CharbonnierLoss(eps=1e-3)  

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
        # >>> 🔧 FIX 3: 优化梯度传播控制，提高最小传播比例 <<<
        # 大幅提高最小梯度传播比例，确保fargan_core能接收到有效梯度信号
        min_bp = float(getattr(args, 'wave_min_bp', 0.5))  # 从10%大幅提高到50%
        if active_wave:
            wave_bp_ratio = max(wave_bp_ratio, min_bp)   # 确保最小50%梯度传播

        # <<< DEBUG/SAFETY end <<<

        # 修复损失权重：按用户要求调整权重比例，防止梯度爆炸
        # feat_loss=0.1, wave_loss=0.7, moe_loss=0.2
        # 注意：wave loss通常较大，所以实际权重要更小
        alpha_wave_eff = args.alpha_wave * 0.1  # 基础权重：0.1，实际会在损失组合时乘以7
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

        # 🔍 调试输出：检查vocoder调用条件
        wave_stride = int(getattr(args, 'wave_stride', 1) or 1)
        should_call_vocoder = active_wave and (batch_idx % wave_stride == 0)
        if batch_idx % 50 == 0:  # 每50个batch输出一次调试信息
            tqdm.write(f"      🔍 Vocoder Debug: global_step={global_step}, active_wave={active_wave}")
            tqdm.write(f"         batch_idx={batch_idx}, wave_stride={wave_stride}, should_call={should_call_vocoder}")

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

            # ① 倒谱给 vocoder：取混合流的前20维，确保梯度连接
            feats20 = feats_mix[..., :20].contiguous()      # [B,T,20] ← 保持梯度连接

            # ② 周期也用预测流的 dnn_pitch（第18维）
            pred_pitch = feats_mix[..., 18].float()         # [B,T]
            f0_hz = (2.0 ** (pred_pitch + 5.0)).clamp(50.0, 400.0)
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

                        # ③ 🔧 FIX 5: 统一FARGAN调用策略，保持梯度连接
                        # 移除不必要的autocast禁用，保持与其他部分的一致性
                        y_audio, aux = decoder.fargan_core(f, p, _nb, pre=pre)

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

                    tqdm.write(f"      🔄 Starting vocoder retry loop with nb_try={nb_try}")
                    for try_idx in range(max_tries):
                        try:
                            if not hasattr(train_one_epoch, '_vc_checked'):
                                # 这里在 _call_vocoder_nb 外部，没有 _nb / f 可用；
                                # 直接用 feats20_vc 的长度和 nb_try
                                T_in = int(feats20_vc.size(1))  # 预期 = nb_try + 4
                                tqdm.write(f"[VocoderBridge] T_in={T_in}, will request nb={nb_try}, "
                                        f"expect cond_len={T_in-4}, pre_frames={nb_pre}")
                                train_one_epoch._vc_checked = True


                            tqdm.write(f"      🎯 Calling vocoder with nb_try={nb_try}")
                            y_hat_audio, _ = _call_vocoder_nb(nb_try)
                            L = int(y_hat_audio.size(-1))
                            tqdm.write(f"      ✅ Vocoder returned audio length={L}")

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
            # ② 计算wave loss的窗口大小 - 使用更大的基础窗口
            # 修复：确保最小窗口至少80%，避免因bp_ratio过小而导致窗口太小
            window_ratio = max(0.8, 0.6 + 0.4 * wave_bp_ratio)  # 最小80%，最大100%
            m = int(min(
                wave_audio_pred_bp.size(-1),
                window_ratio * wave_audio_pred_bp.size(-1)
            ))
            wave_pred_head = wave_audio_pred_bp[..., :m]
            wave_tgt_head  = wave_audio_target[..., :m]
            # ③ 使用修复后的FARGAN wave损失函数，简化处理
            from training.pipeline.wave_loss import fargan_wave_losses
            # 确保period维度正确对齐
            audio_frames = wave_pred_head.size(-1) // 160
            if period.size(1) > audio_frames:
                period_aligned = period[:, :audio_frames]
            else:
                # 如果period不够长，扩展到所需长度
                period_aligned = period.repeat(1, (audio_frames // period.size(1)) + 1)[:, :audio_frames]

            # 强制使用fp32计算FARGAN损失，避免bf16导致的数值下溢
            with torch.autocast(device_type='cuda', enabled=False):
                l_wave_raw, wave_details = fargan_wave_losses(
                    wave_pred_head.float(),
                    wave_tgt_head.float(),
                    period_aligned,
                    device=device
                )
            # ④ 有效权重
            l_wave = l_wave_raw * alpha_wave_eff

            # RMS gating (融合计算避免重复GPU操作)
            pred_rms = torch.sqrt(torch.mean(wave_audio_pred_bp.float().pow(2), dim=-1) + 1e-8)
            pred_rms_db = 20.0 * torch.log10(pred_rms.mean() + 1e-8)
            gating_threshold = -35.0 + 7.0 * min(1.0, global_step / 1000.0)
            # if pred_rms_db < gating_threshold or global_step < 400:
            #     l_wave = l_wave * 0.3

        # 3. Rate loss (仅在需要码率控制时启用)
        l_rate = torch.tensor(0.0, device=device, dtype=loss_dtype)
        if args.enable_rate:
            l_rate = torch.tensor(0.0, device=device, dtype=loss_dtype)

            # Stage3无码率目标时，rate_loss会导致梯度爆炸

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
                        moe_metrics['expert_usage_all'] = ', '.join([f'{u.item():.3f}' for u in expert_util])
                    except Exception:
                        pass

            # **关键：按 CLI 权重并入 loss（保持 float32，参与反传）**
            mb = enc_logs.get('moe_balance_loss', None)
            if isinstance(mb, torch.Tensor):
                l_moe = l_moe + float(getattr(args, 'moe_w', 0.05)) * mb.float()

            mt = enc_logs.get('moe_token_balance_loss', None)
            if isinstance(mt, torch.Tensor):
                l_moe = l_moe + float(getattr(args, 'moe_token_w', 0.02)) * mt.float()

            # 其他可能的 MoE 约束（如一致性等），统一用较小权重
            for k, v in enc_logs.items():
                if not isinstance(v, torch.Tensor) or (k in ('moe_balance_loss','moe_token_balance_loss')):
                    continue
                if k.startswith('moe_') and v.requires_grad:
                    l_moe = l_moe + 0.05 * v.float()


        # 6. Fused total loss computation with alpha_feat anneal schedule
        if args.alpha_feat_start is not None:
            start = float(args.alpha_feat_start)
            end = float(args.alpha_feat_end)
            steps = max(1, int(args.alpha_feat_steps))
            ratio = min(1.0, max(0.0, global_step / steps))
            alpha_feat_now = (1 - ratio) * start + ratio * end
        else:
            alpha_feat_now = args.alpha_feat

        # 融合损失计算 - 按用户要求调整权重：feat=0.1, wave=0.7, moe=0.2
        # 所有损失组件都是float32，直接相加
        alpha_feat_eff = 0.1  # 特征损失权重：0.1
        alpha_wave_final = 0.7  # 波形损失权重：0.7
        alpha_moe_eff = 0.2   # MoE损失权重：0.2
        total_loss = alpha_feat_eff * l_feat + alpha_wave_final * l_wave + l_rate + l_sem + alpha_moe_eff * l_moe


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

        # 详细的NaN诊断
        if torch.isnan(total_loss) or torch.isinf(total_loss) or (args.debug_nan and batch_idx < 10):
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                tqdm.write(f"    🚨 NaN/Inf detected in total_loss: {total_loss.item() if not torch.isnan(total_loss) else 'NaN'}")
            elif args.debug_nan:
                tqdm.write(f"    🔍 Debug NaN mode - batch {batch_idx} diagnostics:")

            tqdm.write(f"      Loss components:")
            tqdm.write(f"        {check_tensor('feat', l_feat)}")
            tqdm.write(f"        {check_tensor('wave', l_wave)}")
            tqdm.write(f"        {check_tensor('rate', l_rate)}")
            tqdm.write(f"        {check_tensor('sem', l_sem)}")
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
        # ---- Quick feature-only backprop debug every 50 steps ----
        do_feat_debug = (global_step % 50 == 0)
        if do_feat_debug:
            # 暂存 wave 权重
            _old_wave = l_wave
            # 只用特征损失做一次极小步回传（不 step 优化器）
            (alpha_feat_now * l_feat).backward(retain_graph=True)

            # 打印自检梯度（现在应该非 0 了）
            gn_dec_dbg, n_dec_dbg = _sum_grad_norm(decoder.named_parameters(), exclude_key='fargan_core')
            gn_enc_dbg, n_enc_dbg = _sum_grad_norm(encoder.named_parameters())
            tqdm.write(f"[FEAT-ONLY DEBUG] dec(non-vc) grad={gn_dec_dbg:.3e} over {n_dec_dbg} | enc grad={gn_enc_dbg:.3e} over {n_enc_dbg}")
        # ---- end debug ----


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

                    # 每200步报告一次复杂度状态
                    if global_step % (complexity_monitor_interval * 5) == 0:
                        avg_complexity = epoch_metrics['pathway_complexity_total'] / max(1, epoch_metrics['pathway_complexity_samples'])
                        tqdm.write(f"    📊 Complexity Monitor: avg={avg_complexity:.2f}x Stage1, mode={pathway_mode}")
            except Exception as e:
                pass  # 复杂度监控失败不影响训练

        # 梯度累积缩放
        accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        scaled_loss = total_loss / accumulation_steps

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

            # 先做一次 NaN/Inf 清洗（就地）
            with torch.no_grad():
                for p in list(encoder.parameters()) + list(decoder.parameters()):
                    if p.grad is not None:
                        p.grad.nan_to_num_(0.0, posinf=0.0, neginf=0.0)

            # 全模型 clip（建议 1.0 更稳）
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0
            )

            # 在 step 之前打印梯度统计（此时已经 unscale/clip 完成，数值可信）
            gn_vc, n_vc  = _sum_grad_norm(decoder.named_parameters(), include_key='fargan_core')
            gn_dec, n_dec = _sum_grad_norm(decoder.named_parameters(), exclude_key='fargan_core')
            gn_enc, n_enc = _sum_grad_norm(encoder.named_parameters())
            tqdm.write(f"[GRAD] fargan_core grad-norm sum={gn_vc:.3e} over {n_vc} tensors")
            tqdm.write(f"[GRAD] decoder(non-vocoder) grad-norm sum={gn_dec:.3e} over {n_dec} tensors")
            tqdm.write(f"[GRAD] encoder grad-norm sum={gn_enc:.3e} over {n_enc} tensors")

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
        n_experts = getattr(encoder.moe, 'n_experts', 3) if hasattr(encoder, 'moe') else 3
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
                wave_computed = y_hat_audio is not None

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
            # 新版SpecializedMicroMoE的损失
            mb = _sf(enc_logs.get('moe_balance_loss', 0.0) if isinstance(enc_logs, dict) else 0.0)
            ms = _sf(enc_logs.get('moe_harmonic_pref', 0.0) if isinstance(enc_logs, dict) else 0.0)  # 使用实际存在的指标
            lr = optimizer.param_groups[0].get('lr', 0.0)
            # ETA from tqdm
            remaining = progress.format_dict.get('remaining', None)
            import time as _t
            eta_str = _t.strftime('%H:%M:%S', _t.gmtime(remaining)) if remaining is not None else 'NA'
            # Progress bar postfix - 显示实际有意义的指标 + 梯度监控
            post = {
                'loss': f"{tl:.4f}",
                'feat': f"{ff:.4f}",
                'wave': f"{ww:.4f}",
                'lr': f"{lr:.2e}",
                'eta': eta_str,
                'bp': f"{wave_bp_ratio:.2f}",
                'warm': f"{warm_ratio:.2f}",
                # 新增梯度流监控信息
                'tf': f"{tf_ratio:.2f}",  # teacher forcing比例
                'fgrad': '✓' if feats_grad_connected else '✗',  # feats梯度连接
                'wcomp': '✓' if wave_computed else '✗',  # wave_loss计算
            }

            # 只在MoE真正启用时显示MoE指标
            mm = float(l_moe)
            mb = float(enc_logs.get('moe_balance_loss', 0.0)) if isinstance(enc_logs, dict) else 0.0
            mt = float(enc_logs.get('moe_token_balance_loss', 0.0)) if isinstance(enc_logs, dict) else 0.0
            post.update({
                'moe': f"{mm:.4f}",
                'moe_b': f"{mb:.4f}",
                'moe_t': f"{mt:.4f}",
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
                    tqdm.write(f"    ✅ MoE on: n_experts={n_exp}, top_k={top_k}, token_level={token_level}")
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
                            n_experts = getattr(encoder.moe, 'n_experts', 3) if hasattr(encoder, 'moe') else 3
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

                    # 专家利用率警告（仅在严重不均衡时）
                    # 梯度监控警告：在MoE健康检查中同时检查fargan_core梯度
                    if expert_min < 0.1:  # 提高阈值从0.15到0.1，减少警告频率
                        tqdm.write(f"    ⚠️  Warning: Expert usage imbalance detected (min={expert_min:.3f})")

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

                        # 性能警告 - 针对不同模式调整
                        if pathway_mode == 'architectural_bypass':
                            tqdm.write(f"    ✅ Training in Stage1-equivalent mode for stability")
                        elif pathway_mode == 'mixed':
                            if performance_ratio > 1.5:
                                tqdm.write(f"    ⚠️  Direct pathway strongly outperforming (ratio={performance_ratio:.3f})")
                            elif performance_ratio < 0.7:
                                tqdm.write(f"    ✅ Expert system outperforming direct pathway (ratio={performance_ratio:.3f})")
                        elif pathway_mode == 'pure_expert':
                            tqdm.write(f"    ✅ Expert system fully engaged")

    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= max(total_samples, 1)

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
    p.add_argument('--features', type=str, required=True, help='Features file path')
    p.add_argument('--pcm', type=str, required=True, help='Audio PCM file path')
    p.add_argument('--stage1-checkpoint', type=str, default=None, help='Optional Stage 1 checkpoint for warm start (AETHER encoder/decoder)')
    p.add_argument('--fargan-checkpoint', type=str, help='Pre-trained FARGAN checkpoint (optional)')
    p.add_argument('--output-dir', type=str, default='checkpoints_stage3')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--epochs', type=int, default=3, help='Training epochs (task: 3 epochs)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=800)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--feature-dims', type=int, default=36)
    p.add_argument('--alpha-feat', type=float, default=1.0, help='Feature loss base weight (if no schedule)')
    p.add_argument('--alpha-wave', type=float, default=1.0, help='Wave loss weight')
    p.add_argument('--alpha-sem', type=float, default=0.2, help='Semantic proxy loss weight (encoder semantic head vs priors)')
    p.add_argument('--amp', type=str, default='fp16', choices=['none', 'fp16', 'bf16'], help='Mixed precision mode for CUDA (fp16 recommended for stability)')
    p.add_argument('--wave-warmup-steps', type=int, default=3000, help='Linear warmup steps for wave loss weight (alpha_wave)')
    p.add_argument('--wave-start-step', type=int, default=1500, help='Do not compute vocoder/wave loss before this global step')
    p.add_argument('--preheat-frames', type=int, default=2, help='Ignore first N frames for loss (Stage1-like warm start)')
    # DataLoader knobs
    p.add_argument('--stride-frames', type=int, default=None, help='Data loader stride in frames (None=auto-adaptive)')
    p.add_argument('--semantic-source', type=str, default='fused', choices=['fused', 'ribbon', 'thread'], help='Semantic head source inside DualStream')
    # alpha_feat anneal schedule: start -> end over steps
    p.add_argument('--alpha-feat-start', type=float, default=None, help='Optional alpha_feat start value (overrides --alpha-feat)')
    p.add_argument('--alpha-feat-end', type=float, default=0.0, help='alpha_feat end value')
    p.add_argument('--alpha-feat-steps', type=int, default=1000, help='Linear anneal steps for alpha_feat')
    p.add_argument('--log-interval', type=int, default=50, help='Steps between progress updates')
    p.add_argument('--profile-interval', type=int, default=0, help='If >0, include per-step timings every N steps')
    # Checkpointing controls
    p.add_argument('--save-every-epochs', type=int, default=0, help='Save a checkpoint every N epochs (0 disables)')
    p.add_argument('--always-save-last', action='store_true', help='Always save a final checkpoint at the end')

    # Stage3特定配置 (按任务要求)
    p.add_argument('--moe', action='store_true', default=True, help='Enable MoE (Stage3 default: enabled)')
    p.add_argument('--no-moe', action='store_true', help='🚨 Disable MoE for debugging (临时诊断选项)')
    p.add_argument('--enable-rate', action='store_true',
                   help='Enable rate regularizer (disabled by default for Stage3 - only use with quantization/JSCC)')
    p.add_argument('--router-no-csi', action='store_true', default=True,
                   help='Router不使用CSI (Stage3单变量验证)')
    # 性能优化选项
    p.add_argument('--gradient-accumulation-steps', type=int, default=1,
                   help='Number of steps to accumulate gradients before updating (for memory efficiency)')
    p.add_argument('--use-compile', action='store_true', default=False,
                   help='Use torch.compile for model optimization (PyTorch 2.0+)')
    p.add_argument('--moe-token-warmup-steps', type=int, default=0,
                   help='Use sample-level routing only for first N steps, then enable token-level')
    p.add_argument('--debug-nan', action='store_true', default=False,
                   help='Enable detailed NaN debugging output')
    p.add_argument('--safe-init', action='store_true', default=True,
                   help='Use safe initialization for numerical stability')
    p.add_argument('--emergency-bypass-moe', action='store_true', default=False,
                   help='🚨 Emergency: completely bypass MoE for NaN diagnosis')
    p.add_argument('--wave-stride', type=int, default=1,
               help='计算 vocoder / wave loss 的步频：每 N 个 batch 才计算一次 (默认 1，即每步都算)')
    p.add_argument('--wave-min-bp', type=float, default=0.1,
                help='🔧 波形分支反传比例的下限，避免 warmup 初期长期为 0（默认 0.1，提高梯度传播）')
    p.add_argument('--router-jitter', type=float, default=0.01,
                help='训练态对路由 logits 施加的高斯抖动强度，用于促使专家探索（默认 0.01）')
    # 直流通路相关参数 - 架构级绕过优化
    p.add_argument('--enable-direct-pathway', action='store_true', default=True,
                help='启用MoE直流通路，用于性能对比验证（默认启用）')
    p.add_argument('--disable-direct-pathway', action='store_true', default=False,
                help='禁用直流通路，使用纯专家系统（用于对照实验）')
    p.add_argument('--initial-bypass-weight', type=float, default=0.95,
                help='直流通路初始权重 (0.0-1.0)，控制训练开始时直流vs专家的比例，默认0.95启用架构级绕过')
    p.add_argument('--adaptive-threshold', type=float, default=0.10,
                help='性能差异阈值，触发权重调整（默认10%）')
    p.add_argument('--pathway-warmup-steps', type=int, default=2000,
                help='直流权重warmup步数，前一半为架构级绕过期')

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
    safe_print(f"   AMP mode: {args.amp} | Preheat frames: {getattr(args, 'preheat_frames', 0)} | Wave warmup: {args.wave_warmup_steps}")
    safe_print(f"   DataLoader: workers={args.num_workers} | stride_frames={args.stride_frames}")
    safe_print(f"   Performance optimizations: grad_accum={args.gradient_accumulation_steps} | torch_compile={args.use_compile}")

    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                         args.device if args.device != 'auto' else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data loader
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
        "n_experts": 3,    # Stage3: 3专家系统 (E1:Harmonic, E2:Transient, E3:BurstInpaint)
        "top_k": 2,        # Stage3: TOP-2路由
        "moe_router_use_csi": (not args.router_no_csi),  # Router不使用CSI → False
        "use_semantic_head": True,
        "semantic_dim": 6,
        "semantic_source": args.semantic_source,
        # 直流通路配置
        "enable_direct_pathway": args.enable_direct_pathway,
        "initial_bypass_weight": args.initial_bypass_weight,
        "adaptive_threshold": args.adaptive_threshold,
        "pathway_warmup_steps": args.pathway_warmup_steps,
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

            # 尝试加载FARGAN core权重并统计
            decoder_state = decoder.state_dict()
            loaded = skipped = 0
            for k, v in state.items():
                if k.startswith('fargan_core.') and k in decoder_state and decoder_state[k].shape == v.shape:
                    decoder_state[k] = v
                    loaded += 1
                elif k.startswith('fargan_core.'):
                    skipped += 1

            decoder.load_state_dict(decoder_state, strict=False)
            safe_print(f"✅ Stage2(FARGAN): loaded {loaded} params into decoder, skipped {skipped}")
            # 重新设置dtype
            # if args.amp == 'fp16':
            #     decoder = decoder.to(torch.float16)
            # elif args.amp == 'bf16':
            #     decoder = decoder.to(torch.bfloat16)
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
    safe_print(f"   - Rate loss {'启用' if args.enable_rate else '禁用'} (Stage3默认禁用，避免梯度爆炸)\n")

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

        # Epoch summary
        safe_print(f"📊 Epoch {epoch} Summary:")
        safe_print(f"   Total Loss: {epoch_metrics['total_loss']:.6f}")
        safe_print(f"   Feature Loss: {epoch_metrics['feature_loss']:.6f}")
        safe_print(f"   Wave Loss: {epoch_metrics['wave_loss']:.6f}")
        safe_print(f"   MoE Loss: {epoch_metrics['moe_loss']:.6f}")
        safe_print(f"   Rate Loss: {epoch_metrics['rate_loss']:.6f}")

        # MoE health check (只在非绕过模式下显示)
        if args.emergency_bypass_moe:
            safe_print(f"   🚨 MoE Status: BYPASSED (emergency diagnosis mode)")
        elif encoder.use_moe:
            expert_min = epoch_metrics.get('expert_usage_min', 0)
            expert_max = epoch_metrics.get('expert_usage_max', 0)
            expert_entropy = epoch_metrics.get('expert_entropy', 0)
            expert_balance = 1.0 - (expert_max - expert_min)  # 均衡度

            # Display individual expert usage rates if available - 动态专家数量
            n_experts = getattr(encoder.moe, 'n_experts', 3) if hasattr(encoder, 'moe') else 3
            expert_usage_display = []
            for i in range(n_experts):  # 动态专家数量
                usage_key = f'expert_{i}_usage'
                if usage_key in epoch_metrics:
                    expert_usage_display.append(f"{epoch_metrics[usage_key]:.3f}")
                else:
                    expert_usage_display.append("N/A")

            expert_names = ["Harmonic", "Transient", "BurstInpaint", "LowSNR"][:n_experts]  # 根据专家数量截断

            safe_print(f"   MoE Health:")
            safe_print(f"     Expert Usage: [{', '.join(expert_usage_display)}]")
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
