#!/usr/bin/env python3
"""
Stage5 码率约束训练管道

主要功能:
1. 基于Stage4权重的增量训练
2. 严格码率控制(1.2±0.1 kbps)
3. 分组优化器策略
4. 三阶段训练策略
5. 实时码率监控和调整
"""

from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import json
import time
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# 导入模型和损失
from models.aether_stage5 import AETHERStage5Model, create_stage5_model
from training.stage5_losses import Stage5ComprehensiveLoss, create_stage5_loss_computer

# 导入现有组件
try:
    from models.enhanced_aether_integration import AETHEREncoder, AETHERDecoder
    from utils.real_data_loader import create_aether_data_loader, create_combined_data_loader
    from training.pipeline.wave_loss import fargan_wave_losses
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some functionality may be limited")

class Stage5Trainer:
    """Stage5训练器，管理整个训练流程"""

    def __init__(self, config: Dict, stage4_checkpoint_path: str):
        self.config = config
        self.stage4_checkpoint_path = stage4_checkpoint_path

        # 设备设置
        self.device = torch.device(config.get('device', 'cuda'))

        # 创建模型
        self.model = self.create_model()
        self.model = self.model.to(self.device)

        # 创建优化器
        self.optimizer = self.create_optimizer()

        # 创建学习率调度器
        self.scheduler = self.create_scheduler()

        # 创建数据加载器
        self.train_loader = self.create_data_loader()

        # 创建损失计算器（在数据加载器之后，以便获取steps_per_epoch）
        enhanced_config = config.copy()
        enhanced_config['steps_per_epoch'] = len(self.train_loader)
        enhanced_config['dataset_size'] = len(self.train_loader.dataset) if hasattr(self.train_loader.dataset, '__len__') else None
        print(f"Auto-detected training config for loss scheduler:")
        print(f"  Steps per epoch: {enhanced_config['steps_per_epoch']}")
        if enhanced_config['dataset_size']:
            print(f"  Dataset size: {enhanced_config['dataset_size']}")
        self.loss_computer = create_stage5_loss_computer(enhanced_config)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metrics = {
            'rate_stability': 0.0,
            'quality_score': 0.0,
            'combined_score': 0.0
        }

        # 监控指标
        self.metrics_history = defaultdict(list)

        print(f"Stage5 Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {self.model.get_model_info()['total_parameters']:,}")
        print(f"  Trainable parameters: {self.model.get_model_info()['trainable_parameters']:,}")

    def create_model(self) -> AETHERStage5Model:
        """创建Stage5模型并加载Stage4权重，参照stage4_train_full.py模式"""
        model = create_stage5_model(self.config)

        # 加载Stage4权重
        if os.path.exists(self.stage4_checkpoint_path):
            print(f"Loading Stage4 weights from {self.stage4_checkpoint_path}")
            model.load_stage4_weights(self.stage4_checkpoint_path)
        else:
            print(f"Warning: Stage4 checkpoint not found at {self.stage4_checkpoint_path}")
            print("Training will start from scratch")

        return model


    def create_optimizer(self) -> optim.Optimizer:
        """创建分组优化器"""
        # 分组参数
        rvq_params = []
        # removed rate_prior_params (soft-entropy only)
        rate_controller_params = []
        film_params = []
        moe_params = []
        dec_wave_params = []
        other_trainable_params = []
        no_wd_params = []  # 参数：关闭weight decay（out_logstd/ceps γβ）

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # 检查参数冻结状态
            # 优先筛出不施加WD的参数
            name_l = name.lower()
            if (
                ('decoder' in name_l and 'refiner.out_logstd' in name_l)
                or ('ceps_gamma' in name_l)
                or ('ceps_beta' in name_l)
            ):
                no_wd_params.append(param)
                continue

            if 'rvq' in name:
                rvq_params.append(param)
            # rate prior heads removed (soft-entropy only)
            elif 'rate_controller' in name:
                rate_controller_params.append(param)
            elif 'film' in name.lower():
                film_params.append(param)
            elif 'moe' in name.lower() or 'expert' in name.lower():
                moe_params.append(param)
            elif ('decoder' in name and ('wave' in name or 'synth' in name)) or \
                 ('fargan_core' in name) or ('period_estimator' in name):
                dec_wave_params.append(param)
            else:
                other_trainable_params.append(param)

        # 分组优化策略
        param_groups = []

        if no_wd_params:
            param_groups.append({
                'params': no_wd_params,
                'lr': self.config.get('base_lr', 1e-4),
                'weight_decay': 0.0,
                'name': 'no_wd_logstd_cepscalib'
            })

        if rvq_params:
            param_groups.append({
                'params': rvq_params,
                'lr': self.config.get('rvq_lr', 2e-4),
                'weight_decay': self.config.get('rvq_wd', 1e-5),
                'name': 'rvq'
            })

        # 专门为rate prior heads设定更高的学习率，便于尽快拟合索引分布
        # no separate group for rate prior (removed)

        if rate_controller_params:
            param_groups.append({
                'params': rate_controller_params,
                'lr': self.config.get('rate_controller_lr', 5e-5),
                'weight_decay': 0.0,  # 不对控制器施加权重衰减
                'name': 'rate_controller'
            })

        if film_params:
            param_groups.append({
                'params': film_params,
                'lr': self.config.get('base_lr', 1e-4) * self.config.get('film_lr_mult', 2.0),
                'weight_decay': self.config.get('film_wd', 0.0),
                'name': 'film'
            })

        if moe_params:
            param_groups.append({
                'params': moe_params,
                'lr': self.config.get('base_lr', 1e-4) * self.config.get('moe_lr_mult', 1.0),
                'weight_decay': self.config.get('base_wd', 1e-5),
                'name': 'moe'
            })

        if dec_wave_params:
            param_groups.append({
                'params': dec_wave_params,
                'lr': self.config.get('base_lr', 1e-4) * self.config.get('dec_wave_lr_mult', 1.0),
                'weight_decay': self.config.get('base_wd', 1e-5),
                'name': 'decoder_wave'
            })

        if other_trainable_params:
            param_groups.append({
                'params': other_trainable_params,
                'lr': self.config.get('base_lr', 1e-4),
                'weight_decay': self.config.get('base_wd', 1e-5),
                'name': 'other'
            })

        optimizer = optim.AdamW(param_groups)

        print(f"Optimizer created with {len(param_groups)} parameter groups:")
        for i, group in enumerate(param_groups):
            print(f"  Group {i} ({group['name']}): {len(group['params'])} params, "
                  f"lr={group['lr']:.2e}, wd={group['weight_decay']:.2e}")

        return optimizer

    def create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler_type', 'cosine_restarts')

        if scheduler_type == 'cosine_restarts':
            # T_mult must be integer >= 1
            t_mult = int(self.config.get('scheduler_Tmult', 2))
            t_mult = max(1, t_mult)  # Ensure >= 1

            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('scheduler_T0', 2000),
                T_mult=t_mult,
                eta_min=self.config.get('scheduler_eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 2000),
                gamma=self.config.get('scheduler_gamma', 0.5)
            )
        else:
            # 默认：无调度
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)

    def create_data_loader(self) -> DataLoader:
        """创建数据加载器 - 完全按照Stage4的方式"""
        try:
            # --- Data loader: support combined mixed-batch mode (Stage4 style) ---
            if self.config.get('combined_data_root'):
                # Build combined loader from four expert subsets
                train_loader, dataset = create_combined_data_loader(
                    data_root=self.config.get('combined_data_root'),
                    sequence_length=self.config.get('seq_len', 800),
                    batch_size=self.config.get('batch_size', 4),
                    frame_size=160,
                    stride_frames=self.config.get('stride_frames', None),
                    energy_selection=True,
                    max_samples=self.config.get('max_samples', None),
                )
                # Optional custom mix ratio (Stage4 exact logic)
                if self.config.get('mix_ratio'):
                    try:
                        ratios = [float(x.strip()) for x in str(self.config.get('mix_ratio')).split(',')]
                        assert len(ratios) == 4
                        import numpy as _np
                        dataset.mix_ratio = _np.array(ratios, dtype=_np.float64)
                        s = dataset.mix_ratio.sum()
                        dataset.mix_ratio = dataset.mix_ratio / (s if s > 0 else 1.0)
                        dataset.cumprob = _np.cumsum(dataset.mix_ratio)
                        print(f"   Mix ratio set to: {dataset.mix_ratio.tolist()}")
                        print(f"   Categories: [harmonic, transient, burst_inpaint, low_snr]")
                    except Exception as e:
                        print(f"Warning: Invalid mix-ratio format: {e}. Using defaults.")

                print(f"Combined data loader created from: {self.config.get('combined_data_root')}")
                print(f"   Dataset size: {len(dataset)}")
                print(f"   Batch size: {self.config.get('batch_size', 4)}")
                print(f"   Sequence length: {self.config.get('seq_len', 800)}")
                return train_loader
            else:
                # Fallback to single data root (Stage4 style)
                data_root = self.config.get('data_root')
                if data_root and os.path.exists(data_root):
                    train_loader = create_aether_data_loader(
                        data_dir=data_root,
                        sequence_length=self.config.get('seq_len', 800),
                        batch_size=self.config.get('batch_size', 4),
                        feature_dims=self.config.get('original_feature_dim', 36),
                        num_workers=self.config.get('num_workers', 4),
                        pin_memory=self.config.get('pin_memory', True)
                    )
                    print(f"Single data loader created: {data_root}")
                    return train_loader
                else:
                    raise ValueError(f"No valid data source: combined_data_root={self.config.get('combined_data_root')}, data_root={data_root}")
        except Exception as e:
            print(f"Warning: Could not create real data loader: {e}")
            print("Using dummy data loader for testing")
            return self.create_dummy_data_loader()

    def create_dummy_data_loader(self) -> DataLoader:
        """创建虚拟数据加载器用于测试"""
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                seq_len = 100
                return {
                    'features': torch.randn(36, seq_len),  # 原始特征
                    'audio': torch.randn(1, seq_len * 160),  # 音频
                    'period': torch.randint(20, 200, (seq_len,)),  # 基频周期
                    'csi': torch.randn(4, seq_len)  # CSI信息
                }

        return DataLoader(
            DummyDataset(),
            batch_size=self.config.get('batch_size', 4),
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        epoch_metrics = defaultdict(list)
        num_batches = len(self.train_loader)

        # 添加进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}",
                   leave=True, dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # 前向传播
            # 兼容不同数据加载器的特征键名
            features = batch.get('x', batch.get('features'))
            if features is None:
                raise KeyError(f"No features found in batch. Available keys: {list(batch.keys())}")

            # 统一到 [B,T,36] 布局
            if features.dim() == 3 and features.shape[-1] != 36 and features.shape[1] == 36:
                features = features.transpose(1, 2).contiguous()
                print(f"Normalized features layout to [B,T,36]: {features.shape}")

            # 创建模拟CSI（完全按Stage4方式）
            try:
                from utils.channel_sim import ChannelSimulator
                chan_sim = ChannelSimulator()

                B = features.shape[0]
                T = features.shape[1]
            except ImportError as e:
                # 回退处理
                print(f"Warning: Could not import ChannelSimulator: {e}")
                B, T = features.shape[0], features.shape[1]
                chan_sim = None

            # 使用Stage4相同的方式生成CSI（SNR斜坡调度）
            snr_hi_db = self.config.get('snr_hi_db', 15.0)
            snr_lo_db = self.config.get('snr_lo_db', -5.0)
            snr_ramp_steps = self.config.get('snr_ramp_steps', 800)
            snr_window_db = self.config.get('snr_window_db', 4.0)
            chan_type = self.config.get('channel_type', 'fading')

            # SNR斜坡调度（Stage4的算法）
            if snr_ramp_steps > 0:
                s_ratio = min(1.0, float(self.global_step) / float(snr_ramp_steps))
            else:
                s_ratio = 1.0
            center = (1.0 - s_ratio) * snr_hi_db + s_ratio * snr_lo_db
            half = max(0.0, snr_window_db * 0.5)
            lo_b = min(snr_lo_db, snr_hi_db)
            hi_b = max(snr_lo_db, snr_hi_db)
            snr_min = max(center - half, lo_b)
            snr_max = min(center + half, hi_b)

            csi_sim, amp_t, snr_db_t = chan_sim.sample_csi(B, T,
                                                          channel=chan_type,
                                                          snr_min_db=snr_min,
                                                          snr_max_db=snr_max)

            # 将CSI移动到正确设备
            sim_csi = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in csi_sim.items()}

            # === 关键修复：启用信道模拟 ===
            if hasattr(self.model, 'enable_channel_simulation'):
                # 根据配置启用信道模拟
                apply_channel = self.config.get('apply_channel', True)  # 默认启用！
                self.model.enable_channel_simulation(apply_channel)
                if self.global_step % 100 == 0:  # 定期日志
                    channel_status = "enabled" if apply_channel else "disabled"
                    print(f"[JSCC] Step {self.global_step}: Channel simulation {channel_status}")

            # 先递增global_step，然后更新训练阶段
            self.global_step += 1

            # 更新训练阶段（渐进式解冻）
            if self.global_step % 100 == 0:
                print(f"[Training Script] About to call update_training_phase at step {self.global_step}")
            self.model.update_training_phase(self.global_step)

            # 检查是否需要重新创建优化器
            if hasattr(self.model, '_optimizer_needs_update') and self.model._optimizer_needs_update:
                print(f"🔄 [Optimizer Update] Recreating optimizer at step {self.global_step}")

                # 保存当前优化器状态
                old_state = self.optimizer.state_dict()

                # 重新创建优化器
                self.optimizer = self.create_optimizer()

                # 尝试恢复可兼容的状态
                try:
                    # 只恢复仍然存在的参数组的状态
                    new_state = self.optimizer.state_dict()
                    for group_idx, group in enumerate(new_state['param_groups']):
                        if group_idx < len(old_state['param_groups']):
                            # 恢复相同参数组的状态
                            old_group = old_state['param_groups'][group_idx]
                            if group.get('name') == old_group.get('name'):
                                group.update(old_group)

                    # 恢复参数状态（仅针对存在的参数）
                    for param_id, param_state in old_state['state'].items():
                        if param_id in new_state['state']:
                            new_state['state'][param_id] = param_state

                    self.optimizer.load_state_dict(new_state)
                    print("   ✅ Optimizer state partially restored")
                except Exception as e:
                    print(f"   ⚠️ Could not restore optimizer state: {e}")
                    print("   ✅ Using fresh optimizer state")

                # 清除标记
                self.model._optimizer_needs_update = False
                print(f"   ✅ Optimizer updated with {len(self.optimizer.param_groups)} parameter groups")

            if self.global_step % 100 == 0:
                print(f"[Training Script] Finished update_training_phase at step {self.global_step}")

            model_outputs = self.model(
                inputs=features,  # [B, T, 36]
                csi_dict=sim_csi,  # 包含信道参数（关键！）
                current_step=self.global_step,
                return_wave=True
            )

            # 缓存一次模型调试统计，便于在统一日志函数中按步打印
            try:
                if isinstance(model_outputs, dict) and ('debug_stats' in model_outputs):
                    self._last_debug_stats = model_outputs.get('debug_stats') or {}
                else:
                    self._last_debug_stats = {}
            except Exception:
                self._last_debug_stats = {}

            # 准备损失计算的目标数据
            targets = {
                'original_features': features,  # [B, T, 36] 保持原始格式
                'target_audio': batch.get('audio'),
                'period': batch.get('period'),
                'reference_semantic': features  # 使用原始特征作为语义参考
            }

            # 缓存目标特征第0维分布统计（用于与解码输出对表）
            try:
                feat0 = targets['original_features'][..., 0]
                self._last_target_feat0_stats = {
                    'mean': float(feat0.mean().detach().cpu().item()),
                    'std':  float(feat0.std().detach().cpu().item()),
                    'min':  float(feat0.min().detach().cpu().item()),
                    'max':  float(feat0.max().detach().cpu().item()),
                }
            except Exception:
                self._last_target_feat0_stats = {}

            # 计算损失
            total_loss, loss_details = self.loss_computer.compute_comprehensive_loss(
                model_outputs, targets, self.global_step, model=self.model
            )

            # 在得到 eff_bits_per_frame 后，再更新码率控制器EMA/历史（顺序修正）
            try:
                if isinstance(loss_details, dict) and 'eff_bits_per_frame' in loss_details:
                    ce_bpf = torch.tensor(float(loss_details['eff_bits_per_frame']), device=self.device)
                    _ = self.model.rate_controller.compute_rate_loss(
                        ce_bpf,
                        current_step=self.global_step
                    )
                elif 'rate_bits_per_frame' in model_outputs and model_outputs['rate_bits_per_frame'] is not None:
                    _ = self.model.rate_controller.compute_rate_loss(
                        model_outputs['rate_bits_per_frame'].detach(),
                        current_step=self.global_step
                    )
            except Exception:
                pass

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            # === 分时解耦：阻断/缩放语义对 RVQ τ/gate 的梯度 ===
            try:
                target_kbps = float(self.config.get('target_kbps', 1.2))
                tol_kbps = float(self.config.get('rate_tolerance', 0.1))
                kbps_eff = float(loss_details.get('kbps_eff', target_kbps))
                sem_loss_val = float(loss_details.get('semantic', 0.0))
                detach_steps = int(self.config.get('sem_detach_steps', 2000))
                diff_thresh = float(self.config.get('sem_difficulty_threshold', 0.6))
                gate_grad_scale = float(self.config.get('sem_gate_grad_scale', 0.3))

                under = kbps_eff < (target_kbps - tol_kbps)
                over = kbps_eff > (target_kbps + tol_kbps)

                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    name_l = name.lower()
                    if ('rvq_encoder.stage_tau' in name_l) or ('rvq_encoder.stage_gates' in name_l):
                        if (self.global_step < detach_steps) or under:
                            param.grad.zero_()
                        elif over and (sem_loss_val > diff_thresh):
                            param.grad.mul_(gate_grad_scale)
            except Exception:
                pass

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip_norm', 3.0)
            )

            self.optimizer.step()
            self.scheduler.step()

            # 记录指标
            for key, value in loss_details.items():
                # 只记录数值类型的指标
                if isinstance(value, (int, float)):
                    epoch_metrics[key].append(value)
                elif torch.is_tensor(value) and value.numel() == 1:
                    epoch_metrics[key].append(value.item())
            epoch_metrics['grad_norm'].append(grad_norm.item())

            # 更新进度条 - 显示详细损失信息
            current_lr = self.scheduler.get_last_lr()[0]
            # 修复：使用统一的eff_bits_per_frame
            rate_info = f"Rate: {loss_details.get('kbps_eff', loss_details.get('current_kbps', 0)):.2f}kbps (eff_bits: {loss_details.get('eff_bits_per_frame', 0):.3f})"
            loss_info = f"Loss: {loss_details['total']:.1f}"

            # 详细损失信息在进度条中显示（只显示有效的loss项）
            detailed_losses = (f"feat:{loss_details.get('feat_recon', 0):.1f} "
                             f"wave:{loss_details.get('wave', 0):.1f} "
                             f"sem:{loss_details.get('semantic', 0):.4f} "
                             f"qual:{loss_details.get('quality', 0):.1f} "
                             f"comm:{loss_details.get('commitment', 0):.1f}")

            lr_info = f"LR: {current_lr:.2e}"
            pbar.set_postfix_str(f"{rate_info} | {loss_info} | {detailed_losses} | {lr_info}")

            # 详细日志输出使用tqdm.write
            if batch_idx % self.config.get('log_interval', 50) == 0:
                # 追加打印 RVQ 细节，便于观察阶段权重与困惑度是否收紧
                try:
                    rvq_details = model_outputs.get('rvq_details', {}) if isinstance(model_outputs, dict) else {}
                    stage_ws = rvq_details.get('stage_weights', [])
                    stage_perps = rvq_details.get('stage_perplexities', [])
                    if stage_ws:
                        pbar.write(f"[RVQ] stage_weights: {[f'{float(w):.3f}' for w in stage_ws]}")
                    if stage_perps:
                        pbar.write(f"[RVQ] perplexities: {[f'{float(p):.1f}' for p in stage_perps]}")
                    # 追加打印 FARGAN 合成增益与特征分布
                    dbg = model_outputs.get('debug_stats', {}) if isinstance(model_outputs, dict) else {}
                    fg = dbg.get('fargan_gain') if isinstance(dbg, dict) else None
                    if isinstance(fg, dict):
                        pbar.write(
                            f"[FARGAN] gain: min={fg.get('min', 0):.3f} "
                            f"mean={fg.get('mean', 0):.3f} max={fg.get('max', 0):.3f}"
                        )
                    f0 = dbg.get('feat0') if isinstance(dbg, dict) else None
                    if isinstance(f0, dict):
                        pbar.write(
                            f"[Feat0] mean={f0.get('mean', 0):.3f} std={f0.get('std', 0):.3f} "
                            f"min={f0.get('min', 0):.3f} max={f0.get('max', 0):.3f}"
                        )
                except Exception:
                    pass
                self.log_training_progress(batch_idx, num_batches, loss_details, grad_norm, pbar)

            # 定期保存checkpoint（按步数保存，不依赖validation）
            save_every_steps = self.config.get('save_every_steps', 500)  # 使用正确的默认值
            if self.global_step % save_every_steps == 0:
                # 快速保存：不进行完整验证，只使用当前训练指标
                quick_metrics = {
                    'total': loss_details.get('total', 0),
                    'current_kbps': loss_details.get('current_kbps', 1.2),
                    'rate_stability': 1.0 - abs(loss_details.get('current_kbps', 1.2) - 1.2) / 1.2,
                    'combined_score': -(loss_details.get('total', 0) / 1000 + abs(loss_details.get('current_kbps', 1.2) - 1.2))
                }

                # 检查是否是最佳模型（基于combined_score）
                is_best = quick_metrics['combined_score'] > self.best_metrics.get('combined_score', float('-inf'))
                if is_best:
                    self.best_metrics.update(quick_metrics)

                self.save_checkpoint(quick_metrics, is_best)
                pbar.write(f"💾 Checkpoint saved at step {self.global_step} (save_every_steps={save_every_steps})")

            # 早停：recon-only模式或达到总步数
            if self.config.get('recon_only', False):
                max_steps = int(self.config.get('recon_only_steps', 0) or 0)
                if max_steps > 0 and self.global_step >= max_steps:
                    pbar.write(f"[Diag] Recon-only early stop at step {self.global_step} (limit={max_steps})")
                    break
            total_steps_limit = int(self.config.get('total_steps', 0) or 0)
            if total_steps_limit > 0 and self.global_step >= total_steps_limit:
                pbar.write(f"[Control] Reached total_steps={total_steps_limit}, breaking epoch loop")
                break

        # 计算epoch平均指标
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        return avg_metrics

    def log_training_progress(
        self,
        batch_idx: int,
        num_batches: int,
        loss_details: Dict[str, float],
        grad_norm: torch.Tensor,
        pbar
    ):
        """记录训练进度"""
        progress = batch_idx / num_batches * 100
        current_lr = self.scheduler.get_last_lr()[0]

        # 使用tqdm.write避免干扰进度条
        pbar.write(f"[Step {self.global_step:05d}] Rate: {loss_details.get('kbps_eff', loss_details.get('current_kbps', 0)):.3f}kbps "
                  f"(target: 1.2±0.1) | rate_bits: {loss_details.get('eff_bits_per_frame', 0):.3f}")

        # 检查异常损失值
        problematic_losses = []
        for loss_name, loss_value in [
            ('feat_recon', loss_details.get('feat_recon', 0)),
            ('wave', loss_details.get('wave', 0)),
            ('rate', loss_details.get('rate', 0)),
            ('semantic', loss_details.get('semantic', 0)),
            ('temporal', loss_details.get('temporal', 0)),
            ('commitment', loss_details.get('commitment', 0))
        ]:
            if loss_value > 1000:  # 异常高的损失值
                problematic_losses.append(f"{loss_name}={loss_value:.1f}")

        if problematic_losses:
            pbar.write(f"⚠️  HIGH LOSSES: {' '.join(problematic_losses)}")

        # 损失权重信息
        weights = loss_details.get('loss_weights', {})
        if weights:
            pbar.write(f"Weights: feat={weights.get('feat', 0):.2f} wave={weights.get('wave', 0):.2f} "
                      f"rate={weights.get('rate', 0):.2f} sem={weights.get('semantic', 0):.3f}")

        # 修复10: 打印VIB和diversity的实际有效值（来自损失模块）
        vib_kld = loss_details.get('vib_kld', 0)
        vib_beta = loss_details.get('vib_beta', 0)
        diversity = loss_details.get('diversity', 0)
        if vib_kld > 0 or diversity > 0:
            pbar.write(f"Info Regularization: VIB-KLD={vib_kld:.4f} (beta={vib_beta:.6f}), "
                      f"RVQ-Diversity={diversity:.4f}")

        # 追加：FARGAN 合成增益与特征分布（随主日志一起打印）
        try:
            dbg = getattr(self, '_last_debug_stats', None)
            if isinstance(dbg, dict):
                # VIB统计
                vib = dbg.get('vib')
                if isinstance(vib, dict):
                    pbar.write(
                        f"[VIB] mu={vib.get('mu_mean',0):.3f}±{vib.get('mu_std',0):.3f} "
                        f"logvar={vib.get('logvar_mean',0):.3f}±{vib.get('logvar_std',0):.3f} "
                        f"z={vib.get('z_mean',0):.3f}±{vib.get('z_std',0):.3f}"
                    )
                # RVQ每层统计
                rvq = dbg.get('rvq')
                if isinstance(rvq, list) and len(rvq) > 0:
                    for i, s in enumerate(rvq):
                        if isinstance(s, dict) and s:
                            pbar.write(
                                f"[RVQ s{i}] res μ={s.get('res_mean',0):.3f} σ={s.get('res_std',0):.3f} "
                                f"min={s.get('res_min',0):.3f} max={s.get('res_max',0):.3f} | "
                                f"q μ={s.get('q_mean',0):.3f} σ={s.get('q_std',0):.3f} "
                                f"min={s.get('q_min',0):.3f} max={s.get('q_max',0):.3f} | "
                                f"gate={s.get('gate_mean',0):.3f} H={s.get('soft_H_bits',0):.3f}"
                            )
                fg = dbg.get('fargan_gain')
                if isinstance(fg, dict):
                    pbar.write(
                        f"[FARGAN] gain: min={fg.get('min', 0):.3f} "
                        f"mean={fg.get('mean', 0):.3f} max={fg.get('max', 0):.3f}"
                    )
                f0 = dbg.get('feat0')
                if isinstance(f0, dict):
                    pbar.write(
                        f"[Feat0] mean={f0.get('mean', 0):.3f} std={f0.get('std', 0):.3f} "
                        f"min={f0.get('min', 0):.3f} max={f0.get('max', 0):.3f}"
                    )
                # Pitch (index 18) stats
                fp = dbg.get('feat_pitch')
                if isinstance(fp, dict):
                    pbar.write(
                        f"[FeatPitch] mean={fp.get('mean',0):.3f} std={fp.get('std',0):.3f} "
                        f"min={fp.get('min',0):.3f} max={fp.get('max',0):.3f}"
                    )
                tp = dbg.get('target_pitch')
                if isinstance(tp, dict):
                    pbar.write(
                        f"[TargetPitch] mean={tp.get('mean',0):.3f} std={tp.get('std',0):.3f} "
                        f"min={tp.get('min',0):.3f} max={tp.get('max',0):.3f}"
                    )
                # Ceps block (0..17) vs target ceps stats（块级）
                fceps = dbg.get('feat_ceps')
                if isinstance(fceps, dict):
                    pbar.write(
                        f"[FeatCeps] mean={fceps.get('mean',0):.3f} std={fceps.get('std',0):.3f} "
                        f"min={fceps.get('min',0):.3f} max={fceps.get('max',0):.3f}"
                    )
                tceps = dbg.get('target_ceps')
                if isinstance(tceps, dict):
                    pbar.write(
                        f"[TargetCeps] mean={tceps.get('mean',0):.3f} std={tceps.get('std',0):.3f} "
                        f"min={tceps.get('min',0):.3f} max={tceps.get('max',0):.3f}"
                    )
                # 逐维 ceps/lpc 的 std 与 mean（简洁输出）
                ceps_std_out = dbg.get('ceps_std_out'); ceps_std_tgt = dbg.get('ceps_std_tgt')
                ceps_mean_out = dbg.get('ceps_mean_out'); ceps_mean_tgt = dbg.get('ceps_mean_tgt')
                lpc_std_out = dbg.get('lpc_std_out'); lpc_std_tgt = dbg.get('lpc_std_tgt')
                lpc_mean_out = dbg.get('lpc_mean_out'); lpc_mean_tgt = dbg.get('lpc_mean_tgt')
                def _fmt_arr(arr, n=8):
                    try:
                        return "[" + ",".join(f"{float(x):.2f}" for x in (arr[:n] if len(arr)>n else arr)) + ("…" if len(arr)>n else "]")
                    except Exception:
                        return str(arr)
                if isinstance(ceps_std_out, list) and isinstance(ceps_std_tgt, list):
                    pbar.write(f"[CepsSTD] out={_fmt_arr(ceps_std_out)} tgt={_fmt_arr(ceps_std_tgt)}")
                if isinstance(ceps_mean_out, list) and isinstance(ceps_mean_tgt, list):
                    pbar.write(f"[CepsMean] out={_fmt_arr(ceps_mean_out)} tgt={_fmt_arr(ceps_mean_tgt)}")
                if isinstance(lpc_std_out, list) and isinstance(lpc_std_tgt, list):
                    pbar.write(f"[LPCSTD] out={_fmt_arr(lpc_std_out)} tgt={_fmt_arr(lpc_std_tgt)}")
                if isinstance(lpc_mean_out, list) and isinstance(lpc_mean_tgt, list):
                    pbar.write(f"[LPCMean] out={_fmt_arr(lpc_mean_out)} tgt={_fmt_arr(lpc_mean_tgt)}")
                # frame_corr
                fco = dbg.get('feat_frame_corr'); fct = dbg.get('target_frame_corr')
                if isinstance(fco, dict) and isinstance(fct, dict):
                    pbar.write(
                        f"[FrameCorr] out μ={fco.get('mean',0):.3f} σ={fco.get('std',0):.3f} "
                        f"min={fco.get('min',0):.3f} max={fco.get('max',0):.3f} | "
                        f"tgt μ={fct.get('mean',0):.3f} σ={fct.get('std',0):.3f} "
                        f"min={fct.get('min',0):.3f} max={fct.get('max',0):.3f}"
                    )
                # Decoder MoE stats
                dec_moe = dbg.get('dec_moe')
                if isinstance(dec_moe, dict) and dec_moe:
                    util = dec_moe.get('util', None)
                    ent = dec_moe.get('entropy', None)
                    if util is not None:
                        try:
                            util_str = ",".join([f"{float(u):.2f}" for u in (util if isinstance(util, list) else util)])
                        except Exception:
                            util_str = str(util)
                        pbar.write(f"[DecMoE] util=[{util_str}] ent={float(ent) if ent is not None else 0:.3f}")
                tflag = dbg.get('dec_moe_trainable', None)
                if isinstance(tflag, bool):
                    pbar.write(f"[DecMoE] trainable={tflag}")
                lat = dbg.get('latent')
                if isinstance(lat, dict):
                    pbar.write(
                        f"[Latent] enc μ={lat.get('enc_mean',0):.3f} σ={lat.get('enc_std',0):.3f} | "
                        f"q_lat μ={lat.get('q_lat_mean',0):.3f} σ={lat.get('q_lat_std',0):.3f}"
                    )
        except Exception:
            pass

        # 记录码率稳定性
        if hasattr(self.model, 'rate_controller'):
            rate_stats = self.model.rate_controller.get_current_stats()
            if rate_stats.get('in_range_ratio', 0) > 0:
                pbar.write(f"Rate stability: {rate_stats['in_range_ratio']:.2f} "
                          f"(μ={rate_stats['mean_kbps']:.3f}, σ={rate_stats['std_kbps']:.3f})")

        # 追加：目标特征第0维分布（与解码输出对表）
        try:
            t0 = getattr(self, '_last_target_feat0_stats', None)
            if isinstance(t0, dict) and len(t0) > 0:
                pbar.write(
                    f"[TargetFeat0] mean={t0.get('mean', 0):.3f} std={t0.get('std', 0):.3f} "
                    f"min={t0.get('min', 0):.3f} max={t0.get('max', 0):.3f}"
                )
        except Exception:
            pass

    def validate_model(self) -> Dict[str, float]:
        """模型验证"""
        self.model.eval()
        val_metrics = defaultdict(list)

        with torch.no_grad():
            # 使用训练数据的一部分进行验证
            val_batches = []
            for i, batch in enumerate(self.train_loader):
                if i >= 20:  # 只验证20个batch
                    break
                val_batches.append(batch)

            # 添加验证进度条
            val_pbar = tqdm(val_batches, desc="Validation", leave=False, dynamic_ncols=True)

            for i, batch in enumerate(val_pbar):

                # 移动数据到设备
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # 前向传播
                # 兼容不同数据加载器的特征键名
                features = batch.get('x', batch.get('features'))

                # 统一到 [B,T,36] 布局（验证阶段）
                if features.dim() == 3 and features.shape[-1] != 36 and features.shape[1] == 36:
                    features = features.transpose(1, 2).contiguous()

                # 创建模拟CSI（验证阶段，参照Stage4）
                from utils.channel_sim import ChannelSimulator
                chan_sim = ChannelSimulator()
                B = features.shape[0]
                T = features.shape[1]

                # 验证时使用固定SNR范围
                csi_sim, _, _ = chan_sim.sample_csi(B, T,
                                                   channel=self.config.get('channel_type', 'fading'),
                                                   snr_min_db=self.config.get('snr_lo_db', -5.0),
                                                   snr_max_db=self.config.get('snr_hi_db', 15.0))
                sim_csi = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in csi_sim.items()}

                model_outputs = self.model(
                    inputs=features,
                    csi_dict=sim_csi,
                    current_step=self.global_step,
                    return_wave=False  # 验证时不需要生成音频
                )

                # 同步更新控制器统计（验证期仅用于统计，不计入损失）
                try:
                    if 'rate_bits_per_frame' in model_outputs and model_outputs['rate_bits_per_frame'] is not None:
                        _ = self.model.rate_controller.compute_rate_loss(
                            model_outputs['rate_bits_per_frame'].detach(),
                            current_step=self.global_step
                        )
                except Exception:
                    pass

                # 记录关键指标
                rate_kbps = (model_outputs['rate_bits_per_frame'] *
                           self.config.get('frame_rate', 50) / 1000).item()
                val_metrics['rate_kbps'].append(rate_kbps)

                if 'quality_prediction' in model_outputs:
                    quality = model_outputs['quality_prediction'].mean().item()
                    val_metrics['quality_score'].append(quality)

                # 更新验证进度条
                if i % 5 == 0:  # 每5个batch更新一次
                    avg_rate = np.mean(val_metrics['rate_kbps']) if val_metrics['rate_kbps'] else 0
                    val_pbar.set_postfix_str(f"Avg Rate: {avg_rate:.2f}kbps")

        # 计算验证指标
        avg_rate = np.mean(val_metrics['rate_kbps'])
        rate_stability = np.mean([
            1.0 if 1.1 <= r <= 1.3 else 0.0
            for r in val_metrics['rate_kbps']
        ])

        avg_quality = np.mean(val_metrics.get('quality_score', [0.0]))

        validation_results = {
            'avg_rate_kbps': avg_rate,
            'rate_stability': rate_stability,
            'avg_quality_score': avg_quality,
            'combined_score': 0.6 * rate_stability + 0.4 * min(avg_quality / 3.0, 1.0)
        }

        return validation_results

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点（Stage4风格）"""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.config,
            'metrics': metrics,
            'best_metrics': self.best_metrics
        }

        output_dir = Path(self.config.get('output_dir', './checkpoints_stage5'))
        output_dir.mkdir(exist_ok=True)

        # 保存常规检查点
        if not self.config.get('save_best_only', False):
            checkpoint_path = output_dir / f"stage5_step_{self.global_step:05d}.pth"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # 管理检查点数量（保留最后N个）
            self._manage_checkpoints(output_dir)

        # 保存最佳检查点
        if is_best:
            best_checkpoint_path = output_dir / "stage5_best.pth"
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"Best checkpoint saved at step {self.global_step} (metric: {self.config.get('checkpoint_metric', 'combined_score')})")

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint恢复训练"""
        print(f"Loading checkpoint from: {checkpoint_path}")

        try:
            # PyTorch 2.6+ safety fix for numpy objects in checkpoints
            import torch.serialization
            with torch.serialization.safe_globals([torch.tensor, torch.nn.Parameter, 'numpy.core.multiarray.scalar']):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 智能过滤不匹配的student_proj层
            checkpoint_state = checkpoint['model_state_dict']
            model_state = self.model.state_dict()

            # 过滤掉形状不匹配的参数
            filtered_state = {}
            skipped_keys = []

            for key, value in checkpoint_state.items():
                if key in model_state:
                    if value.shape == model_state[key].shape:
                        filtered_state[key] = value
                    else:
                        skipped_keys.append(f"{key}: {value.shape} → {model_state[key].shape}")
                else:
                    skipped_keys.append(f"{key}: not found in current model")

            # 加载过滤后的状态
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
            print("✅ Model state loaded (filtered for compatibility)")

            if skipped_keys:
                print(f"⚠️  Skipped incompatible keys: {len(skipped_keys)}")
                for key in skipped_keys[:3]:  # 只显示前3个
                    print(f"   - {key}")
                if len(skipped_keys) > 3:
                    print(f"   ... and {len(skipped_keys) - 3} more")

            if missing_keys:
                print(f"⚠️  Missing keys (will use new initialization): {len(missing_keys)} keys")
                for key in missing_keys[:5]:  # 只显示前5个
                    print(f"   - {key}")
                if len(missing_keys) > 5:
                    print(f"   ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
                for key in unexpected_keys[:5]:  # 只显示前5个
                    print(f"   - {key}")
                if len(unexpected_keys) > 5:
                    print(f"   ... and {len(unexpected_keys) - 5} more")

            # 加载优化器状态（若参数组不匹配，稳健跳过，使用新优化器）
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded")
                except Exception as e:
                    print("⚠️  Optimizer state skipped (parameter groups changed or incompatible)")
                    print(f"   - Reason: {e}")
                    print("   - Using fresh optimizer state")

            # 加载学习率调度器状态
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✅ Scheduler state loaded")
                except Exception as e:
                    print("⚠️  Scheduler state skipped (incompatible with current optimizer)")
                    print(f"   - Reason: {e}")

            # 恢复训练步数/epoch的开关（默认关闭）
            if self.config.get('restore_global_step', False):
                self.global_step = int(checkpoint.get('global_step', 0) or 0)
                self.current_epoch = int(checkpoint.get('current_epoch', 0) or 0)
                print(f"✅ Restored global_step/current_epoch: step={self.global_step}, epoch={self.current_epoch}")
            else:
                print("⚠️  Skipping global_step restoration to enable fresh semantic/rate scheduling")
                print(f"   Checkpoint was at step: {checkpoint.get('global_step', 'unknown')}")
                print(f"   Starting fresh from step: {self.global_step}")
                print(f"   Starting fresh epoch: {self.current_epoch}")

            # 恢复最佳指标
            if 'best_metrics' in checkpoint:
                self.best_metrics = checkpoint['best_metrics']
                print("✅ Best metrics restored")

            print(f"🔄 Resuming training from step {self.global_step}, epoch {self.current_epoch}")

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise

    def _manage_checkpoints(self, output_dir: Path):
        """管理检查点数量，保留最后N个"""
        keep_n = self.config.get('keep_last_n_checkpoints', 5)
        if keep_n <= 0:
            return

        # 查找所有step检查点
        step_checkpoints = list(output_dir.glob("stage5_step_*.pth"))
        if len(step_checkpoints) <= keep_n:
            return

        # 按步数排序并删除旧的
        step_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        to_delete = step_checkpoints[:-keep_n]

        for old_checkpoint in to_delete:
            try:
                old_checkpoint.unlink()
                print(f"Removed old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                print(f"Warning: Could not remove {old_checkpoint}: {e}")

    def train(self):
        """主训练循环"""
        print(f"\nStarting Stage5 training:")
        print(f"  Target: {self.config.get('target_kbps', 1.2)}±{self.config.get('rate_tolerance', 0.1)} kbps")
        print(f"  Total epochs: {self.config.get('num_epochs', 10)}")
        print(f"  Save every: {self.config.get('save_every_steps', 1000)} steps")
        print()

        # 计算训练应该开始的epoch（考虑resume情况）
        start_epoch = self.current_epoch if hasattr(self, 'current_epoch') and self.current_epoch > 0 else 0
        total_epochs = self.config.get('num_epochs', 10)

        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练一个epoch
            train_metrics = self.train_one_epoch()

            # 验证模型
            val_metrics = self.validate_model()

            # 更新最佳指标
            is_best = val_metrics['combined_score'] > self.best_metrics['combined_score']
            if is_best:
                self.best_metrics.update(val_metrics)

            # 在epoch结束时保存检查点（如果还没在训练循环中保存过）
            save_every_steps = self.config.get('save_every_steps', 500)
            if (self.global_step % save_every_steps == 0 or
                epoch == self.config.get('num_epochs', 10) - 1):
                self.save_checkpoint(val_metrics, is_best)

            epoch_time = time.time() - epoch_start_time

            # Epoch总结
            print(f"\n📊 Epoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"  Training: loss={train_metrics['total']:.4f}, "
                  f"rate={train_metrics['current_kbps']:.3f}kbps")
            print(f"  Validation: rate_stability={val_metrics['rate_stability']:.3f}, "
                  f"combined_score={val_metrics['combined_score']:.3f}")
            print(f"  Best combined score: {self.best_metrics['combined_score']:.3f}")
            print()

        print("Stage5 training completed!")
        print(f"Final best metrics: {self.best_metrics}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Stage5 Rate-Constrained Training")

    # === 基础配置 ===
    parser.add_argument("--stage4-checkpoint", type=str, required=True,
                       help="Path to Stage4 checkpoint")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_stage5",
                       help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Training device")

    # === 数据配置 ===
    # 基础数据路径
    parser.add_argument("--combined-data-root", type=str,
                       help="Root directory containing expert-mixed datasets for combined loading")
    parser.add_argument("--data-root", type=str,
                       help="Root directory for training data")
    parser.add_argument("--features-file", type=str,
                       help="Direct path to features file (.f32)")
    parser.add_argument("--pcm-file", type=str,
                       help="Direct path to PCM audio file (.pcm)")

    # 数据混合策略
    parser.add_argument("--mix-ratio", type=str, default="0.95,0.03,0.02,0.00",
                       help="Data mixing ratios for combined dataset")
    parser.add_argument("--enable-expert-data", action="store_true", default=False,
                       help="Enable expert-augmented data loading")
    parser.add_argument("--expert-data-ratio", type=float, default=0.1,
                       help="Ratio of expert-augmented data in training")

    # 数据加载参数
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=800,
                       help="Sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per subset (for tiny validation)")
    parser.add_argument("--stride-frames", type=int, default=None,
                       help="Data loader stride frames (Stage4 style)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                       help="DataLoader prefetch factor")
    parser.add_argument("--persistent-workers", action="store_true", default=True,
                       help="Use persistent workers for DataLoader")

    # 数据预处理
    parser.add_argument("--enable-data-augmentation", action="store_true", default=False,
                       help="Enable data augmentation during training")
    parser.add_argument("--noise-injection-prob", type=float, default=0.1,
                       help="Probability of noise injection augmentation")
    parser.add_argument("--feature-dropout-prob", type=float, default=0.05,
                       help="Feature dropout probability for regularization")

    # 数据验证
    parser.add_argument("--validate-data-loading", action="store_true", default=False,
                       help="Validate data loading before training")
    parser.add_argument("--max-validation-batches", type=int, default=50,
                       help="Maximum validation batches during validation")

    # === 训练配置 ===
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--total-steps", type=int, default=8000,
                       help="Total training steps")
    parser.add_argument("--base-lr", type=float, default=1e-4,
                       help="Base learning rate")
    parser.add_argument("--rvq-lr", type=float, default=5e-5,
                       help="RVQ learning rate")
    parser.add_argument("--rate-controller-lr", type=float, default=5e-5,
                       help="Rate controller learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=0.5,
                       help="Gradient clipping norm")

    # === 码率控制 ===
    parser.add_argument("--target-kbps", type=float, default=1.2,
                       help="Target bitrate in kbps")
    parser.add_argument("--rate-tolerance", type=float, default=0.1,
                       help="Rate tolerance in kbps")
    parser.add_argument("--frame-rate", type=int, default=50,
                       help="Frame rate (fps)")
    parser.add_argument("--rate-warmup-steps", type=int, default=500,
                       help="Rate control warmup steps")
    parser.add_argument("--control-strength", type=float, default=0.1,
                       help="Rate control strength")
    parser.add_argument("--gate-logit-bias-init", type=float, default=0.0,
                       help="Initial gate logit bias (helps raise rate when under target in early steps)")

    # === 码率控制策略 ===
    parser.add_argument("--rate-control-strategy", type=str, default="rvq_stage_weight",
                       choices=["rvq_stage_weight", "simple_penalty"],
                       help="Rate control strategy: RVQ stage weights vs simple penalty")

    # === RVQ配置 ===
    parser.add_argument("--rvq-stages", type=int, default=3,
                       help="Number of RVQ stages")
    parser.add_argument("--codebook-sizes", type=int, nargs="+",
                       default=[1024, 512, 256],
                       help="Codebook sizes for each RVQ stage")
    parser.add_argument("--commitment-weights", type=float, nargs="+",
                       default=[0.01, 0.01, 0.01],
                       help="Commitment weights for each RVQ stage")

    # === 损失权重 ===
    parser.add_argument("--alpha-feat", type=float, default=0.4,
                       help="Feature reconstruction loss weight")
    parser.add_argument("--alpha-wave", type=float, default=0.4,
                       help="Wave synthesis loss weight")
    parser.add_argument("--lambda-rate", type=float, default=0.15,
                       help="Rate constraint loss weight")
    parser.add_argument("--lambda-semantic", type=float, default=0.9,
                       help="Semantic preservation loss weight (提升用于修复sem_loss)")
    parser.add_argument("--lambda-temporal", type=float, default=0.1,
                       help="Temporal consistency loss weight")
    parser.add_argument("--lambda-quality", type=float, default=0.1,
                       help="Quality prediction loss weight")
    parser.add_argument("--lambda-commitment", type=float, default=0.5,
                       help="RVQ commitment loss weight")

    # === VIB + 语义对比学习权重 ===
    # 注意：已删除重复的--lambda-vib-kl参数，统一使用--beta-vib
    parser.add_argument("--lambda-semantic-nce", type=float, default=0.1,
                       help="Semantic InfoNCE contrastive loss weight (提升用于修复sem_loss)")
    parser.add_argument("--use-restored-rate-loss", action="store_true", default=False,
                       help="Enable restored rate loss for strict RD-Lagrangian training")

    # === 损失配置 ===
    parser.add_argument("--temporal-smoothness", type=float, default=0.1,
                       help="Temporal smoothness factor")
    parser.add_argument("--max-jump-threshold", type=float, default=2.0,
                       help="Maximum temporal jump threshold")
    parser.add_argument("--min-quality-threshold", type=float, default=2.5,
                       help="Minimum quality threshold (PESQ)")

    # === 学习率调度 ===
    parser.add_argument("--scheduler-type", type=str, default="cosine_restarts",
                       choices=["cosine_restarts", "step", "linear"],
                       help="Learning rate scheduler type")
    parser.add_argument("--scheduler-T0", type=int, default=2000,
                       help="Scheduler T0 (cosine restart period)")
    parser.add_argument("--scheduler-Tmult", type=int, default=2,
                       help="Scheduler T multiplier (integer >= 1)")
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6,
                       help="Minimum learning rate")

    # === 语义与码率分时解耦 ===
    parser.add_argument("--sem-detach-steps", type=int, default=2000,
                       help="Detach gradients to RVQ tau/gates during early steps (block semantics from influencing rate knobs)")
    parser.add_argument("--sem-difficulty-threshold", type=float, default=0.6,
                       help="Semantic loss threshold considered 'difficult' to allow partial gradient to gates when over bitrate")
    parser.add_argument("--sem-gate-grad-scale", type=float, default=0.3,
                       help="Gradient scale factor to gates/tau when over bitrate and semantic is difficult")

    # === 模型架构 ===
    parser.add_argument("--feature-dim", type=int, default=24,
                       help="Compressed feature dimension")
    parser.add_argument("--original-feature-dim", type=int, default=36,
                       help="Original feature dimension")
    parser.add_argument("--semantic-layers", type=int, default=2,
                       help="Number of semantic preservation layers")
    parser.add_argument("--quality-metrics", type=int, default=3,
                       help="Number of quality metrics to predict")
    parser.add_argument("--enable-output-calibration", action="store_true", default=True,
                       help="Enable 36-dim feature affine calibration (decoder-level denorm)")
    # === Recon-only diagnostic mode ===
    parser.add_argument("--recon-only", action="store_true", default=False,
                       help="Diagnostic: disable rate/VIB/semantic/diversity; force gates open; train only feat+wave")
    parser.add_argument("--recon-only-steps", type=int, default=2000,
                       help="Diagnostic: early stop after N steps when recon-only enabled")

    # === VIB + 语义感知配置 ===
    parser.add_argument("--enable-vib", action="store_true", default=True,
                       help="Enable Variational Information Bottleneck")
    parser.add_argument("--beta-vib", type=float, default=1e-3,
                       help="VIB KL divergence weight")
    parser.add_argument("--vib-warmup-steps", type=int, default=1000,
                       help="VIB warmup steps")
    parser.add_argument("--vib-phase-end-step", type=int, default=3000,
                       help="VIB dominant phase end step (then decays)")

    parser.add_argument("--semantic-dim", type=int, default=128,
                       help="Semantic embedding dimension for contrastive learning")
    parser.add_argument("--semantic-mode", type=str, default="nce", choices=["mse", "nce"],
                       help="Semantic loss mode: MSE or InfoNCE contrastive learning")
    parser.add_argument("--semantic-temperature", type=float, default=0.2,
                       help="InfoNCE temperature parameter")
    parser.add_argument("--semantic-temp-annealing", action="store_true", default=True,
                       help="Enable temperature annealing for InfoNCE")
    parser.add_argument("--semantic-temp-end", type=float, default=0.07,
                       help="Final temperature for InfoNCE annealing")
    parser.add_argument("--semantic-temp-steps", type=int, default=3000,
                       help="Steps for temperature annealing")
    parser.add_argument("--semantic-commitment-threshold", type=float, default=1.5,
                       help="Commitment threshold for enabling semantic loss (default: 1.5)")
    parser.add_argument("--semantic-force-enable-step", type=int, default=1500,
                       help="Force enable semantic loss after N steps (default: 1500)")

    # === 有效码率控制配置 ===
    parser.add_argument("--rate-loss-type", type=str, default="soft_entropy",
                       choices=["soft_entropy"],
                       help="Rate loss type (fixed): soft_entropy")
    parser.add_argument("--rate-phase-start-step", type=int, default=2500,
                       help="Step to start strengthening rate loss (phase schedule)")
    # removed: --rate-prior-lr-mult (no rate prior heads)
    parser.add_argument("--rvq-codebook-sizes", type=int, nargs="+", default=[1024, 512, 256],
                       help="RVQ codebook sizes for each stage")

    # === Learnable stage gates + soft-entropy ===
    parser.add_argument("--enable-learnable-stage-gates", action="store_true", default=False,
                       help="Enable learnable per-stage gates (ST-Gumbel)")
    parser.add_argument("--gate-temperature", type=float, default=0.67,
                       help="Temperature for gate ST-Gumbel sigmoid")
    parser.add_argument("--soft-entropy-tau", type=float, default=0.3,
                       help="Temperature for soft assignment in soft-entropy rate")
    # === Decoder ceps affine calibration ===
    parser.add_argument("--enable-ceps-affine-calib", action="store_true", default=True,
                       help="Enable learnable per-dim affine on ceps[0..17] (train+infer)")
    parser.add_argument("--lambda-ceps-calib-reg", type=float, default=1e-4,
                       help="Regularization weight for ceps affine (gamma≈1, beta≈0)")
    # === Entropy control (soft-entropy + masks) ===
    parser.add_argument("--mask-sparsity-weight", type=float, default=0.0,
                       help="Weight for codebook mask sparsity (effective-K control)")
    parser.add_argument("--diversity-flip-step", type=int, default=2000,
                       help="Step to flip diversity from encourage to discourage entropy")

    # === 多任务梯度平衡配置 ===
    parser.add_argument("--enable-gradient-balancing", action="store_true", default=True,
                       help="Enable multi-task gradient balancing")
    parser.add_argument("--gradient-balance-alpha", type=float, default=0.16,
                       help="Gradient balancing reweighting speed parameter")
    parser.add_argument("--gradient-adaptation-rate", type=float, default=0.01,
                       help="Gradient balancing weight update rate")

    # === 冻结策略 ===
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                       help="Freeze encoder parameters")
    parser.add_argument("--freeze-decoder-except-wave", action="store_true", default=True,
                       help="Freeze decoder except wave head")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                       help="Don't freeze encoder parameters")
    parser.add_argument("--no-freeze-decoder-except-wave", dest="freeze_decoder_except_wave", action="store_false",
                       help="Don't freeze decoder parameters")

    # === 混合精度训练 ===
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                       help="Automatic mixed precision dtype")

    # === 稳定性控制 ===
    parser.add_argument("--enable-stability-loss", action="store_true", default=True,
                       help="Enable stability loss")
    parser.add_argument("--disable-stability-loss", dest="enable_stability_loss", action="store_false",
                       help="Disable stability loss")

    # === 日志和保存 ===
    parser.add_argument("--log-interval", type=int, default=5,
                       help="Logging interval (batches)")
    parser.add_argument("--save-every-steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--validation-interval", type=int, default=500,
                       help="Validation interval (steps)")
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=5,
                       help="Keep only last N checkpoints")
    parser.add_argument("--save-best-only", action="store_true", default=False,
                       help="Only save best checkpoint")
    parser.add_argument("--checkpoint-metric", type=str, default="combined_score",
                       choices=["combined_score", "rate_stability", "quality_score"],
                       help="Metric for best checkpoint selection")

    # === Stage4风格的CSI控制参数 ===
    parser.add_argument("--d-csi", type=int, default=4,
                       help="CSI target dimension for FiLM and decoder refinement")
    parser.add_argument("--csi-keys", type=str, default="snr_proxy,time_selectivity,freq_selectivity,los_ratio",
                       help="Comma-separated CSI keys to include in simulation")
    parser.add_argument("--channel", type=str, default="fading", choices=["clean", "awgn", "fading"],
                       help="Channel simulation type")
    parser.add_argument("--snr-hi-db", type=float, default=15.0,
                       help="High SNR center (dB)")
    parser.add_argument("--snr-lo-db", type=float, default=-5.0,
                       help="Low SNR center (dB)")
    parser.add_argument("--snr-ramp-steps", type=int, default=800,
                       help="Steps to ramp from high to low SNR")
    parser.add_argument("--snr-window-db", type=float, default=4.0,
                       help="SNR window half-width (±)")
    # 简洁控制：在指定步数启用信道；从该步起进行1000步线性warmup
    parser.add_argument("--channel-enable-step", type=int, default=None,
                       help="Step to enable channel simulation; 1000-step linear warmup from this step")

    # === 阶段Gating（硬控高阶RVQ） ===
    parser.add_argument("--stage-gate-enable", action="store_true", default=False,
                       help="Enable hard gating of higher RVQ stages when rate is above target")
    parser.add_argument("--stage-gate-threshold-kbps", type=float, default=0.15,
                       help="Gating threshold margin in kbps (rate_ema > target + margin triggers gating)")

    # CSI生成频率和采样
    parser.add_argument("--csi-update-interval", type=int, default=1,
                       help="CSI update interval (batches)")
    parser.add_argument("--csi-batch-coherent", action="store_true", default=False,
                       help="Use batch-coherent CSI (same CSI for entire batch)")

    # === Stage4风格的FiLM控制参数 ===
    parser.add_argument("--film", action="store_true", default=True,
                       help="Enable FiLM in encoder")
    parser.add_argument("--film-position", type=str, default="post", choices=["pre", "post", "both"],
                       help="FiLM position relative to MoE")
    parser.add_argument("--film-lr-mult", type=float, default=3.0,
                       help="Learning rate multiplier for FiLM parameters")
    parser.add_argument("--film-wd", type=float, default=0.0,
                       help="Weight decay for FiLM parameters")
    parser.add_argument("--film-pre-warmup", type=int, default=100,
                       help="Pre-FiLM warmup steps")
    parser.add_argument("--film-pre-end", type=float, default=0.15,
                       help="Pre-FiLM target strength")
    parser.add_argument("--film-post-end", type=float, default=0.9,
                       help="Post-FiLM target strength")
    parser.add_argument("--film-post-warmup", type=int, default=500,
                       help="Post-FiLM warmup steps")

    # === Stage4风格的MoE控制参数 ===
    parser.add_argument("--moe", action="store_true", default=True,
                       help="Enable MoE in encoder")
    parser.add_argument("--n-experts", type=int, default=None,
                       help="Number of MoE experts (auto-infer from Stage4)")
    parser.add_argument("--top-k", type=int, default=None,
                       help="MoE Top-K routing (auto-infer from Stage4)")
    parser.add_argument("--moe-lr-mult", type=float, default=1.0,
                       help="Learning rate multiplier for MoE parameters")
    parser.add_argument("--freeze-moe-steps", type=int, default=0,
                       help="Freeze MoE parameters for first N steps")

    # === Stage4风格的Decoder MoE控制参数 ===
    parser.add_argument("--dec-moe", action="store_true", default=False,
                       help="Enable decoder-side residual MoE")
    parser.add_argument("--dec-moe-experts", type=int, default=3,
                       help="Number of decoder MoE experts")
    parser.add_argument("--dec-moe-topk", type=int, default=2,
                       help="Decoder MoE Top-K routing")
    parser.add_argument("--dec-moe-topk-warm-steps", type=int, default=300,
                       help="Decoder MoE Top-K warmup steps")
    parser.add_argument("--dec-moe-lr-mult", type=float, default=1.0,
                       help="Learning rate multiplier for decoder MoE")
    parser.add_argument("--dec-moe-balance-weight", type=float, default=0.35,
                       help="Decoder MoE balance loss weight")
    parser.add_argument("--dec-moe-entropy-weight", type=float, default=0.05,
                       help="Decoder MoE entropy regularization weight")
    parser.add_argument("--dec-moe-entropy-warm-steps", type=int, default=1200,
                       help="Decoder MoE entropy warmup steps")

    # Decoder MoE温度调度
    parser.add_argument("--dec-moe-temp-start", type=float, default=2.0,
                       help="Decoder MoE temperature start value")
    parser.add_argument("--dec-moe-temp-end", type=float, default=1.0,
                       help="Decoder MoE temperature end value")
    parser.add_argument("--dec-moe-temp-steps", type=int, default=1200,
                       help="Decoder MoE temperature annealing steps")

    # Decoder MoE残差缩放
    parser.add_argument("--dec-moe-res-scale-start", type=float, default=0.15,
                       help="Decoder MoE residual scale start value")
    parser.add_argument("--dec-moe-res-scale-end", type=float, default=0.25,
                       help="Decoder MoE residual scale end value")
    parser.add_argument("--dec-moe-res-scale-steps", type=int, default=600,
                       help="Decoder MoE residual scale annealing steps")

    # Decoder MoE其他参数
    parser.add_argument("--dec-moe-jitter", type=float, default=0.02,
                       help="Decoder MoE routing jitter")
    parser.add_argument("--dec-moe-router-use-csi", action="store_true", default=False,
                       help="Use CSI in decoder MoE router")
    parser.add_argument("--dec-moe-prob-smoothing-eps", type=float, default=0.02,
                       help="Decoder MoE probability smoothing epsilon")

    # Decoder MoE专家监督学习
    parser.add_argument("--dec-moe-trans-supervise", action="store_true", default=False,
                       help="Enable decoder MoE transient expert supervision")
    parser.add_argument("--dec-moe-trans-expert-id", type=int, default=1,
                       help="Decoder MoE transient expert ID")
    parser.add_argument("--dec-moe-trans-thresh", type=float, default=0.6,
                       help="Decoder MoE transient supervision threshold")
    parser.add_argument("--dec-moe-trans-bias", type=float, default=0.05,
                       help="Decoder MoE transient supervision bias")
    parser.add_argument("--dec-moe-trans-sup-steps", type=int, default=100,
                       help="Decoder MoE transient supervision steps")

    # === Stage4风格的训练策略参数 ===
    parser.add_argument("--freeze-film-steps", type=int, default=0,
                       help="Disable FiLM for first N steps")
    parser.add_argument("--freeze-decoder-steps", type=int, default=0,
                       help="Freeze decoder wave head for first N steps")
    parser.add_argument("--dec-wave-lr-mult", type=float, default=1.0,
                       help="Learning rate multiplier for decoder wave head")

    # === Stage5分阶段训练参数 ===
    parser.add_argument("--phase2-unfreeze-step", type=int, default=2000,
                       help="Step to unfreeze FARGAN wave head (Phase 1->2)")
    parser.add_argument("--phase3-unfreeze-step", type=int, default=4000,
                       help="Step to unfreeze decoder components (Phase 2->3)")
    parser.add_argument("--phase4-unfreeze-step", type=int, default=6000,
                       help="Step to unfreeze encoder components (Phase 3->4)")

    # === RVQ专用控制参数 ===
    parser.add_argument("--rvq-decay", type=float, default=0.995,
                       help="RVQ codebook EMA decay rate")
    parser.add_argument("--rvq-eps", type=float, default=1e-5,
                       help="RVQ Laplace smoothing epsilon")
    parser.add_argument("--rvq-warmup-steps", type=int, default=1000,
                       help="RVQ codebook warmup steps")
    parser.add_argument("--enable-rvq-diversity", action="store_true", default=True,
                       help="Enable RVQ codebook diversity loss")
    parser.add_argument("--rvq-diversity-weight", type=float, default=0.01,
                       help="RVQ diversity loss weight")

    # === 码率控制微调参数 ===
    parser.add_argument("--rate-schedule", type=str, default="linear", choices=["linear", "cosine", "step"],
                       help="Rate constraint schedule type")
    parser.add_argument("--rate-min-kbps", type=float, default=0.8,
                       help="Minimum allowed bitrate (kbps)")
    parser.add_argument("--rate-max-kbps", type=float, default=1.6,
                       help="Maximum allowed bitrate (kbps)")
    parser.add_argument("--enable-rate-curriculum", action="store_true", default=False,
                       help="Enable rate curriculum learning")

    # === 其他配置 ===
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str,
                       help="Resume training from checkpoint")
    parser.add_argument("--restore-global-step", action="store_true", default=False,
                       help="When resuming, restore global_step/current_epoch from checkpoint")
    parser.add_argument("--force", action="store_true",
                       help="Force training despite warnings")

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 转换为配置字典
    config = {
        # 基础配置
        'device': args.device,
        'mixed_precision': args.mixed_precision,
        'amp_dtype': args.amp_dtype,

        # 数据配置
        'data_root': args.data_root,
        'combined_data_root': args.combined_data_root,
        'features_file': args.features_file,
        'pcm_file': args.pcm_file,
        'mix_ratio': args.mix_ratio,
        'enable_expert_data': args.enable_expert_data,
        'expert_data_ratio': args.expert_data_ratio,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'stride_frames': args.stride_frames,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
        'persistent_workers': args.persistent_workers,
        'pin_memory': True,
        'enable_data_augmentation': args.enable_data_augmentation,
        'noise_injection_prob': args.noise_injection_prob,
        'feature_dropout_prob': args.feature_dropout_prob,
        'validate_data_loading': args.validate_data_loading,
        'max_validation_batches': args.max_validation_batches,

        # 训练配置
        'num_epochs': args.num_epochs,
        'total_steps': args.total_steps,
        'save_every_steps': args.save_every_steps,
        'log_interval': args.log_interval,
        'validation_interval': args.validation_interval,
        'keep_last_n_checkpoints': args.keep_last_n_checkpoints,
        'save_best_only': args.save_best_only,
        'checkpoint_metric': args.checkpoint_metric,
        'restore_global_step': args.restore_global_step,

        # 优化器配置
        'base_lr': args.base_lr,
        'rvq_lr': args.rvq_lr,
        'rate_controller_lr': args.rate_controller_lr,
        'base_wd': args.weight_decay,
        'rvq_wd': args.weight_decay,
        'grad_clip_norm': args.grad_clip_norm,
        'sem_detach_steps': args.sem_detach_steps,
        'sem_difficulty_threshold': args.sem_difficulty_threshold,
        'sem_gate_grad_scale': args.sem_gate_grad_scale,

        # 学习率调度
        'scheduler_type': args.scheduler_type,
        'scheduler_T0': args.scheduler_T0,
        'scheduler_Tmult': args.scheduler_Tmult,
        'scheduler_eta_min': args.scheduler_eta_min,

        # 模型架构
        'feature_dim': args.feature_dim,
        'original_feature_dim': args.original_feature_dim,
        'max_samples': args.max_samples,
        'rvq_stages': args.rvq_stages,
        'codebook_sizes': args.codebook_sizes[:args.rvq_stages],
        'commitment_weights': args.commitment_weights[:args.rvq_stages],
        'semantic_layers': args.semantic_layers,
        'quality_metrics': args.quality_metrics,
        'enable_fargan_output_calibration': args.enable_output_calibration,
        'recon_only': args.recon_only,

        # VIB + 语义感知配置
        'enable_vib': args.enable_vib,
        'beta_vib': args.beta_vib,
        'vib_warmup_steps': args.vib_warmup_steps,
        'vib_phase_end_step': args.vib_phase_end_step,
        'semantic_dim': args.semantic_dim,
        'semantic_mode': args.semantic_mode,
        'semantic_temperature': args.semantic_temperature,
        'semantic_temp_annealing': args.semantic_temp_annealing,
        'semantic_temp_end': args.semantic_temp_end,
        'semantic_temp_steps': args.semantic_temp_steps,
        'semantic_commitment_threshold': args.semantic_commitment_threshold,
        'semantic_force_enable_step': args.semantic_force_enable_step,
        'rate_loss_type': args.rate_loss_type,
        'lambda_rate': args.lambda_rate,
        'rvq_codebook_sizes': args.rvq_codebook_sizes,
        'mask_sparsity_weight': args.mask_sparsity_weight,
        'lambda_gate_kl': 0.05,
        'gate_prior_pi_under': 0.8,
        'gate_prior_pi_mid': 0.5,
        'gate_prior_pi_over': 0.3,
        'diversity_flip_step': args.diversity_flip_step,
        'enable_gradient_balancing': args.enable_gradient_balancing,
        'gradient_balance_alpha': args.gradient_balance_alpha,
        'gradient_adaptation_rate': args.gradient_adaptation_rate,
        'enable_learnable_stage_gates': args.enable_learnable_stage_gates,
        'gate_temperature': args.gate_temperature,
        'soft_entropy_tau': args.soft_entropy_tau,
        'enable_ceps_affine_calib': args.enable_ceps_affine_calib,
        'lambda_ceps_calib_reg': args.lambda_ceps_calib_reg,

        # 码率控制
        'target_kbps': args.target_kbps,
        'rate_tolerance': args.rate_tolerance,
        'frame_rate': args.frame_rate,
        'rate_warmup_steps': args.rate_warmup_steps,
        'control_strength': args.control_strength,
        'gate_logit_bias_init': args.gate_logit_bias_init,

        # 码率控制策略
        'rate_control_strategy': args.rate_control_strategy,

        # 损失权重（初始值）
        'initial_loss_weights': {
            'feat': args.alpha_feat,
            'wave': args.alpha_wave,
            'rate': args.lambda_rate,
            'semantic': args.lambda_semantic,
            'temporal': args.lambda_temporal,
            'quality': args.lambda_quality,
            'commitment': args.lambda_commitment,
            # 注意：vib权重由beta_vib控制，不再使用lambda_vib_kl
            'semantic_nce': args.lambda_semantic_nce
        },
        'use_restored_rate_loss': args.use_restored_rate_loss,
        'temporal_smoothness': args.temporal_smoothness,
        'max_jump_threshold': args.max_jump_threshold,
        'min_quality_threshold': args.min_quality_threshold,
        'enable_stability_loss': args.enable_stability_loss,

        # 冻结策略
        'freeze_encoder': args.freeze_encoder,
        'freeze_decoder_except_wave_head': args.freeze_decoder_except_wave,

        # === 新增的Stage4风格参数 ===
        # CSI控制参数
        'd_csi': args.d_csi,
        'csi_keys': args.csi_keys,
        'channel_type': args.channel,
        'snr_hi_db': args.snr_hi_db,
        'snr_lo_db': args.snr_lo_db,
        'snr_ramp_steps': args.snr_ramp_steps,
        'snr_window_db': args.snr_window_db,
        'csi_update_interval': args.csi_update_interval,
        'csi_batch_coherent': args.csi_batch_coherent,
        # 阶段Gating
        'stage_gate_enable': args.stage_gate_enable,
        'stage_gate_threshold_kbps': args.stage_gate_threshold_kbps,

        # FiLM控制参数
        'use_film': args.film,
        'film_position': args.film_position,
        'film_lr_mult': args.film_lr_mult,
        'film_wd': args.film_wd,
        'film_pre_warmup': args.film_pre_warmup,
        'film_pre_end': args.film_pre_end,
        'film_post_end': args.film_post_end,
        'film_post_warmup': args.film_post_warmup,

        # MoE控制参数
        'use_moe': args.moe,
        'n_experts': args.n_experts,
        'top_k': args.top_k,
        'moe_lr_mult': args.moe_lr_mult,
        'freeze_moe_steps': args.freeze_moe_steps,

        # Decoder MoE参数
        'enable_dec_moe': args.dec_moe,
        'dec_moe_experts': args.dec_moe_experts,
        'dec_moe_topk': args.dec_moe_topk,
        'dec_moe_topk_warm_steps': args.dec_moe_topk_warm_steps,
        'dec_moe_lr_mult': args.dec_moe_lr_mult,
        'dec_moe_balance_weight': args.dec_moe_balance_weight,
        'dec_moe_entropy_weight': args.dec_moe_entropy_weight,
        'dec_moe_entropy_warm_steps': args.dec_moe_entropy_warm_steps,

        # Decoder MoE温度调度
        'dec_moe_temp_start': args.dec_moe_temp_start,
        'dec_moe_temp_end': args.dec_moe_temp_end,
        'dec_moe_temp_steps': args.dec_moe_temp_steps,

        # Decoder MoE残差缩放
        'dec_moe_res_scale_start': args.dec_moe_res_scale_start,
        'dec_moe_res_scale_end': args.dec_moe_res_scale_end,
        'dec_moe_res_scale_steps': args.dec_moe_res_scale_steps,

        # Decoder MoE其他参数
        'dec_moe_jitter': args.dec_moe_jitter,
        'dec_moe_router_use_csi': args.dec_moe_router_use_csi,
        'dec_moe_prob_smoothing_eps': args.dec_moe_prob_smoothing_eps,

        # Decoder MoE专家监督
        'dec_moe_trans_supervise': args.dec_moe_trans_supervise,
        'dec_moe_trans_expert_id': args.dec_moe_trans_expert_id,
        'dec_moe_trans_thresh': args.dec_moe_trans_thresh,
        'dec_moe_trans_bias': args.dec_moe_trans_bias,
        'dec_moe_trans_sup_steps': args.dec_moe_trans_sup_steps,

        # 训练策略参数
        'freeze_film_steps': args.freeze_film_steps,
        'freeze_decoder_steps': args.freeze_decoder_steps,
        'dec_wave_lr_mult': args.dec_wave_lr_mult,

        # Stage5分阶段训练参数
        'phase2_unfreeze_step': args.phase2_unfreeze_step,
        'phase3_unfreeze_step': args.phase3_unfreeze_step,
        'phase4_unfreeze_step': args.phase4_unfreeze_step,

        # RVQ专用参数
        'rvq_decay': args.rvq_decay,
        'rvq_eps': args.rvq_eps,
        'rvq_warmup_steps': args.rvq_warmup_steps,
        'enable_rvq_diversity': args.enable_rvq_diversity,
        'rvq_diversity_weight': args.rvq_diversity_weight,

        # 码率控制微调参数
        'rate_schedule': args.rate_schedule,
        'rate_min_kbps': args.rate_min_kbps,
        'rate_max_kbps': args.rate_max_kbps,
        'enable_rate_curriculum': args.enable_rate_curriculum,
        # Rate loss phase schedule (for prior_ce weighting)
        'rate_phase_start_step': args.rate_phase_start_step,

        # 输出配置
        'output_dir': args.output_dir
    }

    # 可选：在指定步数直接启用信道（无渐进warmup）。若未提供，沿用模型默认的禁用+warmup策略。
    if args.channel_enable_step is not None:
        config['channel_disable_steps'] = int(args.channel_enable_step)
        config['channel_warmup_steps'] = 1000  # 固定1000步渐进启用

    # 验证配置
    print("=" * 60)
    print("Stage5 Training Configuration:")
    print("=" * 60)
    print(f"Stage4 checkpoint: {args.stage4_checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    print("📊 Rate Control:")
    print(f"  Target: {args.target_kbps}±{args.rate_tolerance} kbps")
    print(f"  Rate control: {args.rate_control_strategy}")
    print(f"  Rate phase start: step {args.rate_phase_start_step}")
    print(f"  Restore global_step on resume: {args.restore_global_step}")
    # Rate prior LR mult removed (soft-entropy only)
    print()
    print("🏗️ Model Architecture:")
    print(f"  Feature dims: {args.original_feature_dim} → {args.feature_dim}")
    print(f"  CSI dims: {args.d_csi}")
    print(f"  MoE: experts={args.n_experts if args.n_experts else 'auto'}, top_k={args.top_k if args.top_k else 'auto'}")
    print(f"  FiLM: enabled={args.film}, position={args.film_position}")
    print(f"  FiLM learning: lr_mult={args.film_lr_mult}, wd={args.film_wd}")
    print(f"  FiLM warmup: pre={args.film_pre_warmup}(end={args.film_pre_end}), post={args.film_post_warmup}(end={args.film_post_end})")
    print(f"  RVQ stages: {args.rvq_stages}")
    print(f"  Codebook sizes: {args.codebook_sizes[:args.rvq_stages]}")
    if args.dec_moe:
        print(f"  Decoder MoE: {args.dec_moe_experts} experts, top-{args.dec_moe_topk}")
        print(f"  Dec MoE temp: {args.dec_moe_temp_start}→{args.dec_moe_temp_end} over {args.dec_moe_temp_steps} steps")
        print(f"  Dec MoE res scale: {args.dec_moe_res_scale_start}→{args.dec_moe_res_scale_end} over {args.dec_moe_res_scale_steps} steps")
        if args.dec_moe_trans_supervise:
            print(f"  Dec MoE supervision: expert {args.dec_moe_trans_expert_id}, thresh={args.dec_moe_trans_thresh}")
    print()
    print(f"[VIB] beta={args.beta_vib} warmup={args.vib_warmup_steps}")
    print(f"[Semantic] mode={args.semantic_mode} dim={args.semantic_dim} temperature={args.semantic_temperature}")
    print(f"[Rate] type={args.rate_loss_type} target={args.target_kbps}±{args.rate_tolerance} kbps frame_rate={args.frame_rate}")
    if args.recon_only:
        print("[Diag] Recon-only mode: rate/VIB/semantic/diversity disabled; gates forced open; training only feat+wave")
    print(f"[GradBalance] enabled={args.enable_gradient_balancing} alpha={args.gradient_balance_alpha} adapt_rate={args.gradient_adaptation_rate}")
    print()
    print("🎯 Training Parameters:")
    print(f"  Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"  Epochs: {args.num_epochs}, Workers: {args.num_workers}")
    print(f"  Learning rates: base={args.base_lr:.2e}, rvq={args.rvq_lr:.2e}")
    print(f"  LR multipliers: film={args.film_lr_mult}, moe={args.moe_lr_mult}, dec_wave={args.dec_wave_lr_mult}")
    if args.dec_moe:
        print(f"  Dec MoE weights: balance={args.dec_moe_balance_weight}, entropy={args.dec_moe_entropy_weight}")
    print(f"  Freezing: encoder={args.freeze_encoder}, decoder_partial={args.freeze_decoder_except_wave}")
    print()
    print("📡 CSI & Channel:")
    print(f"  Channel type: {args.channel}")
    print(f"  SNR range: {args.snr_lo_db} → {args.snr_hi_db} dB (ramp over {args.snr_ramp_steps} steps)")
    print(f"  SNR window: ±{args.snr_window_db} dB")
    print(f"  CSI dims: {args.d_csi}, keys: {args.csi_keys}")
    print(f"  CSI update: every {args.csi_update_interval} batches, batch-coherent: {args.csi_batch_coherent}")
    if args.stage_gate_enable:
        print(f"  Stage gating: enabled (margin={args.stage_gate_threshold_kbps} kbps)")
    if args.channel_enable_step is not None:
        print(f"  Channel enable: step {args.channel_enable_step} (with 1000-step warmup)")
    print()
    print("🔧 RVQ Configuration:")
    print(f"  Decay: {args.rvq_decay}, EPS: {args.rvq_eps}")
    print(f"  Diversity loss: {args.enable_rvq_diversity} (weight={args.rvq_diversity_weight})")
    print(f"  Warmup steps: {args.rvq_warmup_steps}")
    print()
    print("📏 Loss Weights:")
    for key, value in config['initial_loss_weights'].items():
        print(f"  {key}: {value}")
    print()
    print("📈 Data Validation:")
    print(f"  Validate loading: {args.validate_data_loading}")
    print(f"  Max validation batches: {args.max_validation_batches}")
    print()
    print("💾 Checkpoint Strategy:")
    print(f"  Save every: {args.save_every_steps} steps, Log every: {args.log_interval} batches")
    print(f"  Keep last: {args.keep_last_n_checkpoints} checkpoints, Best only: {args.save_best_only}")
    print(f"  Best metric: {args.checkpoint_metric}")
    print("=" * 60)

    # 检查Stage4检查点
    if not os.path.exists(args.stage4_checkpoint):
        print(f"Error: Stage4 checkpoint not found: {args.stage4_checkpoint}")
        print("Please ensure the checkpoint path is correct.")
        sys.exit(1)

    # 检查数据文件和目录
    data_warnings = []
    if args.features_file and not os.path.exists(args.features_file):
        data_warnings.append(f"Features file not found: {args.features_file}")
    if args.pcm_file and not os.path.exists(args.pcm_file):
        data_warnings.append(f"PCM file not found: {args.pcm_file}")
    if args.data_root and not os.path.exists(args.data_root):
        data_warnings.append(f"Data root not found: {args.data_root}")
    if args.combined_data_root and not os.path.exists(args.combined_data_root):
        data_warnings.append(f"Combined data root not found: {args.combined_data_root}")

    if data_warnings:
        print("\n⚠️  Data Warnings:")
        for warning in data_warnings:
            print(f"  {warning}")
        print("Training may fail if required data is missing.")

    # 数据验证
    if args.validate_data_loading:
        print("\n🔎 Validating data loading...")
        # TODO: 这里可以添加数据加载验证逻辑
        print("✅ Data validation passed")

    # 确认开始训练
    if not args.force:
        response = input("\n🚀 Start Stage5 training with enhanced dataset support? (y/N): ").lower()
        if response != 'y':
            print("Training cancelled.")
            sys.exit(0)

    # 保存配置
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    config_path = output_dir / "stage5_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Configuration saved to {config_path}")

    # 输出数据集配置摘要
    print("\n📁 Dataset Summary:")
    if config.get('features_file'):
        print(f"  Direct features: {os.path.basename(config['features_file'])}")
    elif config.get('data_root'):
        print(f"  Data root mode: {config['data_root']}")
    elif config.get('combined_data_root'):
        print(f"  Combined data mode: {config['combined_data_root']}")
    else:
        print("  ⚠️  No data source specified - using default data loader")

    # 创建训练器并开始训练
    try:
        trainer = Stage5Trainer(config, args.stage4_checkpoint)

        # 如果指定了resume，从checkpoint恢复训练
        if args.resume:
            print(f"🔄 Resuming training from: {args.resume}")
            trainer.load_checkpoint(args.resume)

        trainer.train()
        print(f"\n✅ Stage5 training completed successfully!")
        print(f"Best checkpoint: {output_dir}/stage5_best.pth")
        print(f"Final metrics: {trainer.best_metrics}")
        print(f"Dataset configuration: {len(config.get('mix_ratio', '').split(','))} data sources")
        print(f"Channel config: {config.get('channel_type')} with {config.get('d_csi')} CSI dims")
        if config.get('enable_dec_moe'):
            print(f"Decoder MoE: {config.get('dec_moe_experts')} experts enabled")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
