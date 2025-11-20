# -*- coding: utf-8 -*-
"""
Simplified AETHER Integration: clean encoder/decoder aligned with jiagou.md.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .enhanced_gla import GLABackbone, GLABlock
    from .enhanced_moe import CompatibleMicroMoE
    from .enhanced_dual_stream import DualStream
    from .decoder_sculptor import ConvRefineDecoder
    from .kan_field import KANLiteFiLM
    from .utils import build_csi_vec, global_pool, straight_through_latent_quantizer, estimated_bitrate
except ImportError:  # pragma: no cover
    from enhanced_gla import GLABackbone, GLABlock
    from enhanced_moe import CompatibleMicroMoE
    from enhanced_dual_stream import DualStream
    from decoder_sculptor import ConvRefineDecoder
    from kan_field import KANLiteFiLM
    from utils import build_csi_vec, global_pool, straight_through_latent_quantizer, estimated_bitrate


class AETHEREncoder(nn.Module):
    """Simplified AETHER Encoder aligned with jiagou.md."""
    def __init__(
        self,
        d_in: int = 36,
        d_model: int = 128,
        dz: int = 24,
        d_csi: int = 10,
        use_film: bool = True,
        film_position: str = 'pre',  # 'pre' | 'post' | 'both'
        # FiLM temporal smoothing & schedules
        film_temporal_mode: str = 'conv',
        film_kernel_size: int = 5,
        film_pre_start: float = 0.0,
        film_pre_end: float = 0.2,
        film_pre_warmup: int = 1000,
        film_post_start: float = 0.2,
        film_post_end: float = 0.6,
        film_post_warmup: int = 0,
        film_tv_lambda: float = 1e-4,
        use_moe: bool = True,
        use_quantization: bool = False,
        latent_bits: int = 4,
        moe_router_use_csi: bool = True,
        use_semantic_head: bool = False,
        semantic_dim: int = 6,
        semantic_source: str = 'fused',
        n_experts: int = 4,  # Stage3: 支持动态专家数量
        top_k: int = 2,      # Stage3: 支持TOP-K配置
        # MoE训练参数
        moe_balance_weight: float = 0.1,
        expert_dropout: float = 0.0,
        router_jitter: float = 0.01,
        # 直流通路相关参数
        enable_direct_pathway: bool = False,
        initial_bypass_weight: float = 0.1,  # 降低直流通路初始权重
        adaptive_threshold: float = 0.15,    # 提高阈值，减少直流干预
        pathway_warmup_steps: int = 2000,    # 增加warmup步数
        # 输入分流：将前20维映射给Ribbon（coarse），后16维映射给Thread（fine）
        split_stream_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.d_csi = d_csi
        self.use_film = use_film
        self.use_moe = use_moe
        self.film_position = film_position if use_film else 'none'
        self.use_quantization = use_quantization
        self.latent_bits = latent_bits
        self.moe_router_use_csi = moe_router_use_csi
        self.use_semantic_head = use_semantic_head
        self.semantic_dim = semantic_dim
        self.semantic_source = semantic_source

        self.in_proj = nn.Linear(d_in, d_model)
        self.split_stream_inputs = bool(split_stream_inputs)
        if self.split_stream_inputs:
            # 独立的输入映射：20维（声学）→ Ribbon；16维（语义）→ Thread
            self.in_proj_ribbon = nn.Linear(20, d_model)
            self.in_proj_thread = nn.Linear(16, d_model)
        else:
            self.in_proj_ribbon = None
            self.in_proj_thread = None
        self.film = KANLiteFiLM(d_csi=d_csi, d_feat=d_model) if use_film else None
        self.backbone = DualStream(
            d_model=d_model,
            use_semantic_head=use_semantic_head,
            semantic_dim=semantic_dim,
            semantic_source=semantic_source,
        )
        # Use CompatibleMicroMoE wrapper with specialized experts and optional direct pathway
        self.moe = (
            CompatibleMicroMoE(
                d_model=d_model,
                n_experts=n_experts,  # 使用动态专家数量
                top_k=top_k,          # 使用配置的TOP-K
                d_csi=d_csi,
                router_use_csi=moe_router_use_csi,
                expert_dropout=expert_dropout,
                balance_weight=moe_balance_weight,
                use_token_level=True,  # 启用token-level routing
                # 直流通路参数
                enable_direct_pathway=enable_direct_pathway,
                initial_bypass_weight=initial_bypass_weight,
                adaptive_threshold=adaptive_threshold,
                pathway_warmup_steps=pathway_warmup_steps,
            )
            if use_moe
            else None
        )
        self.out_proj = nn.Linear(d_model, dz)

    def forward(
        self,
        x: torch.Tensor,
        csi_dict: Optional[Dict[str, torch.Tensor]] = None,
        training_step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, _ = x.shape
        logs: Dict[str, Any] = {}

        # ---- Numerical stability guards (especially under AMP) ----
        in_dtype = x.dtype
        # Early steps are most unstable under fp16/bf16; run encoder core in fp32 briefly
        use_safe_fp32 = (in_dtype in (torch.float16, torch.bfloat16)) and (int(training_step) < 200)
        # Sanitize inputs to avoid propagating NaN/Inf from data
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0)
        if use_safe_fp32:
            x_safe = x_safe.float()

        # Input projection
        h = self.in_proj(x_safe)  # [B,T,36] -> [B,T,128]
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # CSI-based FiLM modulation (pre)
        a = b = None
        if self.use_film and self.film is not None and csi_dict is not None:
            csi_vec = build_csi_vec(csi_dict, target_dim=self.d_csi)  # [B, d_csi]
            # Ensure CSI is finite and on matching dtype for FiLM computation
            csi_vec = torch.nan_to_num(csi_vec, nan=0.0, posinf=1.0, neginf=-1.0)
            if use_safe_fp32:
                csi_vec = csi_vec.float()
            a, b = self.film(csi_vec, T=t)  # [B,T,D]
            a = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
            b = torch.nan_to_num(b, nan=0.0, posinf=1.0, neginf=-1.0)
            # Temporal smoothing (causal moving average) if enabled
            if hasattr(self, 'film_smooth_kernel') and self.film_smooth_kernel.numel() > 1:
                def _smooth(x: torch.Tensor) -> torch.Tensor:
                    # x: [B,T,D] -> [B,D,T]
                    xd = x.permute(0, 2, 1)
                    pad = torch.nn.functional.pad(xd, (self.film_smooth_kernel.size(-1) - 1, 0))
                    y = torch.nn.functional.conv1d(pad, self.film_smooth_kernel, padding=0, groups=self.d_model)
                    return y.permute(0, 2, 1)
                a = _smooth(a)
                b = _smooth(b)
            # Strength schedules (pre/post)
            def _ramp(step: int, start: float, end: float, warmup: int) -> float:
                if warmup is None or warmup <= 0:
                    return end
                r = max(0.0, min(1.0, float(step) / float(warmup)))
                return start + (end - start) * r
            pre_s = _ramp(training_step, getattr(self, 'film_pre_start', 0.0), getattr(self, 'film_pre_end', 0.2), getattr(self, 'film_pre_warmup', 1000))
            post_s = _ramp(training_step, getattr(self, 'film_post_start', 0.2), getattr(self, 'film_post_end', 0.6), getattr(self, 'film_post_warmup', 0))
            # Numeric constraints for stability
            scale = (0.4 + 1.2 * torch.sigmoid(a.float())).to(h.dtype)  # [0.4,1.6]
            shift = (0.3 * torch.tanh(b.float())).to(h.dtype)          # limited shift
            scale = torch.nan_to_num(scale, nan=1.0, posinf=1.6, neginf=0.4).clamp_(0.25, 2.0)
            shift = torch.nan_to_num(shift, nan=0.0, posinf=0.3, neginf=-0.3).clamp_(-0.4, 0.4)
            if self.film_position in ('pre', 'both'):
                # pre: gain only, small strength
                h = (1.0 + pre_s * (scale - 1.0)) * h
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

            # Expose lightweight FiLM diagnostics for JSCC/Stage-4 monitoring (no grad ops)
            try:
                self._last_film_stats = {
                    'training_step': int(training_step),
                    'position': self.film_position,
                    'pre_s': float(pre_s),
                    'post_s': float(post_s),
                    'a_mean': float(a.detach().mean().item()) if a is not None else 0.0,
                    'b_mean': float(b.detach().mean().item()) if b is not None else 0.0,
                    'scale_mean': float(scale.detach().mean().item()),
                    'shift_mean': float(shift.detach().mean().item()),
                }
            except Exception:
                pass

        # DualStream backbone (returns h and stream separation)
        # Backbone (optionally in safe fp32 for early steps)
        if use_safe_fp32:
            # Optional split inputs for DualStream streams
            ribbon_in = None
            thread_in = None
            if self.split_stream_inputs and (self.in_proj_ribbon is not None) and (self.in_proj_thread is not None):
                # 使用原始输入x切片构造分流输入
                ribbon_in = self.in_proj_ribbon(x_safe[..., :20].float())
                thread_in = self.in_proj_thread(x_safe[..., 20:36].float())
            # Run backbone in fp32 for stability then cast back
            out = self.backbone(h.float(), ribbon_input=ribbon_in, thread_input=thread_in)
            if isinstance(out, tuple):
                h_tmp, aux_tmp = out
                h = h_tmp.to(x_safe.dtype)
                aux = {k: (v.to(x_safe.dtype) if isinstance(v, torch.Tensor) else v) for k, v in aux_tmp.items()}
                out = (h, aux)
            else:
                out = out.to(x_safe.dtype)
        else:
            ribbon_in = None
            thread_in = None
            if self.split_stream_inputs and (self.in_proj_ribbon is not None) and (self.in_proj_thread is not None):
                ribbon_in = self.in_proj_ribbon(x[..., :20])
                thread_in = self.in_proj_thread(x[..., 20:36])
            out = self.backbone(h, ribbon_input=ribbon_in, thread_input=thread_in)
        if isinstance(out, tuple):
            h, aux = out
            # 提取DualStream的分离输出
            dual_streams = {
                'ribbon_stream': aux.get('ribbon_stream', h),
                'thread_stream': aux.get('thread_stream', h),
                'fused_stream': aux.get('fused_stream', h),
            }
            if 'semantic_pred' in aux:
                logs['semantic_pred'] = aux['semantic_pred']
        else:
            h = out
            # 没有分离流时使用融合流作为fallback
            dual_streams = {
                'ribbon_stream': h,
                'thread_stream': h,
                'fused_stream': h,
            }

        # 专业化MoE bottleneck - 基于纯音频特征的路由
        if self.use_moe and self.moe is not None:
            # 新设计：MoE路由完全基于音频内容，不使用CSI
            # LowSNR专家分析原始音频的信噪比，而非信道状态
            csi_vec = None  # 移除CSI依赖

            # 双层特征MoE：传递原始特征x和编码特征h
            if hasattr(self.moe, 'specialized_moe'):
                # 使用CompatibleMicroMoE: 构造声学特征用于路由
                # 预先计算acoustic features (与SpecializedMicroMoE内部一致)
                ribbon_stream = dual_streams.get('ribbon_stream', h)  # [B,T,128]
                thread_stream = dual_streams.get('thread_stream', h)  # [B,T,128]
                # 兼容EnhancedMicroMoEWithBypass和CompatibleMicroMoE
                if hasattr(self.moe, 'acoustic_extractor'):
                    # EnhancedMicroMoEWithBypass case
                    acoustic_features = self.moe.acoustic_extractor(x, ribbon_stream, thread_stream)  # [B,64]
                elif hasattr(self.moe, 'specialized_moe') and hasattr(self.moe.specialized_moe, 'acoustic_extractor'):
                    # CompatibleMicroMoE case
                    acoustic_features = self.moe.specialized_moe.acoustic_extractor(x, ribbon_stream, thread_stream)  # [B,64]
                else:
                    # Fallback: skip acoustic features
                    acoustic_features = torch.zeros(b, 64, device=h.device, dtype=h.dtype)

                # 纯音频特征路由：不拼接CSI，专家基于音频内容特化
                router_input = acoustic_features  # [B,64] 纯声学特征
                # 传递训练步数给MoE系统（用于温度退火）
                if hasattr(self.moe, 'set_training_step'):
                    self.moe.set_training_step(training_step)

                # 传递原始特征x和DualStream分离输出用于三层声学分析
                # 早期强制在fp32跑MoE，避免AMP下的RNN数值不稳定
                if use_safe_fp32:
                    h_moe, aux = self.moe(h.float(), router_input.float(), x_raw=x_safe, dual_streams={k: v.float() for k, v in dual_streams.items()}, training_step=training_step)
                    h_moe = h_moe.to(x_safe.dtype)
                    aux = {k: (v.to(x_safe.dtype) if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}
                else:
                    h_moe, aux = self.moe(h, router_input, x_raw=x, dual_streams=dual_streams, training_step=training_step)  # [B,T,128], aux losses/metrics
                # 把 MoE 的辅助损失并入 logs（保持 Tensor，不要转 float）
                if logs is None:
                    logs = {}
                for k, v in aux.items():
                    # 只保留 Tensor；其余统计/字符串可忽略或另存
                    if isinstance(v, torch.Tensor):
                        logs[k] = v  # 不要 .item()，保持可反传
                # 后续用 h_moe 继续流程
                h = h_moe

            else:
                # 直接使用SpecializedMicroMoE接口，传递三层特征
                h_moe = self.moe(h, csi_vec, x_raw=x, dual_streams=dual_streams, training_step=training_step)  # [B,T,128]
                aux = self.moe.get_aux_losses(h, csi_vec)  # Get aux losses separately
            h = h + h_moe  # Residual connection

            # CSI-based FiLM modulation (post)
            if self.use_film and self.film is not None and a is not None and self.film_position in ('post', 'both'):
                h = (1.0 + post_s * (scale - 1.0)) * h + post_s * shift
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

            # Log enhanced MoE auxiliary losses/metrics
            if isinstance(aux, dict):
                # Balance losses
                if 'balance_loss' in aux:
                    logs['moe_balance_loss'] = aux['balance_loss']
                if 'token_balance_loss' in aux:
                    logs['moe_token_balance_loss'] = aux['token_balance_loss']

                # Expert utilization metrics
                if 'expert_entropy' in aux:
                    logs['moe_expert_entropy'] = aux['expert_entropy']
                if 'min_expert_usage' in aux:
                    logs['moe_min_expert_usage'] = aux['min_expert_usage']

                # Expert specialization monitoring
                if 'harmonic_preference' in aux:
                    logs['moe_harmonic_pref'] = aux['harmonic_preference']
                if 'transient_preference' in aux:
                    logs['moe_transient_pref'] = aux['transient_preference']
                if 'continuity_preference' in aux:
                    logs['moe_continuity_pref'] = aux['continuity_preference']
                if 'snr_preference' in aux:
                    logs['moe_snr_pref'] = aux['snr_preference']

            # 直流通路性能统计（仅增强版本）
            if hasattr(self.moe, 'get_performance_stats'):
                stats = self.moe.get_performance_stats()
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        logs[f'pathway_{key}'] = torch.tensor(value, device=h.device)
                    elif isinstance(value, list) and len(value) > 0:
                        logs[f'pathway_{key}'] = torch.tensor(value, device=h.device)

        # Output projection
        z = self.out_proj(h)  # [B,T,24]
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)

        # Cast back to original dtype if we elevated precision
        if use_safe_fp32 and in_dtype in (torch.float16, torch.bfloat16):
            z = z.to(in_dtype)

        # Optional quantization
        if self.use_quantization:
            z = straight_through_latent_quantizer(z, bits=self.latent_bits)

        # Optional FiLM temporal regularisation (TV on a and b)
        if self.use_film and a is not None and getattr(self, 'film_tv_lambda', 0.0) > 0.0:
            tv = (a[:, 1:] - a[:, :-1]).abs().mean() + (b[:, 1:] - b[:, :-1]).abs().mean()
            logs['film_tv_loss'] = tv * self.film_tv_lambda

        logs["estimated_bitrate_kbps"] = estimated_bitrate(z)
        return z, logs


class AETHERDecoder(nn.Module):
    """Simplified AETHER Decoder aligned with jiagou.md."""
    def __init__(self, dz: int = 24, d_out: int = 36, d_hidden: int = 128, d_csi: int = 10, use_film: bool = True) -> None:
        super().__init__()
        self.d_csi = d_csi
        self.use_film = use_film

        # Global fusion with simple GLA
        self.global_recompose = GLABlock(d_model=dz, heads=2, local_kernel=3, dilation=1)

        # ConvRefine decoder
        self.refiner = ConvRefineDecoder(d_z=dz, d_out=d_out, d_hidden=d_hidden, csi_dim=d_csi, use_film=use_film)

    def forward(self, z: torch.Tensor, csi_dict: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logs: Dict[str, Any] = {}

        # Global recomposition
        z = self.global_recompose(z)  # [B,T,24]

        # CSI-aware refinement
        if csi_dict is not None:
            csi_vec = build_csi_vec(csi_dict, target_dim=self.d_csi)  # [B,10]
        else:
            # Create zero CSI vector when not provided
            csi_vec = torch.zeros(z.size(0), self.d_csi, device=z.device, dtype=z.dtype)

        y = self.refiner(z, csi_vec)  # [B,T,36]

        return y, logs


def create_aether_codec(config: Optional[Dict] = None) -> Tuple[AETHEREncoder, AETHERDecoder]:
    """Create simplified AETHER codec aligned with jiagou.md."""
    cfg = {
        "d_in": 36,
        "d_model": 128,
        "dz": 24,
        "d_csi": 10,
        "use_film": True,
        "use_moe": True,
        "use_quantization": False,
        "latent_bits": 4,
        "moe_router_use_csi": True,
        "use_semantic_head": False,
        "semantic_dim": 6,
        "semantic_source": 'fused',
        "n_experts": 4,  # 默认4专家，Stage3使用4专家系统
        "top_k": 2,      # 默认TOP-2路由
        "moe_balance_weight": 0.5,  # 默认平衡权重
        "expert_dropout": 0.1,      # 默认专家dropout
        "router_jitter": 0.0        # 默认无路由抖动
    }
    if config:
        cfg.update(config)

    enc = AETHEREncoder(**cfg)
    dec = AETHERDecoder(dz=cfg["dz"], d_out=cfg["d_in"], d_hidden=cfg["d_model"], d_csi=cfg["d_csi"], use_film=cfg["use_film"])
    return enc, dec


if __name__ == "__main__":
    pass
