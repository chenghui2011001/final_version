# -*- coding: utf-8 -*-
"""
AETHER-FARGAN混合解码器
结合AETHER的强大编码能力与FARGAN的优秀波形生成能力
继承并扩展原始AETHERDecoder架构
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import math

try:
    # Prefer package-relative imports
    from ..aether_encoder_decoder import AETHERDecoder
    from ..feature_adapter import Feature48To36Adapter, FARGANFeatureSpec
    from ..fargan_components import FARGANCore
except Exception:
    # Fallback to absolute package imports if run with sys.path adjusted
    from models.aether_encoder_decoder import AETHERDecoder
    from models.feature_adapter import Feature48To36Adapter, FARGANFeatureSpec
    from models.fargan_components import FARGANCore

try:
    # Prefer top-level 'utils' namespace under final_version
    from utils.feature_spec import get_default_feature_spec
except Exception:
    # Fallback to fully qualified import if package layout is present
    from dnn.torch.final_version.utils.feature_spec import get_default_feature_spec


class PeriodEstimator(nn.Module):
    """从FARGAN特征估计周期序列"""

    def __init__(self):
        super().__init__()
        self.period_proj = nn.Sequential(
            nn.Linear(20, 32),  # 只用前20维特征
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, fargan_features: torch.Tensor) -> torch.Tensor:
        """
        从FARGAN特征估计周期

        Args:
            fargan_features: [B, T, 36] FARGAN特征

        Returns:
            period: [B, T] 周期序列 (32-255)
        """
        # Stage3一致的映射：由 dnn_pitch → f0_hz → period
        # 训练脚本中使用：f0_hz = sr * 2**(dnn_pitch - 6.5)，再 period = round(16000/f0)
        dnn_pitch = fargan_features[..., 18:19]  # [B, T, 1]
        f0_hz = (16000.0 * torch.pow(2.0, dnn_pitch - 6.5)).clamp(50.0, 400.0)
        period_raw = (16000.0 / f0_hz).clamp(32.0, 255.0)
        period = torch.round(period_raw).to(torch.long).squeeze(-1)

        return period


class AETHERFARGANDecoder(AETHERDecoder):
    """AETHER-FARGAN混合解码器

    继承AETHERDecoder，直接工作在36-dim FARGAN特征域
    移除特征适配器，简化架构
    """

    def __init__(
        self,
        dz: int = 24,
        d_out: int = 36,  # 直接输出36维FARGAN特征
        d_hidden: int = 128,
        d_csi: int = 32,
        decoder_heads: int = 2,
        enable_synth: bool = True,  # 默认启用FARGAN合成
        fargan_subframe_size: int = 40,
        fargan_nb_subframes: int = 4,
        frame_rate_hz: float = 100.0,
        feature_spec_type: str = "fargan",
        use_film: bool = False,
        enable_output_calibration: bool = False,
        enable_ceps_affine_calib: bool = True,
    ):
        # 初始化父类AETHERDecoder，直接使用FARGAN特征规范
        super().__init__(
            dz=dz,
            d_out=d_out,
            d_hidden=d_hidden,
            d_csi=d_csi,
            decoder_heads=decoder_heads,
            enable_synth=False,  # 禁用原有的OLA合成器
            feature_spec_type=feature_spec_type,
            use_film=use_film,
        )

        # FARGAN专用参数
        self.enable_fargan_synth = enable_synth
        self.frame_rate_hz = frame_rate_hz
        self.fargan_frame_size = fargan_subframe_size * fargan_nb_subframes  # 160
        self.enable_output_calibration = bool(enable_output_calibration)
        self.enable_ceps_affine_calib = bool(enable_ceps_affine_calib)

        # 逐维仿射校准（仅作用于 ceps[0..17]），训练/推理一致
        if self.enable_ceps_affine_calib:
            self.register_parameter('ceps_gamma', nn.Parameter(torch.ones(18)))
            self.register_parameter('ceps_beta', nn.Parameter(torch.zeros(18)))
        else:
            self.register_parameter('ceps_gamma', nn.Parameter(torch.ones(18), requires_grad=False))
            self.register_parameter('ceps_beta', nn.Parameter(torch.zeros(18), requires_grad=False))

        # 可选：对完整36维特征做仿射校准（解码端“去白化/标定”）
        if self.enable_output_calibration:
            self.register_parameter('feat_gamma_36', nn.Parameter(torch.ones(36)))
            self.register_parameter('feat_beta_36', nn.Parameter(torch.zeros(36)))
        else:
            self.register_parameter('feat_gamma_36', nn.Parameter(torch.ones(36), requires_grad=False))
            self.register_parameter('feat_beta_36', nn.Parameter(torch.zeros(36), requires_grad=False))

        # 周期估计器
        self.period_estimator = PeriodEstimator()

        # 轻量周期平滑以抑制蜂鸣/嗡嗡伪影（默认启用3帧中值）
        self.period_smooth_ks: int = 3     # 奇数，>=1；1 表示不平滑
        self.period_smooth_mode: str = 'median'  # 'median' 或 'mean'

        # FARGAN核心合成器
        if enable_synth:
            self.fargan_core = FARGANCore(
                subframe_size=fargan_subframe_size,
                nb_subframes=fargan_nb_subframes,
                feature_dim=20,  # 使用前20维特征
                cond_size=256
            )

        print(f"🚀 AETHER-FARGAN解码器初始化完成")
        print(f"   - 基础架构: AETHERDecoder (GLABlock + ConvRefineDecoder)")
        print(f"   - 潜在维度: {dz}")
        print(f"   - 输出维度: {d_out} (FARGAN特征)")
        print(f"   - 特征规范: {feature_spec_type}")
        print(f"   - FARGAN合成: {'启用' if enable_synth else '禁用'}")
        print(f"   - 帧率: {frame_rate_hz} Hz")
        print(f"   - FARGAN帧大小: {self.fargan_frame_size} 样本")

        # --- Decoder-side Residual MoE (token-level Top-K) ---
        self.enable_dec_moe: bool = True
        self.dec_moe = DecoderResidualMoE(d_in=d_out, hidden=64, n_experts=3, top_k=2)

    def _forward_features(
        self,
        z: torch.Tensor,
        csi_dict: Dict[str, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，直接输出36维FARGAN特征

        Returns:
            fargan_features: [B, T, 36] FARGAN特征
        """
        # 直接使用父类方法，现在已经配置为输出36维特征
        fargan_features = super()._forward_features(z, csi_dict, attn_mask)
        # 全维仿射校准（先整体再对ceps子段细化）
        try:
            if self.enable_output_calibration:
                gamma36 = self.feat_gamma_36.to(fargan_features.device, fargan_features.dtype).view(1, 1, -1)
                beta36 = self.feat_beta_36.to(fargan_features.device, fargan_features.dtype).view(1, 1, -1)
                fargan_features = fargan_features * gamma36 + beta36
        except Exception:
            pass
        return fargan_features

    def _estimate_period(self, fargan_features: torch.Tensor) -> torch.Tensor:
        """
        从FARGAN特征估计周期

        Args:
            fargan_features: [B, T, 36] FARGAN特征

        Returns:
            period: [B, T] 周期序列
        """
        period = self.period_estimator(fargan_features)
        return period

    def _generate_waveform(
        self,
        fargan_features: torch.Tensor,
        period: torch.Tensor,
        target_len: Optional[int] = None,
        fargan_pre: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用FARGAN核心生成波形

        Args:
            fargan_features: [B, T, 36] FARGAN特征
            period: [B, T] 周期序列
            target_len: 目标音频长度
            fargan_pre: [B, 1, L] 可选的前序音频，用于teacher forcing

        Returns:
            audio: [B, 1, L] 生成的音频波形
        """
        if not self.enable_fargan_synth:
            raise RuntimeError("FARGAN合成器未启用")

        B, T, _ = fargan_features.shape

        # 计算需要生成的帧数（严格对齐 models/fargan_decoder.py 的逻辑）
        # 条件网络时间维收缩：T -> T-4，可用帧数 = T-4。
        max_available_frames = T - 4
        # 教师强制预热帧数（若提供pre），注意 FARGANCore 里会在前 nb_pre_frames 步直接使用pre，不计入输出
        nb_pre_frames = 0
        pre = fargan_pre
        if pre is not None:
            if pre.dim() == 3:  # [B,1,L] -> [B,L]
                pre = pre.squeeze(1)
            nb_pre_frames = pre.size(1) // self.fargan_frame_size

        gen_capacity = max(0, max_available_frames - nb_pre_frames)
        if target_len is not None:
            target_frames_total = (target_len + self.fargan_frame_size - 1) // self.fargan_frame_size
            target_frames_gen = max(0, target_frames_total - nb_pre_frames)
            nb_frames = min(gen_capacity, target_frames_gen)
        else:
            nb_frames = gen_capacity

        # 至少生成1帧
        nb_frames = max(1, nb_frames)

        # 轻量时间平滑周期，抑制少量帧抖动导致的窄带伪影
        try:
            ks = int(getattr(self, 'period_smooth_ks', 1) or 1)
            mode = str(getattr(self, 'period_smooth_mode', 'median') or 'median')
            if ks > 1 and ks % 2 == 1 and period.dim() == 2:
                import torch.nn.functional as F
                pad = (ks - 1) // 2
                p = period.to(torch.float32)
                p = F.pad(p.unsqueeze(1), (pad, pad), mode='replicate').squeeze(1)  # [B, T+2pad]
                win = p.unfold(dimension=1, size=ks, step=1)                         # [B, T, ks]
                if mode == 'mean':
                    ps = win.mean(dim=-1)
                else:
                    ps, _ = win.median(dim=-1)
                period = ps.round().clamp(32, 255).to(torch.long)
        except Exception:
            pass

        # 使用前20维特征进行FARGAN合成
        features_20 = fargan_features[..., :20]

        # FARGAN生成 (支持teacher forcing)
        audio, _ = self.fargan_core(features_20, period, nb_frames, pre=pre)

        # 确保输出形状为 [B, 1, L]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # 裁剪到目标长度
        if target_len is not None and audio.size(-1) > target_len:
            audio = audio[..., :target_len]

        return audio

    def forward(
        self,
        z: torch.Tensor,
        csi_dict: Dict[str, torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        return_wave: bool = False,
        target_len: Optional[int] = None
    ):
        """
        前向传播，保持与AETHERDecoder兼容的接口

        Args:
            z: [B, T, dz] 潜在表示
            csi_dict: CSI字典
            attn_mask: 注意力掩码
            return_wave: 是否返回波形
            target_len: 目标音频长度

        Returns:
            如果return_wave=False: fargan_features [B, T, 36]
            如果return_wave=True: (fargan_features [B, T, 36], audio [B, 1, L])
        """
        # 步骤1: 直接重建36维FARGAN特征
        fargan_features = super()._forward_features(z, csi_dict, attn_mask)
        # 缓存校准前原始输出，供损失在raw空间进行统计
        try:
            self._last_raw_features = fargan_features.detach()
        except Exception:
            self._last_raw_features = None
        # 缓存第0维特征分布（供训练脚本迭代打印）
        try:
            feat0 = fargan_features[..., 0]
            self._last_feat0_stats = {
                'mean': float(feat0.mean().detach().cpu().item()),
                'std':  float(feat0.std().detach().cpu().item()),
                'min':  float(feat0.min().detach().cpu().item()),
                'max':  float(feat0.max().detach().cpu().item()),
            }
        except Exception:
            pass

        # 训练/推理一致的 ceps 段仿射校准（只作用于特征输出，非合成专用）
        try:
            if self.enable_ceps_affine_calib:
                from models.feature_adapter import FARGANFeatureSpec
                ceps_slice = FARGANFeatureSpec.get_feature_slice('ceps')
                out_ceps = fargan_features[..., ceps_slice]
                # 应用逐维仿射：y = gamma * x + beta
                gamma = self.ceps_gamma.to(out_ceps.device, out_ceps.dtype).view(1, 1, -1)
                beta = self.ceps_beta.to(out_ceps.device, out_ceps.dtype).view(1, 1, -1)
                out_ceps = out_ceps * gamma + beta
                fargan_features = fargan_features.clone()
                fargan_features[..., ceps_slice] = out_ceps
                # 正则化值缓存（供loss读取）
                # 记录整体与ceps的正则（有助于损失端轻微约束）
                reg_ceps = (self.ceps_gamma - 1.0).pow(2).mean() + self.ceps_beta.pow(2).mean()
                try:
                    reg_full = (self.feat_gamma_36 - 1.0).pow(2).mean() + self.feat_beta_36.pow(2).mean()
                except Exception:
                    reg_full = torch.tensor(0.0, device=fargan_features.device)
                self._last_calib_reg = reg_full + reg_ceps
            else:
                self._last_calib_reg = None
        except Exception:
            self._last_calib_reg = None

        # 可选：输出端线性校准（仅用于合成路径），将 ceps[0:18] 校准到目标 batch 统计
        if self.enable_output_calibration and csi_dict is not None:
            try:
                from models.feature_adapter import FARGANFeatureSpec
                ceps_slice = FARGANFeatureSpec.get_feature_slice('ceps')
                tgt_mean = csi_dict.get('calib_ceps_mean', None)
                tgt_std = csi_dict.get('calib_ceps_std', None)
                if torch.is_tensor(tgt_mean) and torch.is_tensor(tgt_std):
                    # 计算输出 ceps 的当前统计
                    out_ceps = fargan_features[..., ceps_slice]  # [B,T,18]
                    mu = out_ceps.mean(dim=(0, 1))               # [18]
                    sd = out_ceps.std(dim=(0, 1)).clamp_min(1e-4)
                    # 标准化再匹配到目标 batch
                    out_ceps_norm = (out_ceps - mu.view(1, 1, -1)) / sd.view(1, 1, -1)
                    out_ceps_cal = out_ceps_norm * tgt_std.view(1, 1, -1) + tgt_mean.view(1, 1, -1)
                    fargan_features = fargan_features.clone()
                    fargan_features[..., ceps_slice] = out_ceps_cal
                    # 额外：frame_corr 夹紧到规范范围
                    fc_slice = FARGANFeatureSpec.get_feature_slice('frame_corr')
                    fargan_features[..., fc_slice] = fargan_features[..., fc_slice].clamp(-0.8, 0.5)
                    # 缓存校准统计
                    self._last_calib_stats = {
                        'out_mu': mu.detach().cpu().tolist(),
                        'out_sd': sd.detach().cpu().tolist(),
                        'tgt_mu': tgt_mean.detach().cpu().tolist(),
                        'tgt_sd': tgt_std.detach().cpu().tolist(),
                    }
            except Exception:
                pass

        # 步骤1.5: 残差MoE结构性补偿（可选）。如果启用，传入CSI并捕获辅助正则loss。
        if self.enable_dec_moe and self.dec_moe is not None:
            delta, aux = self.dec_moe(fargan_features, csi_dict=csi_dict, return_aux=True)
            # 暂存辅助loss和统计，供训练循环读取并加入总loss（不改动forward返回签名）
            try:
                self._dec_moe_aux = aux
                self._dec_moe_aux_loss = aux.get('loss', None)
            except Exception:
                pass
            fargan_features = fargan_features + delta

        if not return_wave:
            # 如果不需要波形，返回FARGAN特征
            return fargan_features

        # 步骤2: 估计周期
        period = self._estimate_period(fargan_features)

        # 步骤3: 生成波形
        if self.enable_fargan_synth:
            # 从csi_dict中提取fargan_pre参数（如果存在）
            fargan_pre = csi_dict.get('fargan_pre') if csi_dict else None
            audio = self._generate_waveform(fargan_features, period, target_len, fargan_pre)
            # 将FARGAN核心的最近一次增益统计缓存到decoder
            try:
                if hasattr(self, 'fargan_core') and hasattr(self.fargan_core, '_last_gain_stats'):
                    self._last_gain_stats = dict(self.fargan_core._last_gain_stats)
            except Exception:
                pass
            return fargan_features, audio
        else:
            # 如果FARGAN合成器未启用，返回零音频
            audio_len = target_len or (fargan_features.size(1) * self.fargan_frame_size)
            audio = torch.zeros(fargan_features.size(0), 1, audio_len,
                              device=fargan_features.device, dtype=fargan_features.dtype)
            return fargan_features, audio

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'model_type': 'AETHER-FARGAN混合解码器',
            'base_architecture': 'AETHERDecoder',
            'latent_dim': self.dz,
            'output_feature_dim': self.d_out,  # 直接输出36维FARGAN特征
            'csi_dim': self.d_csi,
            'fargan_frame_size': self.fargan_frame_size,
            'frame_rate_hz': self.frame_rate_hz,
            'fargan_synth_enabled': self.enable_fargan_synth,
            'decoder_residual_moe': {
                'enabled': bool(self.enable_dec_moe),
                'n_experts': int(getattr(self.dec_moe, 'n_experts', 0)) if hasattr(self, 'dec_moe') else 0,
                'top_k': int(getattr(self.dec_moe, 'top_k', 0)) if hasattr(self, 'dec_moe') else 0,
            },
            'components': {
                'global_recompose': 'GLABlock (继承自AETHERDecoder)',
                'refiner': 'ConvRefineDecoder (继承自AETHERDecoder)',
                'period_estimator': 'PeriodEstimator',
                'fargan_core': 'FARGANCore' if self.enable_fargan_synth else None
            },
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
        return info

    # 提供训练端用于日志的MoE统计
    def get_dec_moe_stats(self) -> Dict[str, Any]:
        if hasattr(self, 'dec_moe') and self.dec_moe is not None:
            return self.dec_moe.get_stats()
        return {}


class DecoderResidualMoE(nn.Module):
    """解码端残差MoE：在FARGAN 36维特征上进行结构性补偿（瞬态/缺口/谐波/谱不稳）。

    - Token-level Top-K 路由（默认K=2），路由输入为 r_feat = [声学描述符(6) + 可选CSI(4)]。
    - 专家为小型MLP，输出残差 δy，最终 y_out = y + δy。
    - 引入可微的负载均衡/熵正则与概率平滑，抑制专家塌缩；可选启用CSI感知的动态选择。
    """
    def __init__(self, d_in: int = 36, hidden: int = 64, n_experts: int = 3, top_k: int = 2, residual_scale: float = 0.2):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.residual_scale = float(residual_scale)
        # Schedules / warm-ups (can be overwritten by the caller)
        self.topk_warm_steps: int = 0
        self.temp_start: float = 1.5
        self.temp_end: float = 0.7
        self.temp_steps: int = 1000
        self.res_scale_start: float = residual_scale
        self.res_scale_end: float = residual_scale
        self.res_scale_steps: int = 0
        self._train_step: int = 0

        # Router增强/正则参数（可在外部覆盖）
        self.router_use_csi: bool = True          # 是否在路由中使用CSI代理
        self.balance_weight: float = 0.2          # 负载均衡损失权重
        self.entropy_weight: float = 0.05         # 熵正则权重（早期更强）
        self.entropy_warm_steps: int = 800        # 熵正则退火步数
        self.prob_smoothing_eps: float = 0.02     # 概率平滑，避免零梯度

        # Very light supervised routing: when transient is high, bias a chosen expert
        self.supervise_transient: bool = False
        self.transient_expert_id: int = 1  # default: second expert
        self.trans_thresh: float = 0.5     # on LN-normalised transient feature
        self.trans_bias: float = 0.1       # small positive bias added to logits
        self.trans_sup_steps: int = 1000   # only active in early steps

        # 路由器（输入固定为10维：6声学+4 CSI；未启用CSI时用0补齐以对齐权重）
        self.router_in_dim = 10  # [transient, continuity, harmonicness, spectral_var, energy, pitch_abs, snr, tsel, fsel, los]
        self.router_ln = nn.LayerNorm(self.router_in_dim)
        self.router_mlp = nn.Sequential(
            nn.Linear(self.router_in_dim, 16), nn.GELU(), nn.Linear(16, self.n_experts)
        )

        # 专家簇：每个专家为小型MLP，残差输出
        experts = []
        for _ in range(self.n_experts):
            experts.append(nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_in)
            ))
        self.experts = nn.ModuleList(experts)

        # 破对称与稳定性：专家门控偏置 + 轻微抖动
        self.gate_bias = nn.Parameter(torch.zeros(self.n_experts))
        nn.init.normal_(self.gate_bias, mean=0.0, std=1e-2)
        self.jitter_std: float = 1e-3

        # 统计缓存
        self._last_gate = None  # [B,T,E]
        self._last_top1 = None  # [B,T]

    def _router_features(self, y: torch.Tensor, csi_dict: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """构造路由特征 r_feat: [B,T,10] = 6(声学) + 4(CSI)。"""
        B, T, D = y.shape
        # 瞬态（帧差）
        dy = y[:, 1:, :] - y[:, :-1, :]
        trans = torch.zeros(B, T, device=y.device, dtype=y.dtype)
        trans[:, 1:] = dy.abs().mean(dim=-1)
        trans = trans.unsqueeze(-1)
        # 连续性偏差（当前与滑动均值差）
        k = 5
        pad = (k - 1) // 2
        y_pad = F.pad(y.permute(0, 2, 1), (pad, pad), mode='replicate')  # [B,D,T+2p]
        y_ma = F.avg_pool1d(y_pad, kernel_size=k, stride=1, padding=0).permute(0, 2, 1)
        cont = (y - y_ma).abs().mean(dim=-1, keepdim=True)
        # 谐波性（来自dnn_pitch与frame_corr）
        pitch = y[..., 18].clamp(-2.0, 2.0)
        pitch_abs = pitch.abs().unsqueeze(-1)
        frame_corr = y[..., 19].clamp(0.0, 1.0).unsqueeze(-1)
        harm = frame_corr  # 作为简化谐波性指标
        # 频谱不稳（LPC方差）
        lpc = y[..., 20:36]
        spec_var = lpc.var(dim=-1, keepdim=True)
        # 能量近似（ceps[0]）
        energy = y[..., 0:1]
        r_acoustic = torch.cat([trans, cont, harm, spec_var, energy, pitch_abs], dim=-1)  # [B,T,6]
        # CSI代理：snr_proxy(或从snr_db映射)、time_selectivity、freq_selectivity、los_ratio
        if self.router_use_csi and csi_dict is not None and isinstance(csi_dict, dict) and len(csi_dict) > 0:
            def _get(key: str, default: float) -> torch.Tensor:
                t = csi_dict.get(key, None)
                if t is None:
                    return torch.full((B,), float(default), device=y.device, dtype=y.dtype)
                t = t.to(y.device).to(y.dtype)
                if t.dim() == 0:
                    t = t.view(1).expand(B)
                if t.dim() == 1 and t.size(0) == B:
                    return t
                try:
                    return t.view(B)
                except Exception:
                    return t.flatten()[:B].to(y.dtype)
            if 'snr_proxy' in csi_dict:
                snr_proxy = _get('snr_proxy', 0.0)
            else:
                snr_db = _get('snr_db', 5.0)
                snr_proxy = torch.clamp((snr_db - 7.5) / 12.5, -1.5, 1.5)
            tsel = _get('time_selectivity', 0.0)
            fsel = _get('freq_selectivity', 0.0)
            losr = _get('los_ratio', 0.0)
            def _bn(v: torch.Tensor) -> torch.Tensor:
                v = torch.clamp(v, -2.0, 2.0)
                return v.view(B, 1, 1).expand(B, T, 1)
            r_csi = torch.cat([_bn(snr_proxy), _bn(tsel), _bn(fsel), _bn(losr)], dim=-1)  # [B,T,4]
        else:
            r_csi = torch.zeros(B, T, 4, device=y.device, dtype=y.dtype)
        return torch.cat([r_acoustic, r_csi], dim=-1)  # [B,T,10]

    def forward(self, y: torch.Tensor, csi_dict: Optional[Dict[str, torch.Tensor]] = None, return_aux: bool = False):
        B, T, D = y.shape
        r = self._router_features(y, csi_dict=csi_dict)  # [B,T,10]
        r = self.router_ln(r)
        logits = self.router_mlp(r)  # [B,T,E]
        # Temperature schedule
        if self.temp_steps > 0:
            s = min(1.0, float(self._train_step) / float(self.temp_steps))
            tau = self.temp_start + (self.temp_end - self.temp_start) * s
        else:
            tau = self.temp_end
        tau = max(0.1, float(tau))
        logits = logits / tau
        logits = logits + self.gate_bias.view(1, 1, -1).to(logits.dtype)
        # Optional jitter for stability in early training
        if self.training and self.jitter_std > 0:
            logits = logits + self.jitter_std * torch.randn_like(logits)

        # Very light supervised routing (transient -> chosen expert)
        if self.supervise_transient and (self._train_step < self.trans_sup_steps):
            try:
                trans_ln = r[..., 0]  # transient after LN
                mask = (trans_ln > float(self.trans_thresh)).to(logits.dtype)
                j = max(0, min(self.transient_expert_id, self.n_experts - 1))
                bias = float(self.trans_bias)
                logits[..., j] = logits[..., j] + bias * mask
            except Exception:
                pass
        probs_soft = F.softmax(logits.float(), dim=-1).to(y.dtype)
        # 概率平滑，避免某些专家完全零梯度（用于正则）
        eps = float(max(0.0, min(0.1, self.prob_smoothing_eps)))
        if eps > 0.0:
            probs_soft = (1.0 - eps) * probs_soft + eps / float(self.n_experts)
        # Top-K mask + 重新归一化（用于实际专家融合）
        probs = probs_soft
        effective_k = self.top_k if (self._train_step >= self.topk_warm_steps) else 1
        effective_k = max(1, min(effective_k, self.n_experts))
        if effective_k < self.n_experts:
            topk = torch.topk(probs, k=effective_k, dim=-1)
            idx = topk.indices  # [B,T,K]
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, idx, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 专家输出并加权融合
        y_flat = y.reshape(B * T, D)
        outs = []
        for e in self.experts:
            outs.append(e(y_flat).reshape(B, T, D))
        outs = torch.stack(outs, dim=-1)  # [B,T,D,E]
        probs_e = probs.unsqueeze(-2)     # [B,T,1,E]
        residual = (outs * probs_e).sum(dim=-1)  # [B,T,D]
        # Residual scale ramp
        if self.res_scale_steps > 0:
            s = min(1.0, float(self._train_step) / float(self.res_scale_steps))
            scale_cur = self.res_scale_start + (self.res_scale_end - self.res_scale_start) * s
        else:
            scale_cur = self.residual_scale
        residual = float(scale_cur) * residual

        # 可微辅助loss：均衡 + 熵（早期强，随步数衰减）
        util = probs_soft.mean(dim=(0, 1))  # [E]
        uni = torch.full_like(util, 1.0 / float(self.n_experts))
        lb = ((util - uni) ** 2).sum()
        ent = (- (probs_soft * (probs_soft.clamp_min(1e-8).log())).sum(dim=-1)).mean()
        if self.entropy_warm_steps > 0:
            w_ent = self.entropy_weight * (1.0 - min(1.0, float(self._train_step) / float(self.entropy_warm_steps)))
        else:
            w_ent = 0.0
        aux_loss = self.balance_weight * lb + (-w_ent) * ent

        # 统计缓存
        with torch.no_grad():
            self._last_gate = probs.detach().clone()  # [B,T,E]
            self._last_top1 = torch.argmax(logits, dim=-1)  # [B,T]
            self._last_res_en = residual.detach().abs().mean().item()
        if return_aux:
            return residual, {
                'balance': lb.detach(),
                'entropy': ent.detach(),
                'loss': aux_loss,
            }
        return residual

    @torch.no_grad()
    def get_stats(self) -> Dict[str, Any]:
        if self._last_gate is None:
            return {}
        g = self._last_gate  # [B,T,E]
        util = g.mean(dim=(0, 1))  # [E]
        top1 = None
        if self._last_top1 is not None:
            E = g.size(-1)
            counts = torch.bincount(self._last_top1.flatten(), minlength=E).float()
            top1 = counts / max(1, self._last_top1.numel())
        entropy = (- (g * (g.clamp_min(1e-8).log())).sum(dim=-1)).mean()  # 平均熵
        return {
            'util': util.cpu().tolist(),
            'top1': top1.cpu().tolist() if top1 is not None else None,
            'entropy': float(entropy.cpu().item()),
            'residual_energy': float(getattr(self, '_last_res_en', 0.0)),
        }

    def set_training_step(self, step: int):
        self._train_step = int(step)


def test_aether_fargan_decoder():
    """测试AETHER-FARGAN解码器"""
    print("测试AETHER-FARGAN解码器接口兼容性...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建解码器
    decoder = AETHERFARGANDecoder(
        dz=24, d_csi=32, d_out=36,
        enable_synth=True,
        feature_spec_type="fargan"
    ).to(device)

    # 打印模型信息
    info = decoder.get_info()
    print(f"模型信息:")
    for k, v in info.items():
        if k != 'components':
            print(f"  {k}: {v}")

    print(f"组件:")
    for k, v in info['components'].items():
        print(f"  {k}: {v}")

    # 测试数据
    B, T, dz = 2, 20, 24
    z = torch.randn(B, T, dz, device=device)
    csi_dict = {
        'snr_db': torch.tensor([10.0, 15.0], device=device),
        'channel_type': 'awgn'
    }

    print(f"\n测试数据: z{z.shape}, CSI{list(csi_dict.keys())}")

    # 测试1: 仅特征重建 (输出36维FARGAN特征)
    with torch.no_grad():
        fargan_features = decoder(z, csi_dict, return_wave=False)
        print(f"特征重建: {fargan_features.shape}")
        assert fargan_features.shape == (B, T, 36), f"特征维度错误: {fargan_features.shape}"

    # 测试2: 特征+波形生成 (FARGAN扩展接口)
    with torch.no_grad():
        target_len = T * 160
        fargan_features, audio = decoder(z, csi_dict, return_wave=True, target_len=target_len)
        print(f"FARGAN特征: {fargan_features.shape}")
        print(f"生成音频: {audio.shape}")
        assert fargan_features.shape == (B, T, 36), f"FARGAN特征维度错误: {fargan_features.shape}"
        assert audio.shape[-1] <= target_len, f"音频长度超出目标: {audio.shape[-1]} > {target_len}"

    print(f"\nAETHER-FARGAN解码器测试通过!")
    print(f"   - 继承AETHERDecoder架构和接口")
    print(f"   - 复用GLABlock、ConvRefineDecoder")
    print(f"   - 直接输出36维FARGAN特征")
    print(f"   - 扩展FARGAN波形生成能力")


if __name__ == "__main__":
    test_aether_fargan_decoder()
