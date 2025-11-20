# -*- coding: utf-8 -*-
"""
FARGAN Components for AETHER Integration
改编自原始FARGAN代码，适配AETHER架构
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# 使用旧版 weight_norm（生成 weight_g / weight_v），以便与 Stage2 权重完全对齐
from torch.nn.utils import weight_norm
from typing import Tuple, Optional, Dict, Any


def add_quantization_noise(x: torch.Tensor, training: bool = True) -> torch.Tensor:
    """添加量化噪声 (对应原始FARGAN的n函数)"""
    if not training:
        return x
    noise = (1.0 / 127.0) * (torch.rand_like(x) - 0.5)
    return torch.clamp(x + noise, min=-1.0, max=1.0)


class GLU(nn.Module):
    """门控线性单元 (Gated Linear Unit)"""

    def __init__(self, feat_size: int):
        super().__init__()
        torch.manual_seed(5)  # 保持与原始FARGAN一致的随机种子
        self.gate = weight_norm(nn.Linear(feat_size, feat_size, bias=False))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class FWConv(nn.Module):
    """前向卷积模块 (Frame-wise Convolution)"""

    def __init__(self, in_size: int, out_size: int, kernel_size: int = 2):
        super().__init__()
        torch.manual_seed(5)
        self.in_size = in_size
        self.kernel_size = kernel_size
        self.conv = weight_norm(nn.Linear(in_size * kernel_size, out_size, bias=False))
        self.glu = GLU(out_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear, nn.Embedding)):
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, in_size] 当前输入
            state: [B, in_size*(kernel_size-1)] 历史状态

        Returns:
            output: [B, out_size] 输出
            new_state: [B, in_size*(kernel_size-1)] 更新的状态
        """
        
        w_dtype = self.conv.weight.dtype
        if x.dtype != w_dtype:      x     = x.to(w_dtype)
        if state.dtype != w_dtype:  state = state.to(w_dtype)
        xcat = torch.cat((state, x), -1)  # [B, in_size*kernel_size]
        out = self.glu(torch.tanh(self.conv(xcat)))
        new_state = xcat[:, self.in_size:]  # 更新状态
        return out, new_state


class FARGANCond(nn.Module):
    """FARGAN条件网络 - 将特征转换为条件信号"""

    def __init__(self, feature_dim: int = 20, cond_size: int = 256, pembed_dims: int = 12):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_size = cond_size

        # 周期嵌入 (Period Embedding)
        self.pembed = nn.Embedding(224, pembed_dims)  # 支持周期32-255

        # 特征处理网络
        self.fdense1 = nn.Linear(self.feature_dim + pembed_dims, 64, bias=False)
        self.fconv1 = nn.Conv1d(64, 128, kernel_size=3, padding='valid', bias=False)
        self.fdense2 = nn.Linear(128, 80 * 4, bias=False)  # 4个子帧

        self._init_weights()
        nb_params = sum(p.numel() for p in self.parameters())
        print(f"FARGANCond model: {nb_params} parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for p in m.named_parameters():
                    if p[0].startswith('weight_hh_'):
                        nn.init.orthogonal_(p[1])

    def forward(self, features: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, feature_dim] 输入特征
            period: [B, T] 周期 (32-255)

        Returns:
            cond: [B, T-2, 320] 条件信号 (80*4维)
        """
        # --- 先做时间维稳健对齐，再丢前2帧 ---
        # 对齐到相同的时间长度，避免 features/period 相差2帧导致 cat 报错
        T = min(features.size(1), period.size(1))
        if features.size(1) != T:
            features = features[:, :T, :]
        if period.size(1) != T:
            period = period[:, :T]
        # 去掉前2帧，保持与原始FARGAN一致
        if T > 2:
            features = features[:, 2:, :]
            period = period[:, 2:]
        else:
            # 不足以丢2帧时，直接返回空时间轴（让上游选择跳过本 batch 的 wave loss）
            return features.new_zeros(features.size(0), 0, 80 * 4)

        # 周期嵌入
        w_dtype = self.fdense1.weight.dtype
        period = (period - 32).clamp(0, 223).to(torch.long)     # 索引必须 long
        p = self.pembed(period).to(w_dtype)                     # 与权重同 dtype
        features = features.to(w_dtype)                         # 输入也对齐
        features = torch.cat((features, p), -1)

        # 网络前向
        tmp = torch.tanh(self.fdense1(features))
        tmp = tmp.permute(0, 2, 1)  # [B, 64, T-2]
        tmp = torch.tanh(self.fconv1(tmp))  # [B, 128, T-4]
        tmp = tmp.permute(0, 2, 1)  # [B, T-4, 128]
        tmp = torch.tanh(self.fdense2(tmp))  # [B, T-4, 320]

        return tmp


class FARGANSub(nn.Module):
    """FARGAN子帧网络 - 逐子帧生成音频"""

    def __init__(self, subframe_size: int = 40, nb_subframes: int = 4, cond_size: int = 256):
        super().__init__()
        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.cond_size = cond_size

        # 条件增益网络
        self.cond_gain_dense = nn.Linear(80, 1)

        # 前向卷积层
        self.fwc0 = FWConv(2 * self.subframe_size + 80 + 4, 192)

        # 三层GRU网络
        self.gru1 = nn.GRUCell(192 + 2 * self.subframe_size, 160, bias=False)
        self.gru2 = nn.GRUCell(160 + 2 * self.subframe_size, 128, bias=False)
        self.gru3 = nn.GRUCell(128 + 2 * self.subframe_size, 128, bias=False)

        # GLU激活
        self.gru1_glu = GLU(160)
        self.gru2_glu = GLU(128)
        self.gru3_glu = GLU(128)
        self.skip_glu = GLU(128)

        # 输出层
        self.skip_dense = nn.Linear(192 + 160 + 2 * 128 + 2 * self.subframe_size, 128, bias=False)
        self.sig_dense_out = nn.Linear(128, self.subframe_size, bias=False)
        self.gain_dense_out = nn.Linear(192, 4)  # 4个基频增益

        self._init_weights()
        nb_params = sum(p.numel() for p in self.parameters())
        print(f"FARGANSub model: {nb_params} parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for p in m.named_parameters():
                    if p[0].startswith('weight_hh_'):
                        nn.init.orthogonal_(p[1])

    def forward(
        self,
        cond: torch.Tensor,
        prev_pred: torch.Tensor,
        exc_mem: torch.Tensor,
        period: torch.Tensor,
        states: Tuple[torch.Tensor, ...],
        gain: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            cond: [B, 80] 当前子帧条件
            prev_pred: [B, 256] 前一预测信号
            exc_mem: [B, 256] 激励缓冲区
            period: [B] 当前周期
            states: 4个GRU状态元组
            gain: [B, 1] 可选的外部增益

        Returns:
            sig_out: [B, subframe_size] 输出信号
            exc_mem: [B, 256] 更新的激励缓冲区
            prev_pred: [B, 256] 更新的预测信号
            states: 更新的GRU状态
        """
        run_dtype = self.fwc0.conv.weight.dtype
        # 兜底：若上游忘记提供 states，这里自举零状态
        if states is None:
            B = cond.size(0)
            states = (
                torch.zeros(B, 160, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 128, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 128, device=cond.device, dtype=run_dtype),
                torch.zeros(B, 2*self.subframe_size + 80 + 4, device=cond.device, dtype=run_dtype),
            )

        if cond.dtype      != run_dtype: cond      = cond.to(run_dtype)
        if prev_pred.dtype != run_dtype: prev_pred = prev_pred.to(run_dtype)
        if exc_mem.dtype   != run_dtype: exc_mem   = exc_mem.to(run_dtype)
        # states 是4个张量的tuple
        states = tuple( s.to(run_dtype) if isinstance(s, torch.Tensor) and s.dtype != run_dtype else s
                        for s in states )
        device = exc_mem.device

        # 添加量化噪声
        cond = add_quantization_noise(cond, self.training)

        # 条件增益 - 修复：添加幅值限制防止音频爆炸
        if gain is None:
            gain_logits = self.cond_gain_dense(cond)
            # 进一步限制增益范围到 [0.2, 1.0]，更保守的范围防止音频幅值过大
            gain = 0.2 + 0.8 * torch.sigmoid(gain_logits)
        if gain is not None:
            gain = torch.nan_to_num(gain, nan=1.0, posinf=20.0, neginf=1e-3).clamp_(1e-3, 20.0)
        # 基频预测 - 从激励缓冲区中提取周期性信号
        idx = 256 - period[:, None]  # [B, 1]
        rng = torch.arange(self.subframe_size + 4, device=device)  # [44]
        idx = idx + rng[None, :] - 2  # [B, 44]

        # 处理周期边界
        mask = idx >= 256
        idx = idx - mask * period[:, None]

        # 处理负索引 - 将负索引设为0
        idx = torch.clamp(idx, min=0, max=255)

        # 提取预测信号
        pred = torch.gather(exc_mem, 1, idx)  # [B, 44]
        pred = add_quantization_noise(pred / (1e-5 + gain), self.training)

        # 前一激励信号
        prev = exc_mem[:, -self.subframe_size:]  # [B, 40]
        prev = add_quantization_noise(prev / (1e-5 + gain), self.training)

        # 组合输入
        tmp = torch.cat((cond, pred, prev), 1)  # [B, 80+44+40]

        # 滤波后的基频信号
        fpitch = pred[:, 2:-2]  # [B, 40]

        # 前向卷积
        fwc0_out, fwc0_state = self.fwc0(tmp, states[3])
        fwc0_out = add_quantization_noise(fwc0_out, self.training)

        # 基频增益
        pitch_gain = torch.sigmoid(self.gain_dense_out(fwc0_out))  # [B, 4]

        # GRU层级联
        gru1_state = self.gru1(
            torch.cat([fwc0_out, pitch_gain[:, 0:1] * fpitch, prev], 1),
            states[0]
        )
        gru1_out = self.gru1_glu(add_quantization_noise(gru1_state, self.training))

        gru2_state = self.gru2(
            torch.cat([gru1_out, pitch_gain[:, 1:2] * fpitch, prev], 1),
            states[1]
        )
        gru2_out = self.gru2_glu(add_quantization_noise(gru2_state, self.training))

        gru3_state = self.gru3(
            torch.cat([gru2_out, pitch_gain[:, 2:3] * fpitch, prev], 1),
            states[2]
        )
        gru3_out = self.gru3_glu(add_quantization_noise(gru3_state, self.training))

        # 跳跃连接
        gru_concat = torch.cat([gru1_out, gru2_out, gru3_out, fwc0_out], 1)
        skip_input = torch.cat([gru_concat, pitch_gain[:, 3:4] * fpitch, prev], 1)
        skip_out = torch.tanh(self.skip_dense(skip_input))
        skip_out = self.skip_glu(add_quantization_noise(skip_out, self.training))

        # 最终输出
        sig_out = torch.tanh(self.sig_dense_out(skip_out))  # [B, 40]
        sig_out = sig_out * gain

        # 更新缓冲区
        exc_mem = torch.cat([exc_mem[:, self.subframe_size:], sig_out], 1)
        prev_pred = torch.cat([prev_pred[:, self.subframe_size:], fpitch], 1)

        new_states = (gru1_state, gru2_state, gru3_state, fwc0_state)

        return sig_out, exc_mem, prev_pred, new_states


class FARGANCore(nn.Module):
    """FARGAN核心模块 - 组合条件网络和子帧网络"""

    def __init__(
        self,
        subframe_size: int = 40,
        nb_subframes: int = 4,
        feature_dim: int = 20,
        cond_size: int = 256
    ):
        super().__init__()
        self.subframe_size = subframe_size
        self.nb_subframes = nb_subframes
        self.frame_size = self.subframe_size * self.nb_subframes
        self.feature_dim = feature_dim
        self.cond_size = cond_size

        self.cond_net = FARGANCond(feature_dim=feature_dim, cond_size=cond_size)
        self.sig_net = FARGANSub(
            subframe_size=subframe_size,
            nb_subframes=nb_subframes,
            cond_size=cond_size
        )

    def forward(
        self,
        features: torch.Tensor,
        period: torch.Tensor,
        nb_frames: int,
        pre: Optional[torch.Tensor] = None,
        states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            features: [B, T, feature_dim] 输入特征
            period: [B, T] 周期序列
            nb_frames: 要生成的帧数
            pre: [B, L] 可选的前置信号
            states: 可选的初始状态

        Returns:
            sig: [B, nb_frames*frame_size] 生成的音频
            states: 最终状态
        """

        device = features.device
        batch_size = features.size(0)

        # === 新增：以 cond_net 的权重作为全网运行dtype ===
        model_dtype = self.cond_net.fdense1.weight.dtype
        if features.dtype != model_dtype:
            features = features.to(model_dtype)
        if pre is not None and pre.dtype != model_dtype:
            pre = pre.to(model_dtype)

        # 初始化缓冲区（指定 dtype）
        prev   = torch.zeros(batch_size, 256, device=device, dtype=model_dtype)
        exc_mem= torch.zeros(batch_size, 256, device=device, dtype=model_dtype)
        # —— dtype 对齐、device、batch_size 保持你现有写法 ——
        # 初始化 RNN/FWConv 状态（当 states=None）
        if states is None:
            states = (
                torch.zeros(batch_size, 160, device=device, dtype=model_dtype),         # GRU1
                torch.zeros(batch_size, 128, device=device, dtype=model_dtype),         # GRU2
                torch.zeros(batch_size, 128, device=device, dtype=model_dtype),         # GRU3
                torch.zeros(batch_size, 2*self.subframe_size + 80 + 4,                  # FWConv state = in_size*(k-1)
                            device=device, dtype=model_dtype),                          # 2*40 + 80 + 4 = 164
            )

        # 预热帧数
        nb_pre_frames = pre.size(1) // self.frame_size if pre is not None else 0

        # （推荐）prime 激励缓冲为 pre 的“最后一帧”，更贴近自回归状态
        if pre is not None and pre.size(1) >= self.frame_size:
            exc_mem[:, -self.frame_size:] = pre[:, -self.frame_size:]

        # 生成条件（长度 = T_in - 4 = nb_frames）
        cond = self.cond_net(features, period)  # [B, nb_frames, 320]

        # —— 关键修复：循环严格以 cond 的时间维为准 ——
        # n ∈ [0, nb_frames-1]；这样 subframe_cond = cond[:, n, ...] 永不越界
        sig = torch.zeros((batch_size, 0), device=device, dtype=model_dtype)

        # 可选：健壮性断言（出问题就立刻早停，便于诊断）
        assert cond.size(1) >= nb_frames, f"cond_len={cond.size(1)} < nb_frames={nb_frames}"
        assert period.size(1) >= nb_frames + 3, f"period_len={period.size(1)} < nb_frames+3={nb_frames+3}"
        if not hasattr(self, "_checked_shapes"):
            print(f"[CoreCheck] cond_len={cond.size(1)}, nb_frames={nb_frames}, "
                f"period_len={period.size(1)}, feat_len={features.size(1)}, "
                f"state_shapes={[tuple(s.shape) for s in states]}")
            self._checked_shapes = True
        for n in range(0, nb_frames):
            for k in range(self.nb_subframes):
                pos = n * self.frame_size + k * self.subframe_size

                # —— 与conv中心对齐：使用 3 + n，并加上界保护（避免末尾极端边界时 +1 越界）
                per_idx  = min(3 + n, period.size(1)   - 1)
                feat_idx = min(3 + n, features.size(1) - 1)

                pitch = period[:, per_idx]
                gain = 0.03 * torch.pow(10.0, 0.5 * features[:, feat_idx, 0:1] / math.sqrt(18.0))
                gain = torch.nan_to_num(gain, nan=1.0, posinf=20.0, neginf=1e-3).clamp_(1e-3, 20.0)


                subframe_cond = cond[:, n, k * 80:(k + 1) * 80]

                out, exc_mem, prev, states = self.sig_net(
                    subframe_cond, prev, exc_mem, pitch, states, gain=gain
                )

                if (n < nb_pre_frames) and (pre is not None):
                    # teacher-forcing: 用真实波形覆盖，并把 exc_mem 以真实输出推进
                    out = pre[:, pos:pos + self.subframe_size]
                    exc_mem[:, -self.subframe_size:] = out
                else:
                    # 自回归：累计输出
                    sig = torch.cat([sig, out], dim=1)

        # 分离状态梯度
        states = tuple(s.detach() for s in states)
        return sig, states



def test_fargan_components():
    """测试FARGAN组件"""
    print("🧪 测试FARGAN组件...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T = 2, 10

    # 测试条件网络
    print("📡 测试FARGANCond...")
    cond_net = FARGANCond(feature_dim=20).to(device)
    features = torch.randn(B, T, 20, device=device)
    period = torch.randint(32, 256, (B, T), device=device)
    cond = cond_net(features, period)
    print(f"条件网络: 输入{features.shape} -> 输出{cond.shape}")

    # 测试子帧网络
    print("🎵 测试FARGANSub...")
    sub_net = FARGANSub().to(device)
    states = (
        torch.zeros(B, 160, device=device),
        torch.zeros(B, 128, device=device),
        torch.zeros(B, 128, device=device),
        torch.zeros(B, 124, device=device)
    )
    exc_mem = torch.randn(B, 256, device=device)
    prev_pred = torch.randn(B, 256, device=device)
    cond_sub = torch.randn(B, 80, device=device)
    period_sub = torch.randint(32, 256, (B,), device=device)

    sig_out, exc_mem_new, prev_pred_new, states_new = sub_net(
        cond_sub, prev_pred, exc_mem, period_sub, states
    )
    print(f"子帧网络: 输出{sig_out.shape}")

    # 测试核心模块
    print("🚀 测试FARGANCore...")
    core = FARGANCore(feature_dim=20).to(device)
    nb_frames = 5
    pre = torch.randn(B, 160, device=device)  # 1帧前置
    sig, final_states = core(features, period, nb_frames, pre=pre)
    print(f"核心模块: 输入{features.shape} -> 输出{sig.shape}")

    print("✅ FARGAN组件测试通过")


if __name__ == "__main__":
    test_fargan_components()
