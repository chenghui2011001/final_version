#!/usr/bin/env python3
"""
48维特征提取器 - 基于LPCNet方法的改进版本
学习FarGAN baseline的成功方法，扩展到48维

特征结构 (48维):
- [0:19]  20维CEPS特征 (DCT对数频谱，与LPCNet相同)
- [20]    F0特征 (简化版本，不用复杂谐波编码)
- [21]    Voicing特征 (语音/非语音判定)
- [22:27] 6维频谱包络增强特征
- [28:43] 16维LPC系数 (与LPCNet相同)
- [44:47] 4维韵律特征 (F0轨迹、能量等)
"""

import numpy as np
import math
import torch
import torchaudio
from pathlib import Path
import argparse
from typing import Optional
import scipy.signal
import scipy.fft


class LPCNetStyleFeatureExtractor:
    """基于LPCNet方法的48维特征提取器（GPU优化）。"""

    def __init__(self, sr: int = 16000, frame_size: int = 160, n_bands: int = 20, device: str = 'cpu', lpc_mode: str = 'off', peak_margin: int = 2, debug_f0: bool = False, biquad_stateful: bool = False, f0_mode: str = 'residual'):
        self.sr = sr
        self.frame_size = frame_size  # 10ms帧
        self.n_bands = n_bands
        self.hop_length = frame_size
        self.device = torch.device(device)
        # lpc_mode: 'off' 保持现有结果; 'approx' 使用LPCNet风格近似; 'strict18' 预留
        self.lpc_mode = lpc_mode
        # F0内部窗口峰值边距
        self.peak_margin = int(max(1, peak_margin))
        self.debug_f0 = bool(debug_f0)
        self._debug_f0_done = False
        # 可选跨批 biquad 状态
        self.biquad_stateful = bool(biquad_stateful)
        self._biquad_mem0_stream = 0.0
        self._biquad_mem1_stream = 0.0
        # F0 mode: 'residual' (LPCNet-like whitening) or 'raw' (preemph+NCCF)
        self.f0_mode = f0_mode

        # LPCNet兼容参数
        self.lpc_order = 16
        self.preemph_coef = 0.85

        # 频谱分析参数
        self.n_fft = 512
        self.win_length = 320  # 20ms窗
        self._hann_win = torch.hann_window(self.win_length, periodic=True, device=self.device)

        # 预计算DCT-II矩阵（n_bands x n_bands）用于CEPS（torch实现）
        self._dct_mat = self._build_dct_matrix(self.n_bands, device=self.device)

        # 预计算频带边界（线性划分，模拟LPCNet）
        self._band_slices = self._build_band_slices(self.n_fft, self.sr, self.n_bands)
        # 近似的补偿向量（将LPCNet的18带compensation插值到n_bands）
        self._comp_vec = self._build_compensation(self.n_bands)
        # 预计算用于频带聚合的索引与掩码（batched快速求和）
        n_bins = self.n_fft // 2 + 1
        lo_idx = []
        hi_idx = []
        for lo, hi in self._band_slices:
            lo_idx.append(max(0, int(lo)))
            hi_idx.append(min(int(hi), n_bins - 1))
        self._band_lo = torch.tensor(lo_idx, dtype=torch.long, device=self.device)
        self._band_hi = torch.tensor(hi_idx, dtype=torch.long, device=self.device)

        # 初始化状态
        self.preemph_mem = 0.0
        self.pitch_mem = np.zeros(self.lpc_order)
        self._init_pitch_states()

    def _f0_cpu_raw_from_preemph_signal(self, y: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
        """CPU逐帧F0：在预加重后的1D信号y上逐帧（320样点）做NCCF+抛物线插值。
        返回 dnn_pitch[T], voicing[T]
        """
        N = self.win_length
        hop = self.hop_length
        min_p, max_p = 32, 255
        dnn = np.zeros(T, dtype=np.float32)
        voi = np.zeros(T, dtype=np.float32)
        margin = max(1, min(self.peak_margin, (max_p - min_p)//2 - 1))
        for i in range(T):
            start = i * hop
            end = start + N
            if end > len(y):
                break
            x = y[start:end]
            # 自相关
            ac_full = np.correlate(x, x, mode='full')
            ac = ac_full[N-1:]
            # NCCF归一化所需能量
            x2 = x.astype(np.float64)**2
            csum = np.cumsum(x2)
            total = csum[-1]
            lags = np.arange(min_p, max_p+1, dtype=np.int32)
            e0 = csum[(N-1)-lags]
            e1 = total - csum[lags-1]
            e0 = e0 + 1e-12
            e1 = e1 + 1e-12
            num = ac[lags]
            den = np.sqrt(e0 * e1)
            r = num / den
            # 内部窗口峰值
            L = len(lags)
            m = min(margin, L//2-1) if L >= 5 else 1
            core = r[m:L-m]
            k_rel = int(np.argmax(core))
            k = k_rel + m
            # 抛物线插值
            if 0 < k < L-1:
                r_m1 = r[k-1]; r_0 = r[k]; r_p1 = r[k+1]
                denom = (r_m1 - 2.0*r_0 + r_p1)
                if abs(denom) > 1e-12:
                    delta = 0.5 * (r_m1 - r_p1) / denom
                    delta = float(np.clip(delta, -0.5, 0.5))
                else:
                    delta = 0.0
            else:
                delta = 0.0
            period = float(lags[k]) + delta
            period = float(np.clip(period, min_p, max_p))
            dnn[i] = np.log2(256.0/period) - 1.5
            voi[i] = 1.0 if r[k] > 0.3 else 0.0
        return dnn, voi

    def _init_pitch_states(self):
        """Initialize pitch detection states"""
        self.last_period_est: Optional[float] = None
        self.last_voiced: bool = False
        self.voicing_hang: int = 0
        self.max_voicing_hang: int = 3

    @staticmethod
    def _levinson_durbin(r: np.ndarray, order: int):
        """Levinson–Durbin recursion returning AR polynomial a with a[0]=1.
        r: autocorrelation lags [0..order]
        """
        a = np.zeros(order + 1, dtype=np.float64)
        e = float(r[0])
        if e <= 1e-12:
            a[0] = 1.0
            return a.astype(np.float32), 0.0
        a[0] = 1.0
        for i in range(1, order + 1):
            acc = 0.0
            for j in range(1, i):
                acc += a[j] * r[i - j]
            k = -(r[i] + acc) / (e + 1e-12)
            a_prev = a.copy()
            for j in range(1, i):
                a[j] = a_prev[j] + k * a_prev[i - j]
            a[i] = k
            e = e * (1.0 - k * k)
            if e <= 1e-12:
                break
        return a.astype(np.float32), float(max(e, 1e-12))

    def _lpc_residual(self, x: np.ndarray, order: int = 16) -> np.ndarray:
        """Compute LPC residual e[n] = x[n] + sum_{k=1..p} a[k] x[n-k]."""
        # Autocorrelation up to order
        ac_full = np.correlate(x, x, mode='full')
        N = len(x)
        r = ac_full[N - 1:N + order]
        a, _ = self._levinson_durbin(r, order)
        # Analysis filter A(z) applied as FIR on x
        residual = scipy.signal.lfilter(a, [1.0], x)
        return residual.astype(np.float32)

    @staticmethod
    def _build_dct_matrix(N: int, device: torch.device):
        n = torch.arange(N, device=device).float()
        k = torch.arange(N, device=device).float().unsqueeze(1)
        mat = torch.cos((np.pi / N) * (n + 0.5) * k)
        mat *= (2.0 / N) ** 0.5
        mat[0, :] *= (0.5) ** 0.5  # k=0行额外系数 sqrt(1/2)
        return mat

    @staticmethod
    def _build_band_slices(n_fft: int, sr: int, n_bands: int):
        # rfft bins = n_fft//2+1; linearly split [0, sr/2]
        n_bins = n_fft // 2 + 1
        freq_per_bin = sr / 2 / (n_fft // 2)
        slices = []
        for i in range(n_bands):
            f_low = i * sr / 2 / n_bands
            f_high = (i + 1) * sr / 2 / n_bands
            bin_low = int(f_low / freq_per_bin)
            bin_high = min(int(f_high / freq_per_bin), n_bins - 1)
            if bin_high <= bin_low:
                bin_high = min(bin_low + 1, n_bins - 1)
            slices.append((bin_low, bin_high))
        return slices

    @staticmethod
    def _build_compensation(n_bands: int) -> np.ndarray:
        # LPCNet 18带补偿表（dnn/freq.c: compensation）
        comp18 = np.array([
            0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            2.0/3.0, 0.5, 0.5, 0.5, 1.0/3.0, 0.25, 0.25, 0.2, 1.0/6.0, 0.173913
        ], dtype=np.float32)
        if n_bands == 18:
            return comp18
        # 线性插值到 n_bands
        x18 = np.linspace(0, 1, num=18)
        xN = np.linspace(0, 1, num=n_bands)
        compN = np.interp(xN, x18, comp18).astype(np.float32)
        return compN

    def _idct_ortho(self, vec: np.ndarray) -> np.ndarray:
        # 我们的DCT矩阵按正交构造，IDCT即转置乘以向量
        vt = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        out = (self._dct_mat.t() @ vt).detach().cpu().numpy()
        return out

    def _interp_bands_to_spectrum(self, Ex: np.ndarray) -> np.ndarray:
        # 将 n_bands 的带能量线性插值到 rfft 半谱长度
        n_bins = self.n_fft // 2 + 1
        j = np.arange(n_bins, dtype=np.float32)
        # 归一化到 [0, n_bands-1]
        pos = j * (self.n_bands - 1) / (n_bins - 1)
        idx = np.floor(pos).astype(np.int32)
        frac = (pos - idx).astype(np.float32)
        idx1 = np.clip(idx + 1, 0, self.n_bands - 1)
        Xr = (1.0 - frac) * Ex[idx] + frac * Ex[idx1]
        return Xr.astype(np.float32)

    def compute_lpc_lpcnet_approx(self, ceps: np.ndarray) -> np.ndarray:
        """LPCNet风格的近似LPC恢复：cepstrum -> IDCT -> pow10 -> compensation -> IFFT -> AC -> Levinson.
        保持48维不变，仅作为近似对齐路径；默认不启用以避免改变输出。
        """
        if ceps.shape[0] != self.n_bands:
            ceps = ceps[:self.n_bands]
        tmp = ceps.copy().astype(np.float32)
        tmp[0] += 4.0
        # IDCT 得到对数能量（log10）
        Ex_log10 = self._idct_ortho(tmp)
        # 还原功率并应用补偿
        Ex = (10.0 ** Ex_log10).astype(np.float64)
        Ex *= self._comp_vec.astype(np.float64)
        # 插值到半谱
        Xr = self._interp_bands_to_spectrum(Ex.astype(np.float32)).astype(np.float64)
        # IFFT 到时间域，自相关近似
        x_auto = np.fft.irfft(Xr, n=self.n_fft).real
        ac = x_auto[: self.lpc_order + 1].astype(np.float64)
        # 噪声底与滞后窗
        ac0 = ac[0]
        ac[0] = ac0 + ac0 * 1e-4 + 320.0 / 12.0 / 38.0
        for i in range(1, self.lpc_order + 1):
            ac[i] *= (1.0 - 6e-5 * i * i)
        a, _ = self._levinson_durbin(ac, self.lpc_order)
        coeffs = a[1:].astype(np.float32)
        return coeffs

    # ===================== Batched GPU pipeline helpers =====================
    def _frame_signal_torch(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(0, self.win_length, self.hop_length)

    def _preemph_with_mem(self, xt: torch.Tensor, mem_in: float) -> tuple[torch.Tensor, float]:
        # xt: 1-D torch tensor on device
        if xt.numel() == 0:
            return xt, mem_in
        yt = xt.clone()
        yt[0] = xt[0] + float(mem_in)
        if xt.numel() > 1:
            yt[1:] = xt[1:] - self.preemph_coef * xt[:-1]
        mem_out = float(-self.preemph_coef * xt[-1].item())
        return yt, mem_out

    def _band_energy_batched(self, power_spec: torch.Tensor) -> torch.Tensor:
        # power_spec: [T, n_bins]
        csum = torch.cumsum(power_spec, dim=-1)
        # inclusive sums on [lo..hi]
        # gather hi and lo-1
        hi_vals = torch.gather(csum, -1, self._band_hi.unsqueeze(0).expand(power_spec.size(0), -1))
        lo_clamped = torch.clamp(self._band_lo - 1, min=-1)
        # Build lo-1 values; for -1, use zeros
        mask_neg = (lo_clamped == -1)
        lo_gather_idx = torch.where(mask_neg, torch.zeros_like(lo_clamped), lo_clamped)
        lo_vals = torch.gather(csum, -1, lo_gather_idx.unsqueeze(0).expand(power_spec.size(0), -1))
        lo_vals = torch.where(mask_neg.unsqueeze(0), torch.zeros_like(lo_vals), lo_vals)
        bandE = hi_vals - lo_vals
        return bandE

    def _log_follow_limit(self, logE: torch.Tensor) -> torch.Tensor:
        # logE: [T, n_bands]
        T, B = logE.shape
        out = torch.empty_like(logE)
        log_max = torch.full((T,), -2.0, device=logE.device)
        follow = torch.full((T,), -2.0, device=logE.device)
        for i in range(B):
            vi = logE[:, i]
            tmp = torch.maximum(log_max - 8.0, torch.maximum(follow - 2.5, vi))
            out[:, i] = tmp
            log_max = torch.maximum(log_max, tmp)
            follow = torch.maximum(follow - 2.5, tmp)
        return out

    def _levinson_batched(self, r: torch.Tensor, order: int) -> torch.Tensor:
        # r: [T, order+1]
        T = r.size(0)
        device = r.device
        a = torch.zeros((T, order + 1), dtype=r.dtype, device=device)
        a[:, 0] = 1.0
        e = r[:, 0].clone()
        eps = 1e-12
        for i in range(1, order + 1):
            # acc = sum_{j=1..i-1} a_j * r_{i-j}
            if i == 1:
                acc = torch.zeros_like(e)
            else:
                aj = a[:, 1:i]                  # [T, i-1]
                # r[:, 1:i] has cols [1..i-1]; reverse order to [i-1..1]
                rij = torch.flip(r[:, 1:i], dims=[1])
                acc = torch.sum(aj * rij, dim=1)
            k = -(r[:, i] + acc) / torch.clamp(e, min=eps)
            a_prev = a.clone()
            if i > 1:
                # Update using reversed previous coefficients without negative step slicing
                a_rev = torch.flip(a_prev[:, 1:i], dims=[1])  # columns i-1..1
                a[:, 1:i] = a_prev[:, 1:i] + k.unsqueeze(1) * a_rev
            a[:, i] = k
            e = e * (1.0 - k * k)
            e = torch.clamp(e, min=eps)
        # return coefficients a1..aP
        return a[:, 1:]

    def _biquad_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LPCNet's low-pass biquad to each frame in batch.
        x: [T, N] tensor on same device.
        Coefficients from lpcnet_enc.c (lp_b, lp_a).
        """
        b0 = -0.84946
        b1 = 1.0
        a0 = -1.54220
        a1 = 0.70781
        b0_a0 = b0 - a0
        b1_a1 = b1 - a1
        T, N = x.shape
        y = torch.empty_like(x)
        if self.biquad_stateful:
            mem0 = torch.full((T,), float(self._biquad_mem0_stream), dtype=x.dtype, device=x.device)
            mem1 = torch.full((T,), float(self._biquad_mem1_stream), dtype=x.dtype, device=x.device)
        else:
            mem0 = torch.zeros((T,), dtype=x.dtype, device=x.device)
            mem1 = torch.zeros((T,), dtype=x.dtype, device=x.device)
        for t in range(N):
            xi = x[:, t]
            yi = xi + mem0
            y[:, t] = yi
            mem00 = mem0
            mem0 = b0_a0 * xi + mem1 - a0 * mem0
            mem1 = b1_a1 * xi - a1 * mem00
        # 更新跨批状态（取最后一帧的滤波器状态作为延续）
        if self.biquad_stateful:
            self._biquad_mem0_stream = float(mem0[-1].item())
            self._biquad_mem1_stream = float(mem1[-1].item())
        return y

    @staticmethod
    def _logistic01(x: torch.Tensor, k: float = 5.0) -> torch.Tensor:
        """Logistic mapping to [0,1] as in LPCNet voicing smoothing."""
        k_t = torch.tensor(k, dtype=x.dtype, device=x.device)
        return torch.log1p(torch.exp(k_t * x)) / torch.log1p(torch.exp(k_t))

    def extract_features_streaming_from_signal(self, audio_np: np.ndarray, preemph_mem: float = 0.0, max_frames: Optional[int] = None) -> tuple[np.ndarray, float]:
        """流式逐帧处理，仿照LPCNet C版本的内存常量级处理"""
        # Setup ring buffer and state management
        hop = self.frame_size  # 160
        win_size = self.win_length  # 320

        # Calculate frame count
        total_samples = len(audio_np)
        n_frames = max(0, (total_samples - win_size) // hop + 1)
        if max_frames is not None:
            n_frames = min(n_frames, max_frames)

        if n_frames <= 0:
            return np.zeros((0, 48), dtype=np.float32), preemph_mem

        # Initialize output array
        features = np.zeros((n_frames, 48), dtype=np.float32)

        # Initialize streaming states (like C ring buffers)
        current_preemph_mem = preemph_mem

        # Enhanced F0 tracking states with multi-frame context
        f0_states = {
            'last_period': None,
            'last_voiced': False,
            'voicing_hang': 0,
            'max_hang': 3,
            # Multi-frame context for better continuity
            'f0_history': [],           # 保存最近几帧的F0值
            'period_history': [],       # 保存最近几帧的period值
            'correlation_history': [],  # 保存最近几帧的相关性
            'max_history': 5           # 历史窗口大小
        }

        # Dynamic range tracking for log energy follow
        log_max = -2.0
        follow = -2.0

        # Process frame by frame (constant memory like C)
        for i in range(n_frames):
            start_idx = i * hop
            end_idx = start_idx + win_size

            if end_idx > total_samples:
                break

            # Get current frame
            frame = audio_np[start_idx:end_idx].copy()

            # Preemphasis with memory (like C state update)
            frame_preemph = self._preemph_frame_with_mem(frame, current_preemph_mem)
            current_preemph_mem = -self.preemph_coef * frame[-1]

            # Extract single frame features (constant memory operations)
            frame_feat = self._extract_single_frame_features(
                frame, frame_preemph, f0_states, log_max, follow
            )

            # Update dynamic states
            log_max = frame_feat['log_max']
            follow = frame_feat['follow']

            # Store features
            features[i] = frame_feat['features']

            # Minimal progress reporting to avoid spam
            if i % 50000 == 0 and i > 0:
                print(f"Streaming processed {i}/{n_frames} frames")

        return features, current_preemph_mem

    def _preemph_frame_with_mem(self, frame: np.ndarray, mem: float) -> np.ndarray:
        """Single frame preemphasis with memory state"""
        result = frame.copy()
        if len(result) > 0:
            result[0] = frame[0] - self.preemph_coef * mem
            if len(result) > 1:
                result[1:] = frame[1:] - self.preemph_coef * frame[:-1]
        return result

    def _extract_single_frame_features(self, frame: np.ndarray, frame_preemph: np.ndarray,
                                     f0_states: dict, log_max: float, follow: float) -> dict:
        """Extract features from single frame (constant memory like C)"""
        # Window the frame
        windowed = frame_preemph * np.hanning(len(frame_preemph))

        # FFT (single frame, small memory footprint)
        X = np.fft.rfft(windowed, n=self.n_fft)
        power_spec = X.real**2 + X.imag**2

        # Band energies (iterate through bands, no large tensors)
        band_energies = np.zeros(self.n_bands, dtype=np.float32)
        for i, (lo, hi) in enumerate(self._band_slices):
            if hi > lo:
                band_energies[i] = np.sum(power_spec[lo:hi])
            else:
                band_energies[i] = power_spec[lo]

        # Log energies with dynamic range control (like C follow logic)
        log_energies = np.log10(band_energies + 1e-2)
        log_energies_clamped = np.zeros_like(log_energies)

        current_log_max = log_max
        current_follow = follow
        for i in range(len(log_energies)):
            val = log_energies[i]
            clamped = max(current_log_max - 8.0, max(current_follow - 2.5, val))
            log_energies_clamped[i] = clamped
            current_log_max = max(current_log_max, clamped)
            current_follow = max(current_follow - 2.5, clamped)

        # CEPS via DCT-II (small matrix operation)
        ceps = self._dct_mat.cpu().numpy() @ log_energies_clamped
        ceps[0] -= 4.0

        # F0 detection (single frame, no batching)
        dnn_pitch, voicing = self._compute_f0_single_frame(frame_preemph, f0_states)

        # Enhanced features (single frame computations)
        enhanced = self._compute_enhanced_single_frame(log_energies_clamped, frame)

        # LPC (single frame autocorrelation)
        lpc = self._compute_lpc_single_frame(frame_preemph)

        # Prosodic features
        prosodic = self._compute_prosodic_single_frame(frame, dnn_pitch, voicing)

        # Assemble 48-dim feature vector
        features = np.zeros(48, dtype=np.float32)
        features[0:20] = ceps[:20]
        features[20] = dnn_pitch
        features[21] = voicing
        features[22:28] = enhanced
        features[28:44] = lpc
        features[44:48] = prosodic

        return {
            'features': features,
            'log_max': current_log_max,
            'follow': current_follow
        }

    def _compute_f0_single_frame(self, frame: np.ndarray, f0_states: dict) -> tuple[float, float]:
        """Advanced F0 detection with multi-frame context validation"""
        # LPC residual for better pitch detection
        residual = self._lpc_residual(frame, self.lpc_order)

        # Autocorrelation via FFT (single frame)
        N = len(residual)
        nfft = 1
        target = 2 * N - 1
        while nfft < target:
            nfft <<= 1

        X = np.fft.rfft(residual, n=nfft)
        ac_full = np.fft.irfft(X.conj() * X, n=nfft).real
        ac = ac_full[:N]

        # NCCF computation
        min_p, max_p = 32, 255
        x2 = residual**2
        csum = np.cumsum(x2)
        total = csum[-1]

        lags = np.arange(min_p, max_p + 1)
        e0_idx = (N - 1) - lags
        e0_idx = np.clip(e0_idx, 0, N-1)
        e0 = csum[e0_idx] + 1e-12
        e1 = total - csum[np.clip(lags - 1, 0, N-1)] + 1e-12

        num = ac[lags]
        den = np.sqrt(e0 * e1)
        r_vals = num / den

        # Enhanced harmonic consistency
        score = r_vals.copy()

        # Subharmonic consistency (2*L)
        l2 = lags * 2
        valid2 = l2 <= max_p
        for i, (lag, valid) in enumerate(zip(lags, valid2)):
            if valid:
                exact_idx = np.where(lags == 2 * lag)[0]
                if len(exact_idx) > 0:
                    score[i] += 0.25 * r_vals[exact_idx[0]]

        # Harmonic consistency (L/2)
        for i, lag in enumerate(lags):
            half_lag = int(np.round(lag / 2.0))
            if min_p <= half_lag <= max_p:
                half_idx = half_lag - min_p
                if 0 <= half_idx < len(r_vals):
                    score[i] += 0.20 * r_vals[half_idx]

        # MULTI-FRAME CONTEXT ENHANCEMENT
        # 1. Strong continuity prior using history
        if len(f0_states['period_history']) > 0:
            # Use weighted average of recent periods for stronger prior
            recent_periods = np.array(f0_states['period_history'])
            weights = np.exp(np.linspace(-1, 0, len(recent_periods)))  # More weight on recent frames
            avg_period = np.average(recent_periods, weights=weights)

            sigma = 0.15 * avg_period  # Tighter bandwidth for better continuity
            prior = np.exp(-0.5 * ((lags - avg_period) / (sigma + 1e-6))**2)
            # Stronger fusion for multi-frame context
            score = score * (0.5 + 0.5 * prior)
        elif f0_states['last_period'] is not None and f0_states['last_voiced']:
            # Fallback to single-frame prior
            last_lag = f0_states['last_period']
            sigma = 0.2 * last_lag
            prior = np.exp(-0.5 * ((lags - last_lag) / (sigma + 1e-6))**2)
            score = score * (0.7 + 0.3 * prior)
        else:
            # Preference for typical speech range
            pref_mask = (lags >= 53) & (lags <= 200)
            score[pref_mask] *= 1.05

        # Find initial peak
        k = np.argmax(score)

        # 2. Multi-candidate validation
        # Find top 3 candidates for validation
        top_candidates = np.argsort(score)[-3:][::-1]  # Top 3 in descending order

        best_candidate = k
        best_continuity_score = -1

        if len(f0_states['f0_history']) >= 2:
            # Validate candidates against F0 trajectory
            for candidate_k in top_candidates:
                candidate_period = lags[candidate_k]
                candidate_f0 = np.log2(256.0 / candidate_period) - 1.5

                # Calculate continuity score based on recent F0 history
                f0_hist = np.array(f0_states['f0_history'])
                if len(f0_hist) >= 2:
                    # Predict next F0 based on linear trend
                    trend = f0_hist[-1] - f0_hist[-2] if len(f0_hist) >= 2 else 0
                    predicted_f0 = f0_hist[-1] + trend

                    # Score based on prediction accuracy
                    prediction_error = abs(candidate_f0 - predicted_f0)
                    trend_score = np.exp(-prediction_error / 0.3)  # Gaussian scoring

                    # Combined score: correlation + trend + consistency
                    correlation_score = score[candidate_k] / np.max(score)
                    consistency_score = 1.0 / (1.0 + prediction_error)

                    combined_score = 0.4 * correlation_score + 0.4 * trend_score + 0.2 * consistency_score

                    if combined_score > best_continuity_score:
                        best_continuity_score = combined_score
                        best_candidate = candidate_k

        k = best_candidate

        # Improved quadratic interpolation with peak correction
        if 0 < k < len(r_vals) - 1:
            r_m1 = r_vals[k - 1]
            r_0 = r_vals[k]
            r_p1 = r_vals[k + 1]
            denom = (r_m1 - 2.0 * r_0 + r_p1)
            delta = 0.0
            if abs(denom) > 1e-12:
                delta = 0.5 * (r_m1 - r_p1) / denom
                delta = np.clip(delta, -0.5, 0.5)
            peak_corr = r_0 - 0.25 * (r_m1 - r_p1) * delta
        else:
            delta = 0.0
            peak_corr = r_vals[k]

        period_est = lags[k] + delta
        period_est = np.clip(period_est, min_p, max_p)

        # 3. Enhanced voicing decision with context
        energy_threshold = total / (N + 1e-6) > 1e-6
        corr_threshold = peak_corr > 0.35

        # Context-aware voicing threshold adjustment
        if len(f0_states['correlation_history']) > 0:
            avg_correlation = np.mean(f0_states['correlation_history'])
            # Adaptive threshold based on recent correlation levels
            adaptive_threshold = max(0.25, min(0.45, avg_correlation - 0.1))
            corr_threshold = peak_corr > adaptive_threshold

        voiced = corr_threshold and energy_threshold

        # Advanced voicing hang with multi-frame validation
        if not voiced and f0_states['last_voiced'] and f0_states['voicing_hang'] < f0_states['max_hang']:
            loose_corr = peak_corr > 0.25
            current_f0 = np.log2(256.0 / period_est) - 1.5

            # Multi-frame continuity check
            f0_continuous = True
            if len(f0_states['f0_history']) > 0:
                recent_f0_diff = abs(current_f0 - f0_states['f0_history'][-1])
                if recent_f0_diff > 0.5:
                    f0_continuous = False

            if loose_corr and f0_continuous:
                voiced = True
                f0_states['voicing_hang'] += 1

        # 4. Advanced F0 smoothing with adaptive strength
        raw_dnn_pitch = float(np.log2(256.0 / period_est) - 1.5)

        if voiced and len(f0_states['f0_history']) > 0:
            # Adaptive smoothing based on local F0 variance
            if len(f0_states['f0_history']) >= 3:
                f0_variance = np.var(f0_states['f0_history'][-3:])
                # Higher variance -> stronger smoothing
                smooth_strength = min(0.5, 0.1 + f0_variance * 0.3)
            else:
                smooth_strength = 0.3

            last_f0 = f0_states['f0_history'][-1]
            f0_diff = abs(raw_dnn_pitch - last_f0)

            # Adaptive smoothing threshold
            smooth_threshold = 0.6 if len(f0_states['f0_history']) >= 3 else 0.8

            if f0_diff > smooth_threshold:
                dnn_pitch = smooth_strength * last_f0 + (1.0 - smooth_strength) * raw_dnn_pitch
            else:
                dnn_pitch = raw_dnn_pitch
        else:
            dnn_pitch = raw_dnn_pitch

        # 5. Update multi-frame history
        max_hist = f0_states['max_history']

        # Update F0 history
        if len(f0_states['f0_history']) >= max_hist:
            f0_states['f0_history'].pop(0)
        f0_states['f0_history'].append(dnn_pitch)

        # Update period history
        if len(f0_states['period_history']) >= max_hist:
            f0_states['period_history'].pop(0)
        f0_states['period_history'].append(period_est)

        # Update correlation history
        if len(f0_states['correlation_history']) >= max_hist:
            f0_states['correlation_history'].pop(0)
        f0_states['correlation_history'].append(peak_corr)

        # Update voicing hang state
        if voiced:
            f0_states['voicing_hang'] = 0
        elif not f0_states['last_voiced']:
            f0_states['voicing_hang'] = 0

        f0_states['last_period'] = period_est
        f0_states['last_voiced'] = voiced
        voicing_val = 1.0 if voiced else 0.0

        return dnn_pitch, voicing_val

    def _compute_enhanced_single_frame(self, log_energies: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Compute enhanced features for single frame"""
        enhanced = np.zeros(6, dtype=np.float32)

        energies = 10.0 ** log_energies
        freqs = np.linspace(0, self.sr/2, len(log_energies))

        # Spectral centroid
        total_energy = np.sum(energies) + 1e-10
        enhanced[0] = np.sum(freqs * energies) / total_energy / (self.sr/2)

        # Spectral bandwidth
        mean_freq = enhanced[0] * (self.sr/2)
        enhanced[1] = np.sqrt(np.sum(((freqs - mean_freq)**2) * energies) / total_energy) / (self.sr/2)

        # Spectral rolloff
        cumsum_energy = np.cumsum(energies)
        rolloff_thresh = 0.85 * cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= rolloff_thresh)[0]
        enhanced[2] = rolloff_idx[0] / len(energies) if len(rolloff_idx) > 0 else 1.0

        # Spectral flatness
        geom_mean = 10.0 ** np.mean(log_energies)
        arith_mean = np.mean(energies)
        enhanced[3] = np.clip(geom_mean / (arith_mean + 1e-10), 0.0, 1.0)

        # Spectral entropy
        prob = energies / (total_energy + 1e-10) + 1e-10
        enhanced[4] = -np.sum(prob * np.log(prob)) / np.log(len(prob))

        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
        enhanced[5] = zero_crossings / len(frame)

        return np.clip(enhanced, 0.0, 1.0)

    def _compute_lpc_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compute LPC coefficients for single frame"""
        if self.lpc_mode == 'approx':
            # Would need cepstrum input - skip for streaming efficiency
            return np.zeros(16, dtype=np.float32)

        # Standard autocorrelation method
        windowed = frame * np.hanning(len(frame))
        ac_full = np.correlate(windowed, windowed, mode='full')
        mid = len(ac_full) // 2
        r = ac_full[mid:mid + self.lpc_order + 1]

        if r[0] <= 1e-9:
            return np.zeros(self.lpc_order, dtype=np.float32)

        a, _ = self._levinson_durbin(r, self.lpc_order)

        # Bandwidth expansion
        coeffs = a[1:].copy()
        for i in range(len(coeffs)):
            coeffs[i] *= (0.98 ** (i + 1))

        return coeffs.astype(np.float32)

    def _compute_prosodic_single_frame(self, frame: np.ndarray, dnn_pitch: float, voicing: float) -> np.ndarray:
        """Compute prosodic features for single frame"""
        prosodic = np.zeros(4, dtype=np.float32)

        # F0 trajectory feature
        prosodic[0] = dnn_pitch / 10.0

        # Energy feature
        frame_energy = np.sum(frame**2) / max(1, len(frame))
        prosodic[1] = np.clip(np.log10(max(frame_energy, 1e-8)), -8.0, 0.0)

        # Voicing
        prosodic[2] = voicing

        # Frequency modulation
        if len(frame) > 1:
            diff_frame = np.diff(frame)
            modulation = np.std(diff_frame) / (np.mean(np.abs(frame)) + 1e-10)
            prosodic[3] = np.tanh(modulation)

        return prosodic

    def preemphasis(self, x: np.ndarray) -> np.ndarray:
        """预加重滤波 - 与LPCNet相同（GPU加速路径）。"""
        # torch实现（保留跨帧状态preemph_mem）
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = xt.clone()
        if xt.numel() > 0:
            y[0] = xt[0] + float(self.preemph_mem)
        if xt.numel() > 1:
            y[1:] = xt[1:] - self.preemph_coef * xt[:-1]
        self.preemph_mem = float(-self.preemph_coef * xt[-1].item())
        return y.detach().cpu().numpy()

    def compute_log_spectrum(self, x: np.ndarray) -> np.ndarray:
        """计算对数频谱（GPU加速）。"""
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if xt.numel() != self.win_length:
            xt = torch.nn.functional.pad(xt, (0, max(0, self.win_length - xt.numel())))[:self.win_length]
        windowed = xt * self._hann_win
        X = torch.fft.rfft(windowed, n=self.n_fft)
        power_spec = (X.real.pow(2) + X.imag.pow(2))

        # 频带能量聚合（线性划分）
        band_energies = []
        for (lo, hi) in self._band_slices:
            if hi > lo:
                band_energies.append(power_spec[lo:hi].sum())
            else:
                band_energies.append(power_spec[lo])
        band_energies = torch.stack(band_energies)

        # 对数变换
        log_energies = torch.log10(band_energies + 1e-2)

        # 动态范围控制（逐带，保持行为一致）
        le = log_energies.clone()
        log_max = torch.tensor(-2.0, device=self.device)
        follow = torch.tensor(-2.0, device=self.device)
        out = torch.empty_like(le)
        for i in range(le.numel()):
            out[i] = torch.maximum(log_max - 8.0, torch.maximum(follow - 2.5, le[i]))
            log_max = torch.maximum(log_max, out[i])
            follow = torch.maximum(follow - 2.5, out[i])
        return out.detach().cpu().numpy()

    def compute_ceps_features(self, log_energies: np.ndarray) -> np.ndarray:
        """计算CEPS特征 - DCT-II（GPU加速）。"""
        le = torch.as_tensor(log_energies, dtype=torch.float32, device=self.device)
        ceps_t = self._dct_mat @ le
        ceps_t[0] = ceps_t[0] - 4.0
        return ceps_t.detach().cpu().numpy()

    def compute_lpc_from_frame(self, x: np.ndarray) -> np.ndarray:
        """基于时间域自相关 + Levinson-Durbin 的LPC（更稳定）。"""
        # 窗函数抑制泄露
        xw = x * np.hanning(len(x))
        # 自相关（无偏）
        ac_full = np.correlate(xw, xw, mode='full')
        mid = len(ac_full)//2
        r = ac_full[mid:mid + self.lpc_order + 1].astype(np.float64)
        if r[0] <= 1e-9:
            return np.zeros(self.lpc_order, dtype=np.float32)
        a, _ = self._levinson_durbin(r, self.lpc_order)
        # 带宽扩展提升稳定性
        bw = 0.98
        coeffs = a[1:].copy()
        for i in range(len(coeffs)):
            coeffs[i] *= (bw ** (i + 1))
        return coeffs.astype(np.float32)

    def fargan_style_pitch_detection(self, x: np.ndarray) -> tuple:
        """改进的LPCNet风格音高检测（LPC残差 + GPU向量化NCCF + 插值 + 先验/挂起）。"""
        # 预加重 + LPC残差（改善峰值清晰度，抑制共振峰影响）
        x_pre = self.preemphasis(x.copy())
        x_res = self._lpc_residual(x_pre, order=self.lpc_order)
        min_period, max_period = 32, 255
        N = len(x_res)
        if N <= max_period:
            return -0.5, 0.0

        xt = torch.as_tensor(x_res, dtype=torch.float32, device=self.device)

        # 基于FFT的线性自相关（零填充防止环形卷积）
        nfft = 1
        target_len = 2 * N - 1
        while nfft < target_len:
            nfft <<= 1
        X = torch.fft.rfft(xt, n=nfft)
        ac_full = torch.fft.irfft(X.conj() * X, n=nfft).real
        ac = ac_full[:N]  # lags 0..N-1

        # 计算重叠段能量用于NCCF归一化（矢量化）
        x2 = xt.pow(2)
        csum = torch.cumsum(x2, dim=0)
        total_energy = csum[-1]
        lags = torch.arange(min_period, max_period + 1, device=self.device)
        # e0(L) = sum_{n=0}^{N-L-1} x[n]^2 = csum[N-L-1]
        e0_idx = (N - 1) - lags
        e0 = csum[e0_idx]
        # e1(L) = sum_{n=L}^{N-1} x[n]^2 = total_energy - csum[L-1]
        e1 = total_energy - csum[lags - 1]
        e0 = e0 + 1e-12
        e1 = e1 + 1e-12
        num = ac[lags]
        den = torch.sqrt(e0 * e1)
        r_vals = num / den

        # 次谐波抑制与倍频一致性（简易组合评分）
        score = r_vals.clone()
        # 与2L一致性
        l2 = lags * 2
        valid2 = l2 <= max_period
        score[valid2] = score[valid2] + 0.25 * r_vals[valid2]
        # 与L/2一致性（四舍五入到最近整数）
        lhalf = torch.clamp((lags.float() / 2.0).round().long(), min_period, max_period)
        score = score + 0.20 * r_vals[lhalf - min_period]

        # 连续性先验（上一帧周期高斯先验）
        if self.last_period_est is not None and self.last_voiced:
            last = torch.tensor(self.last_period_est, device=self.device)
            sigma = 0.2 * last  # 20%带宽
            prior = torch.exp(-0.5 * ((lags.float() - last) / (sigma + 1e-6)) ** 2)
            score = score * (0.7 + 0.3 * prior)  # 温和融合
        else:
            # 轻微偏好典型说话人范围（80-300Hz -> 53-200样本）
            pref = (lags >= 53) & (lags <= 200)
            score[pref] = score[pref] * 1.05

        # 最大峰值
        k = int(torch.argmax(score).item())

        # 二次插值
        if 0 < k < r_vals.numel() - 1:
            r_m1 = r_vals[k - 1]
            r_0 = r_vals[k]
            r_p1 = r_vals[k + 1]
            denom = (r_m1 - 2.0 * r_0 + r_p1)
            delta = 0.0
            if torch.abs(denom) > 1e-12:
                delta = 0.5 * (r_m1 - r_p1) / denom
                delta = torch.clamp(delta, -0.5, 0.5)
            peak_corr = r_0 - 0.25 * (r_m1 - r_p1) * delta
        else:
            delta = torch.tensor(0.0, device=self.device)
            peak_corr = r_vals[k]

        period_est = lags[k].float() + float(delta.item())
        period_est = float(torch.clamp(period_est, min_period, max_period).item())
        dnn_pitch = float(torch.log2(torch.tensor(256.0, device=self.device) / period_est) - 1.5)
        dnn_pitch = float(dnn_pitch)
        peak = float(peak_corr.item())
        # 多条件Voicing（峰值 + 残差能量）
        voiced_now = bool(peak > 0.35 and (total_energy.item() / (N + 1e-6) > 1e-6))
        if not voiced_now and self.last_voiced and self.voicing_hang < self.max_voicing_hang and peak > 0.30:
            # 挂起延续以稳定
            voiced_now = True
            self.voicing_hang += 1
        else:
            self.voicing_hang = 0 if voiced_now else self.voicing_hang

        self.last_voiced = voiced_now
        self.last_period_est = period_est
        voicing = 1.0 if voiced_now else 0.0
        return dnn_pitch, voicing

    def extract_enhanced_features(self, x: np.ndarray, log_energies: np.ndarray) -> np.ndarray:
        """提取增强特征 - 我们的扩展部分"""
        enhanced = np.zeros(6, dtype=np.float32)

        # 频谱质心
        freqs = np.linspace(0, self.sr/2, len(log_energies)).astype(np.float64)
        # 反对数：log_energies为log10能量
        energies = (10.0 ** log_energies).astype(np.float64)
        enhanced[0] = np.sum(freqs * energies) / (np.sum(energies) + 1e-10)
        enhanced[0] = enhanced[0] / (self.sr/2)  # 归一化

        # 频谱带宽
        mean_freq = enhanced[0] * (self.sr/2)
        enhanced[1] = np.sqrt(np.sum(((freqs - mean_freq)**2) * energies) / (np.sum(energies) + 1e-10))
        enhanced[1] = enhanced[1] / (self.sr/2)  # 归一化

        # 频谱滚降
        cumsum_energy = np.cumsum(energies)
        total_energy = cumsum_energy[-1]
        rolloff_thresh = 0.85 * total_energy
        rolloff_idx = np.where(cumsum_energy >= rolloff_thresh)[0]
        enhanced[2] = rolloff_idx[0] / len(energies) if len(rolloff_idx) > 0 else 1.0

        # 频谱平坦度
        geom_mean = 10.0 ** (np.mean(log_energies))
        arith_mean = np.mean(energies)
        enhanced[3] = float(geom_mean / (arith_mean + 1e-10))
        enhanced[3] = float(min(max(enhanced[3], 0.0), 1.0))

        # 频谱熵
        prob = energies / (np.sum(energies) + 1e-10)
        prob = prob + 1e-10  # 避免log(0)
        enhanced[4] = -np.sum(prob * np.log(prob))
        enhanced[4] = enhanced[4] / np.log(len(prob))  # 归一化

        # 零交叉率
        zero_crossings = np.sum(np.diff(np.sign(x)) != 0)
        enhanced[5] = float(zero_crossings / len(x))

        # 数值守护：确保在[0,1]
        enhanced = np.clip(enhanced, 0.0, 1.0)

        return enhanced

    def extract_prosodic_features(self, x: np.ndarray, dnn_pitch: float, voicing: float) -> np.ndarray:
        """提取韵律特征 - 我们的扩展部分"""
        prosodic = np.zeros(4)

        # F0轨迹特征 (基于dnn_pitch)
        prosodic[0] = dnn_pitch / 10.0  # 缩放到合理范围

        # 能量特征
        frame_energy = float(np.sum(x**2) / max(1, len(x)))
        # 防止过度下溢/上溢：限定到[-8, 0]
        prosodic[1] = float(np.clip(np.log10(max(frame_energy, 1e-8)), -8.0, 0.0))

        # 语音概率
        prosodic[2] = voicing

        # 频率调制 (简化)
        if len(x) > 1:
            diff_x = np.diff(x)
            modulation = np.std(diff_x) / (np.mean(np.abs(x)) + 1e-10)
            prosodic[3] = np.tanh(modulation)  # 防止过大
        else:
            prosodic[3] = 0.0

        return prosodic

    def extract_frame_features(self, audio_frame: np.ndarray) -> np.ndarray:
        """提取单帧48维特征"""
        features = np.zeros(48)

        # 预加重
        x_preemph = self.preemphasis(audio_frame.copy())

        # 1. CEPS特征 [0:19] (20维，与LPCNet相同)
        log_energies = self.compute_log_spectrum(x_preemph)
        ceps = self.compute_ceps_features(log_energies)
        features[0:20] = ceps

        # 2. F0和Voicing特征 [20:21] (2维)
        dnn_pitch, voicing = self.fargan_style_pitch_detection(audio_frame)
        features[20] = dnn_pitch
        features[21] = voicing

        # 3. 增强频谱特征 [22:27] (6维，我们的扩展)
        enhanced = self.extract_enhanced_features(x_preemph, log_energies)
        features[22:28] = enhanced

        # 4. LPC系数 [28:43] (16维)
        if self.lpc_mode == 'approx':
            lpc = self.compute_lpc_lpcnet_approx(ceps)
        else:
            # 默认：保持当前结果，不改变输出
            lpc = self.compute_lpc_from_frame(x_preemph)
        features[28:44] = lpc

        # 5. 韵律特征 [44:47] (4维，我们的扩展)
        prosodic = self.extract_prosodic_features(audio_frame, dnn_pitch, voicing)
        features[44:48] = prosodic

        return features


def extract_features_from_audio_lpcnet_style(
    audio_path: str,
    output_path: str,
    device: str = 'cpu',
    max_frames: Optional[int] = 10000,
    lpc_mode: str = 'off',
    batched: bool = False,
    batch_frames: int = 10020000,
    peak_margin: int = 2,
    debug_f0: bool = False,
    biquad_stateful: bool = False,
    f0_mode: str = 'residual',
    streaming: bool = False,
) -> int:
    """使用LPCNet风格提取48维特征"""

    print(f"Loading audio from: {audio_path}")

    # 加载音频（batched模式下避免整段读入，基于文件信息推断总帧数）
    if audio_path.endswith('.pcm'):
        sr = 16000
        if batched:
            file_size = Path(audio_path).stat().st_size
            total_samples = file_size // 2
            audio_data = None
        else:
            if max_frames is not None and max_frames > 0:
                needed = int((max_frames - 1) * 160 + 320)
                audio_data = np.fromfile(audio_path, dtype=np.int16, count=needed).astype(np.float32) / 32768.0
            else:
                audio_data = np.fromfile(audio_path, dtype=np.int16).astype(np.float32) / 32768.0
            total_samples = len(audio_data)
        print(f"Audio shape: ({total_samples},), duration: {total_samples/sr:.2f}s")
    else:
        if batched:
            info = torchaudio.info(audio_path)
            sr = info.sample_rate
            if sr != 16000:
                # In streaming mode, avoid resampling full; assume 16k input per project convention
                resampler = torchaudio.transforms.Resample(sr, 16000)
                sr = 16000
            total_samples = info.num_frames
            audio_data = None
            print(f"Audio file: {total_samples} samples, {total_samples/sr:.2f}s")
        else:
            audio_tensor, sr = torchaudio.load(audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_tensor = resampler(audio_tensor)
                sr = 16000
            if audio_tensor.size(0) > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            audio_data = audio_tensor.numpy()
            total_samples = len(audio_data)
            print(f"Audio shape: {audio_data.shape}, duration: {len(audio_data)/sr:.2f}s")

    # 初始化特征提取器
    extractor = LPCNetStyleFeatureExtractor(device=device, lpc_mode=lpc_mode, peak_margin=peak_margin, debug_f0=debug_f0, biquad_stateful=biquad_stateful, f0_mode=f0_mode)

    # 分帧处理
    frame_size = 160  # 10ms
    if batched and audio_data is None:
        n_frames_total = max(0, (total_samples - 320) // frame_size + 1)
    else:
        n_frames_total = max(0, (len(audio_data) - 320) // frame_size + 1)
    if max_frames is not None and max_frames > 0:
        n_frames = min(n_frames_total, max_frames)
        limited_msg = " (limited)"
    else:
        n_frames = n_frames_total
        limited_msg = ""

    if streaming:
        print(f"Extracting {n_frames} / {n_frames_total} frames{limited_msg} (streaming, frame-by-frame) of 48-dim features...")
        # True streaming processing like LPCNet C implementation
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Process in chunks to manage memory but use streaming within chunks
        chunk_size = min(500000, n_frames)  # Process 500k frames at a time max
        fout = open(output_path, 'wb')
        frames_done = 0
        preemph_mem = 0.0

        def write_features(feats_np: np.ndarray):
            fout.write(feats_np.astype(np.float32).tobytes())

        while frames_done < n_frames:
            remaining = n_frames - frames_done
            this_chunk_frames = min(chunk_size, remaining)

            # Calculate sample range for this chunk
            start_sample = frames_done * frame_size
            end_sample = start_sample + this_chunk_frames * frame_size + (320 - frame_size)

            # Load chunk
            if audio_path.endswith('.pcm'):
                with open(audio_path, 'rb') as f:
                    f.seek(start_sample * 2)
                    samples_to_read = min(end_sample - start_sample, total_samples - start_sample)
                    buf = f.read(samples_to_read * 2)
                    chunk = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_chunk, _ = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=end_sample - start_sample)
                if audio_chunk.size(0) > 1:
                    audio_chunk = audio_chunk.mean(dim=0, keepdim=False)
                chunk = audio_chunk.numpy()

            # Process chunk with streaming (constant memory per frame)
            feats_np, preemph_mem = extractor.extract_features_streaming_from_signal(
                chunk, preemph_mem, max_frames=this_chunk_frames
            )

            write_features(feats_np)
            frames_done += feats_np.shape[0]

            if frames_done % 100000 == 0 or frames_done == n_frames:
                print(f"Processed {frames_done}/{n_frames} frames (streaming)")

        fout.close()
        produced_frames = frames_done
        features_array = np.memmap(output_path, dtype=np.float32, mode='r', shape=(produced_frames, 48))
        print(f"Final features shape: {features_array.shape}")

    elif batched:
        print(f"Extracting {n_frames} / {n_frames_total} frames{limited_msg} (batched GPU) of 48-dim features...")
        # Streamed batched processing to limit GPU memory with adaptive downscaling
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Open output for append-binary
        fout = open(output_path, 'wb')
        frames_done = 0
        preemph_mem = 0.0

        def write_chunk(feats_np: np.ndarray):
            fout.write(feats_np.astype(np.float32).tobytes())

        hop = frame_size
        need_overlap = 320
        # Decide source reading strategy
        if audio_path.endswith('.pcm'):
            with open(audio_path, 'rb') as f:
                total_samples = int(Path(audio_path).stat().st_size // 2)
                # Iterate chunks by frames with adaptive downscaling on OOM
                while frames_done < n_frames:
                    target_frames = min(batch_frames, n_frames - frames_done)
                    attempt = 0
                    while True:
                        this_frames = max(1000, target_frames)
                        start_sample = frames_done * hop
                        samples_to_read = this_frames * hop + (need_overlap - hop)
                        f.seek(start_sample * 2)
                        buf = f.read(samples_to_read * 2)
                        if not buf:
                            break
                        chunk = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                        try:
                            # Try streaming first on OOM, then fall back to smaller batches
                            try:
                                feats_np, preemph_mem = extractor.extract_features_batched_from_signal(chunk, preemph_mem, max_frames=this_frames)
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    print(f"GPU OOM on batch {this_frames}, falling back to streaming...")
                                    feats_np, preemph_mem = extractor.extract_features_streaming_from_signal(chunk, preemph_mem, max_frames=this_frames)
                                else:
                                    raise

                            write_chunk(feats_np)
                            frames_done += feats_np.shape[0]
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            if frames_done % 100000 == 0 or frames_done == n_frames:
                                print(f"Processed {frames_done}/{n_frames} frames (batched)")
                            break
                        except RuntimeError as e:
                            msg = str(e)
                            if 'out of memory' in msg.lower() or 'cuda oom' in msg.lower():
                                attempt += 1
                                if this_frames <= 1000 or attempt >= 5:
                                    print("Falling back to pure streaming mode due to persistent OOM...")
                                    feats_np, preemph_mem = extractor.extract_features_streaming_from_signal(chunk, preemph_mem, max_frames=this_frames)
                                    write_chunk(feats_np)
                                    frames_done += feats_np.shape[0]
                                    break
                                target_frames = max(1000, this_frames // 2)
                                print(f"CUDA OOM on batch of {this_frames} frames; retrying with {target_frames}...")
                                continue
                            else:
                                raise
        else:
            # Use torchaudio incremental loads with frame_offset
            frame_offset = 0
            while frames_done < n_frames:
                target_frames = min(batch_frames, n_frames - frames_done)
                attempt = 0
                while True and frames_done < n_frames:
                    this_frames = max(1000, target_frames)
                    start_sample = frame_offset * hop
                    samples_to_read = this_frames * hop + (need_overlap - hop)
                    audio_chunk, _ = torchaudio.load(audio_path, frame_offset=start_sample, num_frames=samples_to_read)
                    if audio_chunk.size(0) > 1:
                        audio_chunk = audio_chunk.mean(dim=0, keepdim=False)
                    chunk = audio_chunk.numpy()
                    try:
                        # Try batched first, fall back to streaming on OOM
                        try:
                            feats_np, preemph_mem = extractor.extract_features_batched_from_signal(chunk, preemph_mem, max_frames=this_frames)
                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                print(f"GPU OOM on batch {this_frames}, falling back to streaming...")
                                feats_np, preemph_mem = extractor.extract_features_streaming_from_signal(chunk, preemph_mem, max_frames=this_frames)
                            else:
                                raise

                        write_chunk(feats_np)
                        frames_done += feats_np.shape[0]
                        frame_offset += this_frames
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        if frames_done % 100000 == 0 or frames_done == n_frames:
                            print(f"Processed {frames_done}/{n_frames} frames (batched)")
                        break
                    except RuntimeError as e:
                        msg = str(e)
                        if 'out of memory' in msg.lower() or 'cuda oom' in msg.lower():
                            attempt += 1
                            if this_frames <= 1000 or attempt >= 5:
                                print("Falling back to pure streaming mode due to persistent OOM...")
                                feats_np, preemph_mem = extractor.extract_features_streaming_from_signal(chunk, preemph_mem, max_frames=this_frames)
                                write_chunk(feats_np)
                                frames_done += feats_np.shape[0]
                                frame_offset += this_frames
                                break
                            target_frames = max(1000, this_frames // 2)
                            print(f"CUDA OOM on batch of {this_frames} frames; retrying with {target_frames}...")
                            continue
                        else:
                            raise
        fout.close()
        # For reporting, try to map the file back minimally to get shapes (avoid loading all)
        produced_frames = frames_done
        features_array = np.memmap(output_path, dtype=np.float32, mode='r', shape=(produced_frames, 48))
        print(f"Final features shape: {features_array.shape}")
    else:
        print(f"Extracting {n_frames} / {n_frames_total} frames{limited_msg} of 48-dim features...")
        all_features = []
        for i in range(n_frames):
            start = i * frame_size
            end = start + 320  # 20ms窗口
            if end > len(audio_data):
                break
            frame = audio_data[start:end]
            features = extractor.extract_frame_features(frame)
            all_features.append(features)
            if i % 10000 == 0:
                print(f"Processed {i}/{n_frames} frames")
        features_array = np.array(all_features, dtype=np.float32)
        print(f"Final features shape: {features_array.shape}")

    # 保存特征
    print(f"Saving features to: {output_path}")
    if not batched and not streaming:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        features_array.tofile(output_path)

    # 验证文件
    expected_size = features_array.shape[0] * 48 * 4
    actual_size = Path(output_path).stat().st_size
    print(f"Expected file size: {expected_size} bytes")
    print(f"Actual file size: {actual_size} bytes")

    if expected_size == actual_size:
        print("✅ Feature extraction successful!")

        # 验证F0特征质量 + 其他特征安全范围
        print(f"\n🔍 验证F0特征质量...")
        ncheck = min(5000, features_array.shape[0])
        f0_features = features_array[:ncheck, 20]
        f0_temporal_var = f0_features.var()
        unique_ratio = len(np.unique(f0_features)) / max(1, len(f0_features))
        print(f"F0时间方差: {f0_temporal_var:.6f}")
        print(f"F0唯一值比例: {unique_ratio:.4f}")
        if f0_temporal_var > 0.05 and unique_ratio > 0.5:
            print("✅ F0特征质量良好！")
        else:
            print("⚠️ F0特征质量需要改进")

        print("\n🔎 其他特征安全范围检测 (前5000帧或更少)...")
        def stats_block(name, data):
            dmin = float(np.min(data)); dmax = float(np.max(data))
            dmean = float(np.mean(data)); dstd = float(np.std(data))
            print(f"=== {name} ===")
            print(f"范围: [{dmin:.4f}, {dmax:.4f}]  均值: {dmean:.4f}  标准差: {dstd:.4f}")

        safe = {
            'CEPS': (-20.0, 20.0),
            'F0': (-1.55, 1.55),
            'Voicing': (0.0, 1.0),
            'Enhanced': (0.0, 1.0),
            'LPC': (-3.0, 3.0),
            'Pros0': (-0.2, 0.2),
            'Pros1': (-8.0, 0.0),
            'Pros2': (0.0, 1.0),
            'Pros3': (-1.0, 1.0),
        }

        # CEPS
        ceps = features_array[:ncheck, 0:20]
        stats_block('CEPS_0-19 (DCT对数频谱)', ceps)
        ceps_out = float(np.mean((ceps < safe['CEPS'][0]) | (ceps > safe['CEPS'][1])))
        print(f"越界比例(|x|>20): {ceps_out:.6f}")
        if ceps_out > 0.01:
            print("⚠️ CEPS越界比例较高，注意动态范围或归一化")

        # Voicing
        vc = features_array[:ncheck, 21]
        stats_block('Voicing (语音判定)', vc)
        frac0 = float(np.mean(vc == 0.0)); frac1 = float(np.mean(vc == 1.0))
        frac_other = 1.0 - frac0 - frac1
        print(f"0比例: {frac0:.3f}  1比例: {frac1:.3f}  其他: {frac_other:.6f}")
        if frac_other > 0.001:
            print("⚠️ Voicing存在非{0,1}值，检查判定逻辑/阈值")

        # Enhanced (0..1)
        enh = features_array[:ncheck, 22:28]
        stats_block('Enhanced_0-5 (增强频谱)', enh)
        low, high = safe['Enhanced']
        out_ratio = float(np.mean((enh < low) | (enh > high)))
        print(f"越界比例(应在[0,1]): {out_ratio:.6f}")
        for i in range(enh.shape[1]):
            di = enh[:, i]
            dmin, dmax = float(np.min(di)), float(np.max(di))
            outi = float(np.mean((di < low) | (di > high)))
            print(f"  Enhanced[{i}] min={dmin:.4f} max={dmax:.4f} 越界={outi:.6f}")
        if out_ratio > 0.01:
            print("⚠️ 增强特征越界比例较高，检查归一化/尺度")

        # LPC
        lpc = features_array[:ncheck, 28:44]
        stats_block('LPC_0-15 (线性预测)', lpc)
        lpc_out = float(np.mean((lpc < safe['LPC'][0]) | (lpc > safe['LPC'][1])))
        print(f"越界比例(|x|>3): {lpc_out:.6f}")
        if lpc_out > 0.01:
            print("⚠️ LPC系数幅度偏大，可能不稳定")

        # Prosodic
        pro = features_array[:ncheck, 44:48]
        stats_block('Prosodic_0-3 (韵律)', pro)
        p0_out = float(np.mean((pro[:, 0] < safe['Pros0'][0]) | (pro[:, 0] > safe['Pros0'][1])))
        p1_out = float(np.mean((pro[:, 1] < safe['Pros1'][0]) | (pro[:, 1] > safe['Pros1'][1])))
        p2_frac0 = float(np.mean(pro[:, 2] == 0.0)); p2_frac1 = float(np.mean(pro[:, 2] == 1.0))
        p2_other = 1.0 - p2_frac0 - p2_frac1
        p3_out = float(np.mean((pro[:, 3] < safe['Pros3'][0]) | (pro[:, 3] > safe['Pros3'][1])))
        print(f"  Pros[0]=F0/10 越界比例: {p0_out:.6f}")
        print(f"  Pros[1]=log10(E) 越界比例: {p1_out:.6f}")
        print(f"  Pros[2]=voicing  0比例 {p2_frac0:.3f} 1比例 {p2_frac1:.3f} 其他 {p2_other:.6f}")
        print(f"  Pros[3]=tanh(mod) 越界比例: {p3_out:.6f}")
        if p1_out > 0.05:
            print("⚠️ Prosodic[1] 能量分布异常，注意输入幅度/静音段")
    else:
        print("❌ File size mismatch!")

    return features_array.shape[0]


def main():
    parser = argparse.ArgumentParser(description='Extract 48-dim features using LPCNet style')
    parser.add_argument('--audio-path', default='/home/bluestar/FARGAN/opus/data_cn/out_speech.pcm',
                       help='Input audio file path')
    parser.add_argument('--output-path', default='/tmp/test_features_48_fixed.f32',
                       help='Output features file path')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--max-frames', type=int, default=10000,
                        help='Max frames to extract (default 10000). Set 0 to process all frames.')
    parser.add_argument('--lpcnet-lpc', choices=['off','approx'], default='off',
                        help='Use LPCNet-style LPC recovery (approx). Default off to preserve outputs.')
    parser.add_argument('--batched', action='store_true', help='Enable batched GPU pipeline for speed')
    parser.add_argument('--batch-frames', type=int, default=10020000, help='Frames per GPU batch when --batched (memory control)')
    parser.add_argument('--peak-margin', type=int, default=2, help='Internal-window peak margin (avoid boundary lags)')
    parser.add_argument('--debug-f0', action='store_true', help='Print one-time F0 debug metrics for first batch')
    parser.add_argument('--biquad-stateful', action='store_true', help='Carry biquad filter state across batches')
    parser.add_argument('--f0-mode', choices=['residual','raw','cpu-raw'], default='residual', help='F0 path: residual (LPCNet-like), raw (GPU), or cpu-raw (per-frame CPU)')
    parser.add_argument('--streaming', action='store_true', help='Use streaming frame-by-frame processing like LPCNet C (prevents OOM)')

    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        print(f"❌ Input file not found: {args.audio_path}")
        return

    try:
        n_frames = extract_features_from_audio_lpcnet_style(
            args.audio_path,
            args.output_path,
            args.device,
            max_frames=(None if args.max_frames is None else (args.max_frames if args.max_frames > 0 else None)),
            lpc_mode=args.lpcnet_lpc,
            batched=bool(args.batched),
            batch_frames=int(args.batch_frames),
            peak_margin=int(args.peak_margin),
            debug_f0=bool(args.debug_f0),
            biquad_stateful=bool(args.biquad_stateful),
            f0_mode=str(args.f0_mode),
            streaming=bool(args.streaming)
        )

        print(f"\n🎉 Feature extraction completed!")
        print(f"   Total frames: {n_frames}")
        if n_frames > 0:
            file_size_mb = Path(args.output_path).stat().st_size / (1024**2)
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Features file: {args.output_path}")

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
