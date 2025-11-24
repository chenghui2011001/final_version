"""
StableCodec Teacher wrapper for hash-bottleneck distillation.

设计目标（对照 stage5_docs/3.md）：
1. 当依赖满足时，直接调用 external_repos/stable-codec 提供的 StableCodec 模型做推理，
   只用于 encode（提取 pre-bottleneck latent），不参与训练。
2. 当依赖/环境不匹配（缺少 stable-audio-tools、预训练权重等）时，优雅降级为轻量 mock，
   或使用离线预计算 latent（PrecomputedStableCodecTeacher）。
3. 提供统一接口：encode_latent(audio, align_to_fargan=True) → [B, T_frames, D_teacher]。
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio  # 用于可选的重采样


def _import_stablecodec_class() -> Optional[type]:
    """
    尝试从 external_repos/stable-codec 中导入 StableCodec 类。
    如果 stable-audio-tools 或其它依赖缺失，则返回 None。
    """
    try:
        # 先尝试直接导入（用户可能已经 pip 安装过 stable-codec）
        from stable_codec.model import StableCodec  # type: ignore
        return StableCodec
    except Exception:
        pass

    # 再尝试从本仓库 external_repos 中导入
    try:
        repo_root = Path(__file__).resolve().parents[2] / "external_repos" / "stable-codec"
        if repo_root.is_dir():
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from stable_codec.model import StableCodec  # type: ignore
            return StableCodec
    except Exception:
        return None

    return None


class StableCodecTeacher(nn.Module):
    """
    StableCodec Teacher 封装：

    - 优先使用 external_repos/stable-codec 的 StableCodec 进行推理；
    - 如果依赖不满足或实例化失败，则自动降级为轻量 Mock encoder；
    - encode_latent() 始终返回 [B, T_frames, D] 的连续 latent。
    """

    def __init__(
        self,
        pretrained_model: Optional[str] = "stabilityai/stable-codec-speech-16k",
        model_config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_mock_if_unavailable: bool = True,
        fargan_frame_rate: float = 100.0,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.fargan_frame_rate = float(fargan_frame_rate)
        self.backend = "mock"  # "stablecodec" 或 "mock"
        self.sample_rate = 16000  # 默认值，后面根据 StableCodec 覆盖
        self.teacher_dim: Optional[int] = None

        StableCodecCls = _import_stablecodec_class()
        self._sc = None

        if StableCodecCls is not None and (pretrained_model or model_config_path):
            try:
                # 尝试构造 StableCodec 实例（只做推理）
                kwargs_sc: Dict[str, Any] = {"device": self.device}
                if pretrained_model is not None:
                    kwargs_sc["pretrained_model"] = pretrained_model
                else:
                    kwargs_sc["model_config_path"] = model_config_path
                    if ckpt_path is not None:
                        kwargs_sc["ckpt_path"] = ckpt_path

                self._sc = StableCodecCls(**kwargs_sc)  # type: ignore
                self._sc.eval().requires_grad_(False)
                # sample_rate / teacher_dim 从模型配置中读取
                try:
                    self.sample_rate = int(getattr(self._sc, "sample_rate", self.sample_rate))
                except Exception:
                    pass

                self.backend = "stablecodec"
                warnings.warn(
                    f"StableCodecTeacher: using StableCodec backend "
                    f"(pretrained_model={pretrained_model}, sample_rate={self.sample_rate})."
                )
            except Exception as e:
                # 依赖不满足或加载失败，降级到 mock
                if not use_mock_if_unavailable:
                    raise RuntimeError(f"Failed to initialise StableCodecTeacher backend: {e}") from e
                warnings.warn(
                    f"StableCodecTeacher: failed to init StableCodec backend ({e}); "
                    f"falling back to mock encoder."
                )
                self.backend = "mock"
        else:
            if not use_mock_if_unavailable:
                raise RuntimeError(
                    "StableCodecTeacher: StableCodec backend unavailable "
                    "(missing stable-codec / stable-audio-tools); "
                    "set use_mock_if_unavailable=True or use PrecomputedStableCodecTeacher."
                )
            warnings.warn(
                "StableCodecTeacher: stable-codec dependencies not available; using mock encoder."
            )
            self.backend = "mock"

        # Mock encoder（仅在 backend == 'mock' 时使用）
        if self.backend == "mock":
            # 约 25Hz：stride 640 at 16kHz → 25 frames/s
            self.mock_encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=1024, stride=640, padding=192),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 6, kernel_size=1, stride=1, padding=0),
            )
            self.teacher_dim = 6

    def _preprocess_audio_tensor(self, audio: torch.Tensor) -> torch.Tensor:
        """
        预处理音频 tensor 到 StableCodec 期望的形状 [B, 1, T]。
        不做重采样，假定外部已提供与 StableCodec 一致的采样率（通常 16kHz）。
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, T]
        if audio.dim() == 2:
            # [B, T] -> [B, 1, T]
            audio = audio.unsqueeze(1)
        if audio.dim() == 3 and audio.size(1) != 1:
            # 多通道转单通道
            audio = audio.mean(dim=1, keepdim=True)
        return audio.to(self.device)

    @torch.no_grad()
    def encode_latent(
        self,
        audio: Union[torch.Tensor, str],
        align_to_fargan: bool = True,
        posthoc_bottleneck: bool = False,
    ) -> torch.Tensor:
        """
        提取 StableCodec pre-bottleneck latent。

        Args:
            audio: [B, T] / [B, 1, T] 或 wav 路径（str）
            align_to_fargan: 是否上采样到 FARGAN 帧率（默认 100Hz）
            posthoc_bottleneck: 是否使用 posthoc bottleneck（与 stable-codec 一致）

        Returns:
            latent: [B, T_frames, D]，若 align_to_fargan=True，则 T_frames≈T_audio/160
        """
        # 1) 读取/预处理音频
        if isinstance(audio, str):
            # 本地文件路径，使用 torchaudio 读取并转换为 StableCodec 采样率
            wav, sr = torchaudio.load(audio)
            if sr != self.sample_rate:
                try:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                except Exception:
                    # 退化为最近邻 up/down-sample
                    target_len = int(wav.shape[-1] * self.sample_rate / sr)
                    wav = F.interpolate(
                        wav.unsqueeze(1),
                        size=target_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)
            audio_tensor = wav.to(self.device)
        else:
            audio_tensor = audio

        audio_tensor = self._preprocess_audio_tensor(audio_tensor)  # [B,1,T]
        B, _, T_audio = audio_tensor.shape

        # 2) 根据 backend 提取 latent
        if self.backend == "stablecodec" and self._sc is not None:
            # StableCodec.encode 返回 (pre_bottleneck_latents, tokens)
            # latents: [B, H, S]，S ≈ T_audio / hop_size
            latents, _tokens = self._sc.encode(  # type: ignore
                audio_tensor,
                posthoc_bottleneck=posthoc_bottleneck,
            )
            # [B, H, S] -> [B, S, H]
            latent_bts = latents.transpose(1, 2).contiguous()
            self.teacher_dim = latent_bts.size(-1)
        else:
            # Mock encoder: [B,1,T] -> [B, D_mock, S_mock]
            feats = self.mock_encoder(audio_tensor)  # [B, D, S]
            latent_bts = feats.transpose(1, 2).contiguous()  # [B, S, D]

        # 3) 可选：对齐到 FARGAN 帧率（约 100Hz）
        if align_to_fargan and self.fargan_frame_rate > 0:
            # target_frames ≈ T_audio / (sample_rate / fargan_frame_rate)
            frames_per_second = self.fargan_frame_rate
            target_T = int(round(T_audio / self.sample_rate * frames_per_second))
            if target_T <= 1:
                return latent_bts
            latent_btC = latent_bts.transpose(1, 2)  # [B,C,S]
            latent_up = F.interpolate(
                latent_btC,
                size=target_T,
                mode="linear",
                align_corners=False,
            )
            latent_bts = latent_up.transpose(1, 2).contiguous()

        return latent_bts

    @torch.no_grad()
    def get_latent_stats(self, latent: torch.Tensor) -> Dict[str, float]:
        """分析 latent 统计信息，便于调试/监控。"""
        return {
            "mean": float(latent.mean().item()),
            "std": float(latent.std().item()),
            "min": float(latent.min().item()),
            "max": float(latent.max().item()),
            "norm": float(latent.norm().item()),
        }


class PrecomputedStableCodecTeacher(nn.Module):
    """
    预计算版StableCodec Teacher

    使用离线预计算的StableCodec latent，避免运行时依赖
    适用于训练时已经提取好teacher latent的场景
    """

    def __init__(self,
                 teacher_dim: int = 6,
                 cache_dir: str = './teacher_cache'):
        super().__init__()

        self.teacher_dim = teacher_dim
        self.cache_dir = cache_dir
        self.latent_cache = {}

        os.makedirs(cache_dir, exist_ok=True)

    def cache_latent(self, audio_key: str, latent: torch.Tensor):
        """缓存teacher latent"""
        cache_path = os.path.join(self.cache_dir, f"{audio_key}.pt")
        torch.save(latent.cpu(), cache_path)
        self.latent_cache[audio_key] = latent

    def load_latent(self, audio_key: str) -> Optional[torch.Tensor]:
        """加载缓存的teacher latent"""
        if audio_key in self.latent_cache:
            return self.latent_cache[audio_key]

        cache_path = os.path.join(self.cache_dir, f"{audio_key}.pt")
        if os.path.exists(cache_path):
            latent = torch.load(cache_path, map_location='cpu')
            self.latent_cache[audio_key] = latent
            return latent

        return None

    def encode_latent(self, audio_key: str) -> torch.Tensor:
        """通过audio_key获取预计算的latent"""
        latent = self.load_latent(audio_key)
        if latent is None:
            raise ValueError(f"未找到audio_key '{audio_key}' 的teacher latent")
        return latent


class StableCodecDistillationLoss(nn.Module):
    """
    StableCodec蒸馏损失

    计算Hash bottleneck输出与StableCodec teacher之间的蒸馏损失
    """

    def __init__(self,
                 temperature: float = 1.0,
                 feature_weight: float = 1.0,
                 cosine_weight: float = 0.5,
                 contrastive_weight: float = 0.3):
        super().__init__()

        self.temperature = temperature
        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.contrastive_weight = contrastive_weight

    def forward(self,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失

        Args:
            student_features: Hash decoder输出 [B, T, D_s]
            teacher_features: StableCodec latent [B, T, D_t]

        Returns:
            损失字典
        """
        # 如果维度不匹配，需要投影
        if student_features.size(-1) != teacher_features.size(-1):
            # 简单线性投影 (实际使用中可能需要更复杂的对齐)
            if not hasattr(self, 'projection'):
                self.projection = nn.Linear(
                    student_features.size(-1),
                    teacher_features.size(-1)
                ).to(student_features.device)
            student_projected = self.projection(student_features)
        else:
            student_projected = student_features

        losses = {}

        # 1. 特征重建损失
        losses['feature_mse'] = F.mse_loss(student_projected, teacher_features)

        # 2. 余弦相似度损失
        cos_sim = F.cosine_similarity(student_projected, teacher_features, dim=-1)
        losses['cosine_loss'] = 1 - cos_sim.mean()

        # 3. 对比学习损失 (时序上下文)
        if teacher_features.size(1) > 1:
            # 正样本：同一时刻
            pos_sim = F.cosine_similarity(student_projected, teacher_features, dim=-1)

            # 负样本：时移
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=1)
            neg_sim = F.cosine_similarity(student_projected, neg_teacher, dim=-1)

            logits = torch.stack([pos_sim, neg_sim], dim=-1) / self.temperature
            targets = torch.zeros(logits.size(0), logits.size(1),
                                dtype=torch.long, device=logits.device)

            losses['contrastive'] = F.cross_entropy(
                logits.view(-1, 2), targets.view(-1)
            )
        else:
            losses['contrastive'] = torch.tensor(0.0, device=student_features.device)

        # 总损失
        losses['total'] = (
            self.feature_weight * losses['feature_mse'] +
            self.cosine_weight * losses['cosine_loss'] +
            self.contrastive_weight * losses['contrastive']
        )

        return losses


if __name__ == "__main__":
    # 简单自测：在当前环境下检查 StableCodecTeacher 是否能工作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧪 测试 StableCodecTeacher ...")

    # 构造 teacher（可能使用 StableCodec，也可能回退到 mock）
    teacher = StableCodecTeacher(device=device)

    # 模拟音频数据（2 秒 16kHz 噪声）
    sample_rate = 16000
    duration = 2.0
    audio = torch.randn(1, sample_rate * int(duration), device=device)

    latent = teacher.encode_latent(audio, align_to_fargan=True)
    print(f"输入音频: {audio.shape}, 输出 latent: {latent.shape}")
    print("Latent 统计:", teacher.get_latent_stats(latent))

    # 蒸馏损失 quick check
    distill = StableCodecDistillationLoss().to(device)
    student = torch.randn_like(latent)
    losses = distill(student, latent)
    print("蒸馏损失示例:", {k: float(v.item()) for k, v in losses.items()})

    print("✅ StableCodecTeacher 自测完成。")
