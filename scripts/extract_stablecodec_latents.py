#!/usr/bin/env python3
"""
离线提取StableCodec latent脚本

使用独立环境运行，为JSCC训练提供teacher latent
适配data_expert_augmented_small200k数据集的PCM格式

使用方法:
    conda activate stablecodec-offline  # 激活独立环境
    python scripts/extract_stablecodec_latents.py --data-root ./data_expert_augmented_small200k --output-dir ./teacher_latents
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import time

import torch
import torchaudio
import numpy as np
from tqdm import tqdm


def setup_imports():
    """设置StableCodec导入"""
    try:
        from stable_codec.model import StableCodec
        return StableCodec
    except ImportError as e:
        print(f"❌ StableCodec导入失败: {e}")
        print("请确保在stablecodec-offline环境中运行:")
        print("  conda activate stablecodec-offline")
        print("  python scripts/extract_stablecodec_latents.py ...")
        sys.exit(1)


def load_pcm_file(pcm_path: Path, sample_rate: int = 16000) -> torch.Tensor:
    """
    加载PCM文件为torch tensor

    Args:
        pcm_path: PCM文件路径
        sample_rate: 采样率

    Returns:
        audio: [T] float32 tensor, normalized to [-1,1]
    """
    # PCM文件是raw 16-bit signed integer
    with open(pcm_path, 'rb') as f:
        raw_data = f.read()

    # 转换为numpy数组
    audio_int16 = np.frombuffer(raw_data, dtype=np.int16)

    # 归一化到[-1,1]
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # 转为torch tensor
    audio_tensor = torch.from_numpy(audio_float)

    return audio_tensor


def generate_audio_key(pcm_path: Path, data_root: Path) -> str:
    """
    生成统一的audio_key

    例子:
        pcm_path: /path/to/data_expert_augmented_small200k/harmonic_200k.pcm
        data_root: /path/to/data_expert_augmented_small200k
        -> audio_key: "harmonic_200k"
    """
    rel_path = pcm_path.relative_to(data_root)
    audio_key = str(rel_path.with_suffix(''))  # 去掉.pcm扩展名
    return audio_key


def find_pcm_files(data_root: Path) -> List[Path]:
    """查找所有PCM文件（递归）"""
    pcm_files = list(data_root.rglob("*.pcm"))
    pcm_files.sort()
    return pcm_files


def estimate_frames_and_duration(audio_tensor: torch.Tensor, sample_rate: int) -> Tuple[int, int, float]:
    """
    估计StableCodec latent的参数

    Returns:
        audio_samples: 音频样本数
        estimated_frames: 估计的StableCodec帧数 (~25Hz)
        duration_sec: 时长(秒)
    """
    audio_samples = audio_tensor.size(0)
    duration_sec = audio_samples / sample_rate

    # StableCodec约25Hz帧率
    stablecodec_frame_rate = 25.0
    estimated_frames = int(duration_sec * stablecodec_frame_rate)

    return audio_samples, estimated_frames, duration_sec


def upsample_to_fargan_rate(latent_25hz: torch.Tensor, fargan_rate: float = 100.0) -> torch.Tensor:
    """
    将25Hz latent上采样到FARGAN帧率(100Hz)

    Args:
        latent_25hz: [T_25hz, D] StableCodec latent at ~25Hz
        fargan_rate: FARGAN帧率

    Returns:
        latent_100hz: [T_100hz, D] 上采样后的latent
    """
    if latent_25hz.size(0) <= 1:
        return latent_25hz

    # 计算上采样倍数
    scale_factor = fargan_rate / 25.0  # 通常是4倍

    # [T_25hz, D] -> [1, D, T_25hz] -> interpolate -> [1, D, T_100hz] -> [T_100hz, D]
    latent_1dt = latent_25hz.transpose(0, 1).unsqueeze(0)  # [1, D, T_25hz]

    latent_up_1dt = torch.nn.functional.interpolate(
        latent_1dt,
        scale_factor=scale_factor,
        mode='linear',
        align_corners=False
    )

    latent_100hz = latent_up_1dt.squeeze(0).transpose(0, 1)  # [T_100hz, D]

    return latent_100hz


def extract_latents_for_pcm(pcm_path: Path,
                           stablecodec_model,
                           data_root: Path,
                           output_dir: Path,
                           upsample_to_fargan: bool = True,
                           fargan_rate: float = 100.0) -> Dict:
    """
    为单个PCM文件提取StableCodec latent

    Returns:
        结果字典包含统计信息
    """
    audio_key = generate_audio_key(pcm_path, data_root)
    output_path = output_dir / f"{audio_key}.pt"

    # 如果已存在，跳过
    if output_path.exists():
        return {"status": "skipped", "audio_key": audio_key}

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 加载音频
        audio_tensor = load_pcm_file(pcm_path, sample_rate=16000)
        audio_samples, est_frames, duration = estimate_frames_and_duration(audio_tensor, 16000)

        # 为StableCodec准备输入 [1, 1, T]
        dev = next(stablecodec_model.parameters()).device
        audio_batch = audio_tensor.unsqueeze(0).unsqueeze(0).to(dev)

        # 提取latent
        with torch.no_grad():
            # StableCodec.encode返回 (pre_bottleneck_latents, tokens)
            # pre_bottleneck_latents: [B, H, S] where S is ~25Hz frames
            pre_latents, _tokens = stablecodec_model.encode(audio_batch, posthoc_bottleneck=False)

            # [B, H, S] -> [S, H] (去掉batch维度并转置)
            latent_25hz = pre_latents.squeeze(0).transpose(0, 1).contiguous()  # [S, H]

        # 可选：上采样到FARGAN帧率
        if upsample_to_fargan:
            latent_final = upsample_to_fargan_rate(latent_25hz, fargan_rate)
            frame_info = f"25Hz({latent_25hz.size(0)}) -> {fargan_rate}Hz({latent_final.size(0)})"
        else:
            latent_final = latent_25hz
            frame_info = f"25Hz({latent_final.size(0)})"

        # 保存到磁盘
        torch.save(latent_final.cpu(), output_path)

        return {
            "status": "success",
            "audio_key": audio_key,
            "duration_sec": duration,
            "audio_samples": audio_samples,
            "latent_shape": list(latent_final.shape),
            "frame_info": frame_info,
            "output_path": str(output_path)
        }

    except Exception as e:
        return {
            "status": "error",
            "audio_key": audio_key,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="离线提取StableCodec teacher latents")
    parser.add_argument("--data-root", type=str, required=True,
                      help="数据集根目录 (e.g., ./data_expert_augmented_small200k)")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="输出latent目录 (e.g., ./teacher_latents)")
    parser.add_argument("--pretrained-model", type=str,
                      default="stabilityai/stable-codec-speech-16k",
                      help="StableCodec预训练模型")
    parser.add_argument("--upsample-to-fargan", action="store_true", default=True,
                      help="是否上采样到FARGAN帧率(100Hz) (当前默认开启)")
    parser.add_argument("--fargan-rate", type=float, default=100.0,
                      help="FARGAN帧率(Hz)")
    parser.add_argument("--device", type=str, default="auto",
                      help="计算设备 (cuda/cpu/auto)")

    args = parser.parse_args()

    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🚀 StableCodec Latent提取器")
    print(f"📂 数据根目录: {args.data_root}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📱 设备: {device}")
    print(f"🎵 目标帧率: {args.fargan_rate}Hz (上采样: {args.upsample_to_fargan})")

    # 导入StableCodec
    StableCodec = setup_imports()

    # 检查数据目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找PCM文件
    pcm_files = find_pcm_files(data_root)
    if not pcm_files:
        print(f"❌ 在 {data_root} 中未找到PCM文件")
        sys.exit(1)

    print(f"📊 找到 {len(pcm_files)} 个PCM文件:")
    for pcm in pcm_files:
        print(f"  - {pcm.name}")

    # 加载StableCodec模型
    print(f"🔧 加载StableCodec模型: {args.pretrained_model}")
    try:
        stablecodec = StableCodec(
            pretrained_model=args.pretrained_model,
            device=device
        )
        stablecodec.eval().requires_grad_(False)
        print(f"✅ StableCodec加载成功! 采样率: {stablecodec.sample_rate}Hz")
    except Exception as e:
        print(f"❌ StableCodec加载失败: {e}")
        sys.exit(1)

    # 处理每个PCM文件
    print(f"🔄 开始提取latents...")
    results = []
    start_time = time.time()

    for pcm_path in tqdm(pcm_files, desc="提取latents"):
        result = extract_latents_for_pcm(
            pcm_path=pcm_path,
            stablecodec_model=stablecodec,
            data_root=data_root,
            output_dir=output_dir,
            upsample_to_fargan=args.upsample_to_fargan,
            fargan_rate=args.fargan_rate
        )
        results.append(result)

        # 打印进度
        if result["status"] == "success":
            tqdm.write(f"✅ {result['audio_key']}: {result['frame_info']}, "
                      f"{result['duration_sec']:.1f}s")
        elif result["status"] == "error":
            tqdm.write(f"❌ {result['audio_key']}: {result['error']}")
        else:  # skipped
            tqdm.write(f"⏭️  {result['audio_key']}: 已存在")

    # 统计结果
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    skip_count = sum(1 for r in results if r["status"] == "skipped")

    print(f"\n📊 提取完成!")
    print(f"⏱️  总耗时: {total_time:.1f}秒")
    print(f"✅ 成功: {success_count}")
    print(f"⏭️  跳过: {skip_count}")
    print(f"❌ 失败: {error_count}")

    if error_count > 0:
        print(f"\n❌ 失败详情:")
        for r in results:
            if r["status"] == "error":
                print(f"  {r['audio_key']}: {r['error']}")

    # 保存提取日志
    log_path = output_dir / "extraction_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"StableCodec Latent提取日志\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据根目录: {args.data_root}\n")
        f.write(f"输出目录: {args.output_dir}\n")
        f.write(f"设备: {device}\n")
        f.write(f"上采样到FARGAN: {args.upsample_to_fargan} ({args.fargan_rate}Hz)\n")
        f.write(f"总文件数: {len(pcm_files)}\n")
        f.write(f"成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}\n")
        f.write(f"总耗时: {total_time:.1f}秒\n\n")

        f.write("详细结果:\n")
        for r in results:
            f.write(f"{r}\n")

    print(f"📝 提取日志已保存: {log_path}")


if __name__ == "__main__":
    main()
