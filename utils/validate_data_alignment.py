#!/usr/bin/env python3
"""
数据对齐验证脚本
检验特征文件(.f32)与音频文件(.pcm)的严格10ms帧率对齐
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from feature_spec import get_default_feature_spec


def validate_alignment(features_path: str, audio_path: str, frame_size: int = 160):
    """验证特征文件与音频文件的对齐"""
    print("🔍 验证数据对齐...")

    features_path = Path(features_path)
    audio_path = Path(audio_path)

    # 检查文件存在
    assert features_path.exists(), f"特征文件不存在: {features_path}"
    assert audio_path.exists(), f"音频文件不存在: {audio_path}"

    # 加载特征
    spec = get_default_feature_spec()
    features_data = np.fromfile(features_path, dtype=np.float32)
    feature_frames = len(features_data) // spec.total_dim
    features = features_data.reshape(-1, spec.total_dim)

    print(f"📊 特征文件: {features_path.name}")
    print(f"  特征维度: {spec.total_dim}")
    print(f"  特征帧数: {feature_frames:,}")
    print(f"  文件大小: {features_path.stat().st_size / (1024**2):.1f} MB")

    # 加载音频
    audio_data = np.fromfile(audio_path, dtype=np.int16).astype(np.float32) / 32768.0
    audio_samples = len(audio_data)
    audio_frames = audio_samples // frame_size

    print(f"🎵 音频文件: {audio_path.name}")
    print(f"  采样点数: {audio_samples:,}")
    print(f"  音频帧数: {audio_frames:,} (基于{frame_size}样本/帧)")
    print(f"  时长: {audio_samples/16000:.1f}秒")
    print(f"  文件大小: {audio_path.stat().st_size / (1024**2):.1f} MB")

    # 对齐检查
    print(f"\n⚖️ 对齐检查:")
    frame_diff = abs(feature_frames - audio_frames)
    print(f"  帧数差异: {frame_diff}")

    if frame_diff == 0:
        print("  ✅ 完美对齐！")
        alignment_status = "perfect"
    elif frame_diff <= 1:
        print("  ✅ 基本对齐 (差异在1帧内，可接受)")
        alignment_status = "good"
    elif frame_diff <= 5:
        print("  ⚠️ 轻微不对齐 (差异在5帧内，可接受)")
        alignment_status = "acceptable"
    elif frame_diff <= 100:
        print("  ⚠️ 中等不对齐，但可能可用于训练")
        alignment_status = "marginal"
    else:
        print("  ❌ 严重不对齐！")
        alignment_status = "bad"

    # 时间对齐检查
    feature_duration = feature_frames * 10  # ms
    audio_duration = audio_frames * 10      # ms
    time_diff = abs(feature_duration - audio_duration)

    print(f"  特征时长: {feature_duration:.1f}ms ({feature_frames}帧)")
    print(f"  音频时长: {audio_duration:.1f}ms ({audio_frames}帧)")
    print(f"  时长差异: {time_diff:.1f}ms")

    # 特征质量检查
    print(f"\n🧪 特征质量检查:")

    # 使用FeatureSpec检查各部分特征
    # 转换为tensor以确保正确提取
    features_tensor = torch.from_numpy(features[:1000])
    f0 = spec.extract_feature(features_tensor, 'f0').numpy()
    voicing = spec.extract_feature(features_tensor, 'voicing').numpy()
    ceps = spec.extract_feature(features_tensor, 'ceps').numpy()

    print(f"  F0范围: [{f0.min():.3f}, {f0.max():.3f}], 均值: {f0.mean():.3f}")

    # 正确显示voicing值
    unique_voicing, counts = np.unique(voicing.flatten(), return_counts=True)
    print(f"  Voicing分布: 共{len(unique_voicing)}种值")
    for val, count in zip(unique_voicing, counts):
        print(f"    {val}: {count}次 ({count/len(voicing.flatten())*100:.1f}%)")

    print(f"  CEPS范围: [{ceps.min():.3f}, {ceps.max():.3f}], 标准差: {ceps.std():.3f}")

    # 静音检测
    audio_rms = np.sqrt(np.mean(audio_data**2))
    print(f"  音频RMS: {audio_rms:.6f} ({20*np.log10(audio_rms+1e-12):.1f} dB)")

    if audio_rms < 1e-5:
        print("  ⚠️ 音频能量过低，可能存在静音段")

    return {
        'alignment_status': alignment_status,
        'frame_diff': frame_diff,
        'time_diff_ms': time_diff,
        'feature_frames': feature_frames,
        'audio_frames': audio_frames,
        'audio_rms': audio_rms
    }


def main():
    parser = argparse.ArgumentParser(description='验证特征与音频的对齐')
    parser.add_argument('--features', type=str, required=True, help='特征文件路径(.f32)')
    parser.add_argument('--audio', type=str, required=True, help='音频文件路径(.pcm)')
    parser.add_argument('--frame-size', type=int, default=160, help='帧大小(样本数)')

    args = parser.parse_args()

    print("🧪 数据对齐验证")
    print("=" * 50)

    try:
        result = validate_alignment(args.features, args.audio, args.frame_size)

        print("=" * 50)
        if result['alignment_status'] in ['perfect', 'good']:
            print("🎉 验证通过！数据对齐良好")
        elif result['alignment_status'] == 'acceptable':
            print("⚠️ 验证通过，但存在轻微对齐问题")
        else:
            print("❌ 验证失败，存在严重对齐问题")

        return result['alignment_status'] in ['perfect', 'good', 'acceptable']

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()