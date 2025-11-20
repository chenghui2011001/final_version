#!/usr/bin/env python3
"""
特征一致性检查脚本
验证extract_features_48_lpcnet_style.py与feature_spec.py的完全一致性
"""

import numpy as np
import torch
from pathlib import Path
from feature_spec import get_default_feature_spec
import argparse


def check_feature_layout_consistency():
    """检查特征布局定义的一致性"""
    print("🔍 检查特征布局一致性...")

    spec = get_default_feature_spec()

    # 验证总维度
    assert spec.total_dim == 48, f"总维度应为48，实际为{spec.total_dim}"

    # 验证各子特征维度
    expected_dims = {
        'ceps': 20,      # [0:19] -> 20维
        'f0': 1,         # [20] -> 1维
        'voicing': 1,    # [21] -> 1维
        'enhanced': 6,   # [22:27] -> 6维
        'lpc': 16,       # [28:43] -> 16维
        'prosodic': 4    # [44:47] -> 4维
    }

    for feature_name, expected_dim in expected_dims.items():
        actual_dim = spec.get_feature_dims(feature_name)
        assert actual_dim == expected_dim, f"{feature_name}维度不匹配: 期望{expected_dim}, 实际{actual_dim}"
        print(f"  ✅ {feature_name}: {actual_dim}维")

    # 验证范围连续性
    ranges = [
        spec.ceps_range,      # (0, 20)
        spec.f0_range,        # (20, 21)
        spec.voicing_range,   # (21, 22)
        spec.enhanced_range,  # (22, 28)
        spec.lpc_range,       # (28, 44)
        spec.prosodic_range   # (44, 48)
    ]

    expected_start = 0
    for i, (start, end) in enumerate(ranges):
        assert start == expected_start, f"范围{i}开始位置错误: 期望{expected_start}, 实际{start}"
        expected_start = end

    assert expected_start == 48, f"最终位置应为48，实际为{expected_start}"

    print("  ✅ 特征布局完全一致")
    return True


def check_feature_extraction_consistency(features_data: np.ndarray):
    """检查特征提取的一致性"""
    print("🔍 检查特征提取一致性...")

    spec = get_default_feature_spec()

    # 转换为torch张量进行测试
    features = torch.from_numpy(features_data).float()
    B, T, D = features.shape

    if D != 48:
        print(f"❌ 特征维度不匹配: 期望48, 实际{D}")
        return False

    # 提取各子特征
    ceps = spec.extract_feature(features, 'ceps')
    f0 = spec.extract_feature(features, 'f0')
    voicing = spec.extract_feature(features, 'voicing')
    enhanced = spec.extract_feature(features, 'enhanced')
    lpc = spec.extract_feature(features, 'lpc')
    prosodic = spec.extract_feature(features, 'prosodic')

    # 重新拼接
    reconstructed = torch.cat([ceps, f0, voicing, enhanced, lpc, prosodic], dim=-1)

    # 验证拼接结果与原始特征完全一致
    max_diff = torch.max(torch.abs(features - reconstructed)).item()

    if max_diff > 1e-6:
        print(f"❌ 特征拼接不一致，最大差异: {max_diff}")
        return False

    print(f"  ✅ 特征提取与拼接完全一致 (最大差异: {max_diff:.2e})")

    # 打印各特征统计信息
    print("  📊 各特征统计:")
    for name, feat in [('ceps', ceps), ('f0', f0), ('voicing', voicing),
                      ('enhanced', enhanced), ('lpc', lpc), ('prosodic', prosodic)]:
        mean = feat.mean().item()
        std = feat.std().item()
        min_val = feat.min().item()
        max_val = feat.max().item()
        print(f"    {name:>8}: mean={mean:7.3f}, std={std:6.3f}, range=[{min_val:7.3f}, {max_val:7.3f}]")

    return True


def check_normalization_consistency(features_data: np.ndarray):
    """检查归一化一致性"""
    print("🔍 检查归一化一致性...")

    spec = get_default_feature_spec()
    features = torch.from_numpy(features_data).float()

    # 应用LPCNet风格归一化
    normalized = spec.apply_lpcnet_normalization(features)

    # 检查各特征的归一化结果
    f0_norm = spec.extract_feature(normalized, 'f0')
    voicing_norm = spec.extract_feature(normalized, 'voicing')

    # F0应该在[0,1]范围内
    f0_min, f0_max = f0_norm.min().item(), f0_norm.max().item()
    print(f"  F0归一化范围: [{f0_min:.3f}, {f0_max:.3f}]")

    # Voicing应该是0/1值
    voicing_unique = torch.unique(voicing_norm).tolist()
    print(f"  Voicing唯一值: {voicing_unique}")

    # CEPS应该接近零均值单位方差
    ceps_norm = spec.extract_feature(normalized, 'ceps')
    ceps_mean = ceps_norm.mean().item()
    ceps_std = ceps_norm.std().item()
    print(f"  CEPS归一化: mean={ceps_mean:.3f}, std={ceps_std:.3f}")

    print("  ✅ 归一化检查完成")
    return True


def main():
    parser = argparse.ArgumentParser(description='特征一致性检查')
    parser.add_argument('--features-file', type=str, default=None,
                       help='48维特征文件路径(.f32)')
    parser.add_argument('--sample-frames', type=int, default=1000,
                       help='采样帧数进行检查')

    args = parser.parse_args()

    print("🧪 特征一致性检查开始")
    print("=" * 50)

    # 检查特征布局一致性
    try:
        check_feature_layout_consistency()
    except Exception as e:
        print(f"❌ 特征布局检查失败: {e}")
        return False

    # 如果提供了特征文件，进行实际数据检查
    if args.features_file:
        features_path = Path(args.features_file)
        if not features_path.exists():
            print(f"❌ 特征文件不存在: {features_path}")
            return False

        print(f"📁 加载特征文件: {features_path}")

        # 加载特征数据
        features_data = np.fromfile(features_path, dtype=np.float32)
        total_frames = len(features_data) // 48
        features_data = features_data.reshape(-1, 48)

        print(f"  总帧数: {total_frames:,}")

        # 采样进行检查
        sample_frames = min(args.sample_frames, total_frames)
        if sample_frames < total_frames:
            indices = np.random.choice(total_frames, sample_frames, replace=False)
            sample_data = features_data[indices]
        else:
            sample_data = features_data

        # 重新整形为[B, T, D]格式
        sample_data = sample_data.reshape(1, -1, 48)

        print(f"  采样检查: {sample_data.shape[1]}帧")

        # 检查特征提取一致性
        try:
            check_feature_extraction_consistency(sample_data)
        except Exception as e:
            print(f"❌ 特征提取检查失败: {e}")
            return False

        # 检查归一化一致性
        try:
            check_normalization_consistency(sample_data)
        except Exception as e:
            print(f"❌ 归一化检查失败: {e}")
            return False

    print("=" * 50)
    print("🎉 所有一致性检查通过！")
    return True


def quick_demo():
    """快速演示用法"""
    print("🎮 快速演示模式")

    # 生成随机特征数据
    features = torch.randn(2, 100, 48)  # [B=2, T=100, D=48]

    spec = get_default_feature_spec()

    # 打印特征布局
    spec.print_feature_layout()

    # 演示特征提取
    print("\n📊 特征提取演示:")
    for feature_name in ['ceps', 'f0', 'voicing', 'enhanced', 'lpc', 'prosodic']:
        feat = spec.extract_feature(features, feature_name)
        print(f"  {feature_name:>8}: {feat.shape}")

    # 演示归一化
    normalized = spec.apply_lpcnet_normalization(features)
    print(f"\n🔄 归一化结果: {normalized.shape}")

    return check_feature_extraction_consistency(features.numpy())


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 无参数时运行演示
        quick_demo()
    else:
        # 有参数时运行完整检查
        main()