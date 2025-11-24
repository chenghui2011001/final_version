#!/usr/bin/env python3
"""
Stage5 训练配置文件

包含所有超参数配置和预设配置方案
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union
import json

@dataclass
class Stage5TrainingConfig:
    """Stage5训练配置类"""

    # === 基础配置 ===
    device: str = 'cuda'
    mixed_precision: bool = True
    amp_dtype: str = 'fp16'  # fp16 or bf16

    # === 数据配置 ===
    data_root: Optional[str] = None
    batch_size: int = 4
    seq_len: int = 800
    num_workers: int = 4
    pin_memory: bool = True

    # === 训练配置 ===
    num_epochs: int = 10
    total_steps: int = 8000
    save_every_steps: int = 1000
    log_interval: int = 50
    validation_interval: int = 500

    # === 优化器配置 ===
    base_lr: float = 1e-4
    rvq_lr: float = 2e-4
    rate_controller_lr: float = 5e-5
    base_wd: float = 1e-5
    rvq_wd: float = 1e-5
    grad_clip_norm: float = 3.0

    # === 学习率调度 ===
    scheduler_type: str = 'cosine_restarts'  # cosine_restarts, step, linear
    scheduler_T0: int = 2000
    scheduler_Tmult: float = 1.5
    scheduler_eta_min: float = 1e-6
    scheduler_step_size: int = 2000
    scheduler_gamma: float = 0.5

    # === 模型架构 ===
    feature_dim: int = 24
    original_feature_dim: int = 36
    rvq_stages: int = 3
    codebook_sizes: List[int] = None
    commitment_weights: List[float] = None
    semantic_layers: int = 2
    quality_metrics: int = 3

    # === 码率控制 ===
    target_kbps: float = 1.2
    rate_tolerance: float = 0.1
    frame_rate: int = 50
    rate_warmup_steps: int = 500
    control_strength: float = 0.1

    # === PID控制参数 ===
    pid_kp: float = 0.8
    pid_ki: float = 0.1
    pid_kd: float = 0.05

    # === 损失权重（初始值）===
    initial_loss_weights: Dict[str, float] = None
    temporal_smoothness: float = 0.1
    max_jump_threshold: float = 2.0
    min_quality_threshold: float = 2.5
    enable_stability_loss: bool = True

    # === 冻结策略 ===
    freeze_encoder: bool = True
    freeze_decoder_except_wave_head: bool = True

    def __post_init__(self):
        """后处理：设置默认值"""
        if self.codebook_sizes is None:
            self.codebook_sizes = [1024, 512, 256][:self.rvq_stages]

        if self.commitment_weights is None:
            self.commitment_weights = [0.25, 0.35, 0.45][:self.rvq_stages]

        if self.initial_loss_weights is None:
            self.initial_loss_weights = {
                'feat': 0.3,
                'wave': 0.4,
                'rate': 0.8,
                'semantic': 0.4,
                'temporal': 0.1,
                'quality': 0.1,
                'commitment': 0.5
            }

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Stage5TrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, file_path: str) -> 'Stage5TrainingConfig':
        """从文件加载配置"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# === 预设配置 ===

def get_development_config() -> Stage5TrainingConfig:
    """开发/调试配置：快速训练，较少epoch"""
    return Stage5TrainingConfig(
        batch_size=2,
        seq_len=400,
        num_epochs=3,
        total_steps=1500,
        save_every_steps=500,
        log_interval=20,
        base_lr=2e-4,
        rvq_lr=4e-4,
        rate_warmup_steps=200,
        scheduler_T0=800
    )

def get_standard_config() -> Stage5TrainingConfig:
    """标准训练配置：均衡的质量和速度"""
    return Stage5TrainingConfig(
        batch_size=4,
        seq_len=800,
        num_epochs=10,
        total_steps=8000,
        save_every_steps=1000,
        log_interval=50,
        base_lr=1e-4,
        rvq_lr=2e-4,
        rate_warmup_steps=500
    )

def get_production_config() -> Stage5TrainingConfig:
    """生产配置：高质量，较长训练时间"""
    return Stage5TrainingConfig(
        batch_size=6,
        seq_len=1000,
        num_epochs=15,
        total_steps=12000,
        save_every_steps=1000,
        log_interval=100,
        base_lr=8e-5,
        rvq_lr=1.5e-4,
        rate_controller_lr=4e-5,
        grad_clip_norm=2.0,
        rate_warmup_steps=800,
        scheduler_T0=3000,
        temporal_smoothness=0.15,
        min_quality_threshold=2.8
    )

def get_memory_efficient_config() -> Stage5TrainingConfig:
    """内存优化配置：适合GPU内存受限的情况"""
    return Stage5TrainingConfig(
        batch_size=2,
        seq_len=600,
        num_epochs=12,
        total_steps=10000,
        save_every_steps=1200,
        log_interval=30,
        base_lr=1.2e-4,
        rvq_lr=2.5e-4,
        mixed_precision=True,
        amp_dtype='fp16'
    )

def get_high_quality_config() -> Stage5TrainingConfig:
    """高质量配置：优先音质，适度码率约束"""
    return Stage5TrainingConfig(
        batch_size=4,
        seq_len=800,
        num_epochs=12,
        total_steps=10000,
        base_lr=8e-5,
        rvq_lr=1.6e-4,
        rate_tolerance=0.15,  # 稍微放宽码率容忍度
        min_quality_threshold=3.0,  # 更高的质量阈值
        initial_loss_weights={
            'feat': 0.2,
            'wave': 0.6,  # 增强波形质量权重
            'rate': 0.4,  # 适度码率约束
            'semantic': 0.3,
            'temporal': 0.2,
            'quality': 0.3,  # 增强质量监督
            'commitment': 0.3
        },
        temporal_smoothness=0.2,  # 增强时间平滑性
        control_strength=0.05  # 降低控制强度，避免过度约束
    )

def get_fast_convergence_config() -> Stage5TrainingConfig:
    """快速收敛配置：较高学习率，快速达到目标性能"""
    return Stage5TrainingConfig(
        batch_size=6,
        seq_len=800,
        num_epochs=8,
        total_steps=6000,
        base_lr=2e-4,
        rvq_lr=4e-4,
        rate_controller_lr=1e-4,
        rate_warmup_steps=300,  # 更快的预热
        scheduler_T0=1500,      # 更短的周期
        grad_clip_norm=4.0,     # 稍高的梯度裁剪
        initial_loss_weights={
            'feat': 0.4,
            'wave': 0.5,
            'rate': 1.0,  # 强化码率学习
            'semantic': 0.5,
            'temporal': 0.05,
            'quality': 0.1,
            'commitment': 0.6
        }
    )

# === 配置工厂函数 ===

CONFIG_PRESETS = {
    'development': get_development_config,
    'standard': get_standard_config,
    'production': get_production_config,
    'memory_efficient': get_memory_efficient_config,
    'high_quality': get_high_quality_config,
    'fast_convergence': get_fast_convergence_config
}

def create_config(preset: str = 'standard', **overrides) -> Stage5TrainingConfig:
    """
    创建配置对象

    Args:
        preset: 预设配置名称
        **overrides: 要覆盖的配置项
    """
    if preset not in CONFIG_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(CONFIG_PRESETS.keys())}")

    config = CONFIG_PRESETS[preset]()

    # 应用覆盖
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key: {key}")

    return config

def get_config_for_target_rate(target_kbps: float) -> Stage5TrainingConfig:
    """根据目标码率创建优化的配置"""
    if target_kbps <= 1.0:
        # 极低码率：需要更强的压缩
        return create_config(
            'production',
            target_kbps=target_kbps,
            rate_tolerance=0.05,
            rvq_stages=4,
            codebook_sizes=[512, 256, 128, 64],
            commitment_weights=[0.2, 0.3, 0.4, 0.5],
            initial_loss_weights={
                'feat': 0.2, 'wave': 0.3, 'rate': 1.2,
                'semantic': 0.3, 'temporal': 0.15, 'quality': 0.1, 'commitment': 0.4
            }
        )
    elif target_kbps <= 1.5:
        # 标准低码率
        return create_config(
            'standard',
            target_kbps=target_kbps,
            rate_tolerance=0.1
        )
    else:
        # 较高码率：优先质量
        return create_config(
            'high_quality',
            target_kbps=target_kbps,
            rate_tolerance=0.15
        )

# === 验证函数 ===

def validate_config(config: Stage5TrainingConfig) -> List[str]:
    """验证配置的合理性，返回警告列表"""
    warnings = []

    # 检查码率配置
    if config.target_kbps < 0.5:
        warnings.append(f"Very low target rate: {config.target_kbps} kbps may be too aggressive")

    if config.rate_tolerance < 0.05:
        warnings.append(f"Very tight rate tolerance: {config.rate_tolerance} may cause instability")

    # 检查学习率配置
    if config.base_lr > 5e-4:
        warnings.append(f"High base learning rate: {config.base_lr} may cause instability")

    if config.rvq_lr < config.base_lr:
        warnings.append("RVQ learning rate lower than base LR - consider increasing")

    # 检查批次大小
    if config.batch_size < 2:
        warnings.append("Very small batch size may affect training stability")

    # 检查RVQ配置
    if len(config.codebook_sizes) != config.rvq_stages:
        warnings.append("Codebook sizes length doesn't match RVQ stages")

    if len(config.commitment_weights) != config.rvq_stages:
        warnings.append("Commitment weights length doesn't match RVQ stages")

    # 检查损失权重
    if 'rate' not in config.initial_loss_weights:
        warnings.append("Missing rate loss weight in initial_loss_weights")

    total_weight = sum(config.initial_loss_weights.values())
    if total_weight > 5.0:
        warnings.append(f"Very high total loss weight: {total_weight}")

    return warnings

# === 使用示例 ===

if __name__ == "__main__":
    # 创建标准配置
    config = create_config('standard')
    print("Standard Configuration:")
    print(f"  Target rate: {config.target_kbps} ± {config.rate_tolerance} kbps")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rates: base={config.base_lr:.2e}, rvq={config.rvq_lr:.2e}")

    # 验证配置
    warnings = validate_config(config)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguration validated successfully")

    # 保存配置示例
    config.save_to_file("/tmp/stage5_standard_config.json")
    print(f"\nConfiguration saved to /tmp/stage5_standard_config.json")

    # 展示所有预设
    print(f"\nAvailable presets: {list(CONFIG_PRESETS.keys())}")

    # 针对不同码率的配置示例
    for rate in [0.8, 1.2, 1.6]:
        rate_config = get_config_for_target_rate(rate)
        print(f"\nOptimized config for {rate} kbps:")
        print(f"  RVQ stages: {rate_config.rvq_stages}")
        print(f"  Rate tolerance: {rate_config.rate_tolerance}")
        print(f"  Rate loss weight: {rate_config.initial_loss_weights.get('rate', 'N/A')}")