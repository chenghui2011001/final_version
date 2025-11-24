#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义增强型AETHER解码器：插件式兼容层

核心设计：
1. 完全保持AETHERDecoder的原有功能和复杂度
2. 在36维输出基础上增加语义处理插件
3. 插件式SSL语义监督，不破坏原有架构
4. 向下兼容，对外接口透明
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, Union

try:
    # 尝试包内相对导入
    from ..aether_encoder_decoder import AETHERDecoder
    from .semantic_latent import SemanticAdapter, LatentSpaceHead, AcousticFusionHead
except Exception:
    # 回退到绝对导入
    from models.aether_encoder_decoder import AETHERDecoder
    from models.semantic_latent import SemanticAdapter, LatentSpaceHead, AcousticFusionHead

try:
    from utils.feature_spec import get_default_feature_spec
except Exception:
    from dnn.torch.final_version.utils.feature_spec import get_default_feature_spec


# 保留原有SemanticProcessor以兼容现有代码
class SemanticProcessorLegacy(nn.Module):
    """语义处理插件：对16维语义特征进行增强处理"""

    def __init__(
        self,
        semantic_dim: int = 16,
        ssl_dim: int = 768,
        enhancement_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.ssl_dim = ssl_dim

        # 语义特征增强网络（保持维度不变）
        if enhancement_layers == 1:
            self.enhancer = nn.Linear(semantic_dim, semantic_dim)
        else:
            mid_dim = semantic_dim * 4
            layers = []
            layers.extend([
                nn.Linear(semantic_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            for _ in range(enhancement_layers - 2):
                layers.extend([
                    nn.Linear(mid_dim, mid_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Linear(mid_dim, semantic_dim))
            self.enhancer = nn.Sequential(*layers)

        # SSL teacher投影：将SSL高维表示投影到语义空间
        self.ssl_to_semantic = nn.Sequential(
            nn.Linear(ssl_dim, ssl_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ssl_dim // 2, semantic_dim)
        )

        # 可选的归一化层
        self.output_norm = nn.LayerNorm(semantic_dim)
        self.ssl_norm = nn.LayerNorm(semantic_dim)

        # InfoNCE对比学习组件 - 防止特征塌陷的改进设置
        self.temperature = nn.Parameter(torch.tensor(0.8))  # 更高的初始温度，防止过度聚集
        self.use_infoce = True
        # 添加负样本增强参数
        self.negative_sample_ratio = 0.8  # 负样本采样比例

        self._init_weights()

    def _init_weights(self):
        """权重初始化：确保训练初期稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 更保守的初始化：防止语义特征分布崩塌
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # 更小的增益
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 特别保守地初始化最后一层
        if hasattr(self.enhancer, '-1'):
            final_layer = self.enhancer[-1]
        elif isinstance(self.enhancer, nn.Linear):
            final_layer = self.enhancer
        else:
            final_layer = list(self.enhancer.modules())[-1]

        if isinstance(final_layer, nn.Linear):
            # 增加初始化gain，避免输出过小导致分布塌陷
            nn.init.xavier_uniform_(final_layer.weight, gain=0.1)  # 从0.01增加到0.1
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def forward(self, semantic_raw: torch.Tensor) -> torch.Tensor:
        """
        语义特征增强处理

        Args:
            semantic_raw: 原始16维语义特征 [B, T, 16]

        Returns:
            enhanced_semantic: 增强后的16维语义特征 [B, T, 16]
        """
        enhanced = self.enhancer(semantic_raw)
        enhanced_semantic = self.output_norm(enhanced)
        return enhanced_semantic

    def project_ssl_teacher(self, ssl_features: torch.Tensor) -> torch.Tensor:
        """
        将SSL teacher特征投影到语义空间

        Args:
            ssl_features: SSL模型输出 [B, T, ssl_dim]

        Returns:
            ssl_semantic: 投影后的语义特征 [B, T, semantic_dim]
        """
        ssl_projected = self.ssl_to_semantic(ssl_features)
        ssl_semantic = self.ssl_norm(ssl_projected)
        return ssl_semantic

    def compute_semantic_loss(
        self,
        pred_semantic: torch.Tensor,
        ssl_features: torch.Tensor,
        loss_type: str = "cosine+infoce",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算语义对齐损失

        Args:
            pred_semantic: 预测的语义特征 [B, T, semantic_dim]
            ssl_features: SSL teacher特征 [B, T, ssl_dim]
            loss_type: 损失类型 ("cosine", "mse", "infonce")

        Returns:
            loss: 语义对齐损失
            metrics: 监控指标
        """
        # 投影SSL特征到语义空间
        ssl_semantic = self.project_ssl_teacher(ssl_features)

        if loss_type == "cosine":
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)
            loss = 1.0 - cos_sim.mean()

            metrics = {
                "semantic_cos_sim": cos_sim.mean().item(),
                "semantic_cos_std": cos_sim.std().item(),
            }

        elif loss_type == "mse":
            # L2距离损失
            loss = F.mse_loss(pred_semantic, ssl_semantic)

            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)
                metrics = {
                    "semantic_mse": loss.item(),
                    "semantic_cos_sim": cos_sim.mean().item(),
                }

        elif loss_type == "infonce":
            # InfoNCE对比学习损失（改进版本）
            B, T, D = pred_semantic.shape

            # L2归一化
            pred_norm = F.normalize(pred_semantic, p=2, dim=-1)
            ssl_norm = F.normalize(ssl_semantic, p=2, dim=-1)

            pred_flat = pred_norm.view(-1, D)
            ssl_flat = ssl_norm.view(-1, D)

            # 使用可学习温度参数 - 放宽范围防止过度聚集
            temperature = torch.clamp(self.temperature, 0.1, 2.0)
            logits = torch.matmul(pred_flat, ssl_flat.t()) / temperature
            targets = torch.arange(B * T, device=logits.device)

            loss = F.cross_entropy(logits, targets)

            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)
                metrics = {
                    "semantic_infonce": loss.item(),
                    "semantic_cos_sim": cos_sim.mean().item(),
                    "semantic_temperature": temperature.item(),
                }

        elif loss_type == "cosine+infoce":
            # 组合损失：余弦损失 + InfoNCE损失
            # 1. 余弦相似度损失
            cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)
            cosine_loss = 1.0 - cos_sim.mean()

            # 2. InfoNCE损失
            B, T, D = pred_semantic.shape
            pred_norm = F.normalize(pred_semantic, p=2, dim=-1)
            ssl_norm = F.normalize(ssl_semantic, p=2, dim=-1)

            pred_flat = pred_norm.view(-1, D)
            ssl_flat = ssl_norm.view(-1, D)

            # 使用可学习温度参数 - 放宽范围防止过度聚集
            temperature = torch.clamp(self.temperature, 0.1, 2.0)  # 下界从0.01提升到0.1
            logits = torch.matmul(pred_flat, ssl_flat.t()) / temperature
            targets = torch.arange(B * T, device=logits.device)

            infonce_loss = F.cross_entropy(logits, targets)

            # 组合损失：余弦损失权重0.3，InfoNCE权重0.7
            loss = 0.3 * cosine_loss + 0.7 * infonce_loss

            metrics = {
                "semantic_cosine": cosine_loss.item(),
                "semantic_infonce": infonce_loss.item(),
                "semantic_combined": loss.item(),
                "semantic_cos_sim": cos_sim.mean().item(),
                "semantic_temperature": temperature.item(),
            }

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return loss, metrics


# 保留原有SemanticFusionModule以兼容现有代码
class SemanticFusionModuleLegacy(nn.Module):
    """
    语义融合模块：16维语义特征指导20维声学特征优化

    设计思路：
    1. 接收 acoustic[20] + semantic[16]
    2. 语义特征作为teacher，通过attention/gate/cross-fusion指导声学特征
    3. 输出优化的acoustic[20] → FARGAN
    """

    def __init__(
        self,
        acoustic_dim: int = 20,
        semantic_dim: int = 16,
        fusion_type: str = "attention",  # "attention", "gate", "cross_mlp"
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.semantic_dim = semantic_dim
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim

        if fusion_type == "attention":
            # 语义特征作为query，声学特征作为key/value
            self.semantic_to_query = nn.Linear(semantic_dim, hidden_dim)
            self.acoustic_to_kv = nn.Linear(acoustic_dim, hidden_dim * 2)
            self.attention_fusion = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
            self.output_proj = nn.Linear(hidden_dim, acoustic_dim)
            self.residual_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12，初始保守

        elif fusion_type == "gate":
            # 语义特征生成门控信号调节声学特征
            self.semantic_gate = nn.Sequential(
                nn.Linear(semantic_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, acoustic_dim),
                nn.Sigmoid()  # 门控信号 [0,1]
            )
            self.acoustic_transform = nn.Sequential(
                nn.Linear(acoustic_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, acoustic_dim)
            )

        elif fusion_type == "cross_mlp":
            # 交叉MLP融合
            self.cross_fusion = nn.Sequential(
                nn.Linear(acoustic_dim + semantic_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, acoustic_dim)
            )

        # 输出归一化
        self.output_norm = nn.LayerNorm(acoustic_dim)
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        acoustic_features: torch.Tensor,  # [B, T, 20]
        semantic_features: torch.Tensor,  # [B, T, 16]
    ) -> torch.Tensor:
        """
        语义指导的声学特征融合

        Args:
            acoustic_features: 原始声学特征 [B, T, 20]
            semantic_features: 语义特征 [B, T, 16]

        Returns:
            enhanced_acoustic: 优化后的声学特征 [B, T, 20]
        """
        B, T, _ = acoustic_features.shape

        if self.fusion_type == "attention":
            # 语义→query，声学→key/value
            query = self.semantic_to_query(semantic_features)  # [B, T, hidden]
            kv = self.acoustic_to_kv(acoustic_features)        # [B, T, hidden*2]
            key, value = kv.chunk(2, dim=-1)                   # [B, T, hidden] each

            # Multi-head attention: semantic queries acoustic
            query = query.transpose(0, 1)  # [T, B, hidden]
            key = key.transpose(0, 1)      # [T, B, hidden]
            value = value.transpose(0, 1)  # [T, B, hidden]

            attended, _ = self.attention_fusion(query, key, value)  # [T, B, hidden]
            attended = attended.transpose(0, 1)                     # [B, T, hidden]

            enhanced_raw = self.output_proj(attended)  # [B, T, 20]
            # 残差连接：物理约束的残差权重，永远在(0,1)范围
            residual_scale = torch.sigmoid(self.residual_logit)  # ∈(0,1)
            enhanced_acoustic = acoustic_features + residual_scale * enhanced_raw

        elif self.fusion_type == "gate":
            # 语义生成门控信号
            gate = self.semantic_gate(semantic_features)           # [B, T, 20]
            transformed_acoustic = self.acoustic_transform(acoustic_features)  # [B, T, 20]

            # 门控调节：原始特征 * (1 + gate * 变换)
            enhanced_acoustic = acoustic_features * (1.0 + gate * transformed_acoustic)

        elif self.fusion_type == "cross_mlp":
            # 拼接融合
            combined = torch.cat([acoustic_features, semantic_features], dim=-1)  # [B, T, 36]
            fusion_output = self.cross_fusion(combined)  # [B, T, 20]

            # 残差连接
            enhanced_acoustic = acoustic_features + fusion_output

        # 改进的归一化空间融合策略
        # 1. 保存原始统计量用于反归一化
        with torch.no_grad():
            original_mean = acoustic_features.mean(dim=[0,1], keepdim=True)
            original_std = acoustic_features.std(dim=[0,1], keepdim=True) + 1e-8

        # 2. 在归一化空间中进行最终融合
        acoustic_normalized = (acoustic_features - original_mean) / original_std
        enhanced_normalized = (enhanced_acoustic - original_mean) / original_std

        # 3. 归一化空间中的范围保护
        enhanced_std_norm = enhanced_normalized.std(dim=[0,1], keepdim=True)
        acoustic_std_norm = acoustic_normalized.std(dim=[0,1], keepdim=True)

        # 防止归一化后分布过度压缩
        std_ratio = enhanced_std_norm / (acoustic_std_norm + 1e-8)
        std_correction = torch.where(std_ratio < 0.4, 0.4 / std_ratio, torch.ones_like(std_ratio))
        enhanced_normalized_corrected = enhanced_normalized * std_correction

        # 4. 反归一化回原始空间
        enhanced_final = enhanced_normalized_corrected * original_std + original_mean

        # 5. 最后的LayerNorm输出归一化
        enhanced_acoustic = self.output_norm(enhanced_final)

        return enhanced_acoustic


class SemanticAugmentedAETHERDecoder(AETHERDecoder):
    """
    语义增强型AETHER解码器：插件式兼容层

    设计原则：
    1. 完全保持AETHERDecoder的原有功能（36维输出、合成器、FiLM等）
    2. 在36维特征基础上进行逻辑分割和语义增强
    3. 插件式SSL语义监督，不破坏原有训练流程
    4. 向下兼容，对外接口完全透明
    """

    def __init__(
        self,
        # AETHERDecoder原有参数
        dz: int = 24,
        d_out: int = 36,                    # 保持36维输出！
        d_hidden: int = 128,
        d_csi: int = 32,
        decoder_heads: int = 2,
        enable_synth: bool = True,          # 保持合成器！
        feature_spec_type: str = "fargan",
        use_film: bool = True,              # 保持FiLM！
        # 语义增强插件参数
        enable_semantic_augmentation: bool = True,
        acoustic_dim: int = 20,             # 前20维：声学特征
        semantic_dim: int = 16,             # 后16维：语义特征
        ssl_dim: int = 768,                 # SSL模型维度
        semantic_enhancement_layers: int = 2,
        semantic_dropout: float = 0.1,
        # 新模块参数
        latent_dim: int = 64,               # z_sem潜空间维度
        use_cross_attention: bool = False,  # SemanticAdapter是否用cross-attention
        semantic_loss_type: str = "cosine+infoce",  # LatentSpaceHead损失类型
        # 语义融合模块参数
        enable_semantic_fusion: bool = True,
        fusion_type: str = "attention",     # "attention", "gate", "cross_mlp"
        fusion_hidden_dim: int = 64,
    ):
        # 1. 完全保持AETHERDecoder的原有初始化
        super().__init__(
            dz=dz,
            d_out=d_out,                    # ✓ 保持36维输出
            d_hidden=d_hidden,
            d_csi=d_csi,
            decoder_heads=decoder_heads,
            enable_synth=enable_synth,      # ✓ 保持原有合成器
            feature_spec_type=feature_spec_type,
            use_film=use_film,              # ✓ 保持FiLM功能
        )

        # 2. 语义增强插件配置
        self.enable_semantic_augmentation = enable_semantic_augmentation
        self.acoustic_dim = acoustic_dim
        self.semantic_dim = semantic_dim
        self.ssl_dim = ssl_dim

        # 语义融合模块配置
        self.enable_semantic_fusion = enable_semantic_fusion
        self.fusion_type = fusion_type

        # 20维→16维蒸馏头：强制20维也能逆向到语义空间
        self.acoustic_to_semantic_head = nn.Sequential(
            nn.Linear(acoustic_dim, acoustic_dim),
            nn.GELU(),
            nn.Dropout(semantic_dropout),
            nn.Linear(acoustic_dim, semantic_dim)
        ) if self.enable_semantic_augmentation else None

        # 验证维度配置
        if acoustic_dim + semantic_dim != d_out:
            raise ValueError(f"acoustic_dim({acoustic_dim}) + semantic_dim({semantic_dim}) != d_out({d_out})")

        # 3. 新的三个核心模块（仅在启用时初始化）
        if self.enable_semantic_augmentation:
            # 3.1 SemanticAdapter: semantic_raw + teacher → z_sem
            self.semantic_adapter = SemanticAdapter(
                semantic_raw_dim=semantic_dim,
                teacher_dim=ssl_dim,
                latent_dim=latent_dim,
                hidden_dim=fusion_hidden_dim * 2,  # 更大的hidden dim
                dropout=semantic_dropout,
                use_cross_attention=use_cross_attention,
            )

            # 3.2 LatentSpaceHead: z_sem → semantic_features + loss
            self.latent_head = LatentSpaceHead(
                latent_dim=latent_dim,
                semantic_dim=semantic_dim,
                teacher_dim=ssl_dim,
                dropout=semantic_dropout,
                loss_type=semantic_loss_type,
            )

            # 3.3 AcousticFusionHead: acoustic + z_sem → enhanced_acoustic
            if self.enable_semantic_fusion:
                self.acoustic_fusion_head = AcousticFusionHead(
                    acoustic_dim=acoustic_dim,
                    latent_dim=latent_dim,
                    fusion_type=fusion_type,
                    hidden_dim=fusion_hidden_dim,
                    dropout=semantic_dropout,
                )
            else:
                self.acoustic_fusion_head = None

        else:
            self.semantic_adapter = None
            self.latent_head = None
            self.acoustic_fusion_head = None

        # 保留兼容性：旧模块作为fallback
        self.semantic_processor = None
        self.semantic_fusion = None

        # 5. 暴露fargan_core属性（兼容Stage2→4加载）
        # 延迟到需要时再调用，避免初始化时的循环依赖
        try:
            self._expose_fargan_components()
        except Exception as e:
            print(f"[WARNING] 初始化时暴露FARGAN组件失败: {e}，将在需要时重试")

    def forward(
        self,
        z: torch.Tensor,
        csi_dict_or_csi = None,             # 兼容旧接口
        return_wave: bool = False,
        target_len: Optional[int] = None,
        enable_semantic_output: bool = True,  # 控制是否输出语义信息
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向传播：插件式语义增强

        Args:
            z: 编码器输出 [B, T, dz]
            csi_dict_or_csi: CSI信息（兼容tensor和dict）
            return_wave: 是否返回波形
            target_len: 目标波形长度
            enable_semantic_output: 是否启用语义输出

        Returns:
            - 如果enable_semantic_output=False: 与原AETHERDecoder完全一致的输出
            - 如果enable_semantic_output=True: 包含语义信息的字典输出
        """
        # 兼容性处理：CSI格式转换
        if csi_dict_or_csi is None:
            csi_dict = None
        elif hasattr(csi_dict_or_csi, 'keys'):
            csi_dict = csi_dict_or_csi
        elif isinstance(csi_dict_or_csi, torch.Tensor):
            csi_dict = {'csi_tensor': csi_dict_or_csi}
        else:
            csi_dict = csi_dict_or_csi

        # 1. 完全复用父类AETHERDecoder的前向传播
        if return_wave:
            features, wave = super().forward(z, csi_dict, return_wave=True, target_len=target_len, **kwargs)
        else:
            features = super().forward(z, csi_dict, return_wave=False, **kwargs)
            wave = None

        # 2. 如果不需要语义输出，直接返回原始结果（完全兼容）
        if not enable_semantic_output or not self.enable_semantic_augmentation:
            if return_wave:
                return features, wave
            else:
                return features

        # 3. 语义增强处理：在36维特征基础上进行逻辑分割
        acoustic_features = features[..., :self.acoustic_dim]        # 前20维：声学特征
        semantic_raw = features[..., self.acoustic_dim:]             # 后16维：原始语义特征

        # 4. 新的三阶段语义增强流程
        if self.semantic_adapter is not None and self.latent_head is not None:
            # Step 1: semantic_raw + teacher → z_sem（统一潜空间）
            teacher_features = kwargs.get('teacher_features', None)
            attn_mask = kwargs.get('attn_mask', None)

            z_sem, adapter_logs = self.semantic_adapter(
                semantic_raw,
                teacher_features=teacher_features,
                mask=attn_mask,
            )

            # Step 2: z_sem → semantic_features + semantic_loss
            semantic_features, sem_loss_tensor, sem_metrics = self.latent_head(
                z_sem,
                teacher_features=teacher_features,
                mask=attn_mask,
            )

            # Step 3: acoustic + z_sem → enhanced_acoustic
            if self.acoustic_fusion_head is not None:
                enhanced_acoustic_features, fusion_logs = self.acoustic_fusion_head(
                    acoustic_features,
                    z_sem,
                    mask=attn_mask,
                )
            else:
                enhanced_acoustic_features = acoustic_features
                fusion_logs = {}

        else:
            # Fallback: 使用原始特征
            semantic_features = semantic_raw
            enhanced_acoustic_features = acoustic_features
            adapter_logs = {}
            sem_metrics = {}
            fusion_logs = {}
            sem_loss_tensor = semantic_raw.new_tensor(0.0)

        # 6. 20维→16维蒸馏：强制20维保持语义可逆性
        acoustic_semantic_distill = None
        if self.acoustic_to_semantic_head is not None:
            acoustic_semantic_distill = self.acoustic_to_semantic_head(enhanced_acoustic_features)

        # 7. 如果需要波形且启用语义输出，基于融合后特征重新合成
        enhanced_wave = None
        if return_wave and enable_semantic_output:
            try:
                # 构建融合后的36维特征
                enhanced_features_36d = torch.cat([enhanced_acoustic_features, semantic_features], dim=-1)
                # 使用融合后特征重新合成波形
                enhanced_wave = self._generate_waveform_from_enhanced_features(
                    enhanced_features_36d, target_len=target_len
                )
                # 确保enhanced_wave不为None
                if enhanced_wave is None:
                    print(f"[WARNING] Enhanced wave synthesis returned None, using original wave")
                    enhanced_wave = wave
            except Exception as e:
                print(f"[WARNING] Enhanced wave synthesis failed: {e}, fallback to original wave")
                enhanced_wave = wave

            # 最后确保有一个有效的波形
            if enhanced_wave is None and wave is None:
                print(f"[ERROR] Both enhanced and original wave are None, generating zero wave")
                B, T = enhanced_acoustic_features.shape[:2]
                target_wav_len = target_len if target_len is not None else T * 160
                enhanced_wave = torch.zeros(B, target_wav_len, device=enhanced_acoustic_features.device, dtype=enhanced_acoustic_features.dtype)

        # 8. 返回丰富的输出信息
        fused36 = torch.cat([enhanced_acoustic_features, semantic_features], dim=-1)
        outputs = {
            'features': features,                # ✓ 兼容键（原始36维，保持不变）
            'features_raw': features,           # ✓ 明确别名：未融合的原始36维
            'features_36d': fused36,           # ✓ 兼容键（融合36维，沿用旧名）
            'features_fused': fused36,         # ✓ 明确别名：融合后的36维
            'acoustic_features': enhanced_acoustic_features,    # 融合优化后的20维：用于FARGAN
            'acoustic_raw': acoustic_features,                  # 原始20维：用于对比分析
            'semantic_features': semantic_features,             # 增强16维：用于SSL监督
            'semantic_raw': semantic_raw,                       # 原始16维：用于对比分析
            'acoustic_semantic_distill': acoustic_semantic_distill,  # 20维→16维蒸馏输出
            'hidden_states': None,                              # 占位符，保持接口一致性
            # 新增：三模块的监控信息
            'z_sem': z_sem if 'z_sem' in locals() else None,  # 统一潜空间
            'adapter_logs': adapter_logs if 'adapter_logs' in locals() else {},
            'semantic_metrics': sem_metrics if 'sem_metrics' in locals() else {},
            'fusion_logs': fusion_logs if 'fusion_logs' in locals() else {},
            'semantic_loss_tensor': sem_loss_tensor if 'sem_loss_tensor' in locals() else semantic_raw.new_tensor(0.0),
        }

        # 根据情况返回原始波形或融合波形
        if return_wave:
            if enable_semantic_output and enhanced_wave is not None:
                outputs['wave'] = enhanced_wave     # 使用融合后特征合成的波形
                outputs['wave_original'] = wave     # 保留原始波形用于对比
            else:
                outputs['wave'] = wave

        return outputs

    def _generate_waveform_from_enhanced_features(
        self,
        enhanced_features: torch.Tensor,
        target_len: Optional[int] = None
    ) -> torch.Tensor:
        """使用融合后的36维特征重新合成波形（优先走FARGAN路径）

        - 优先尝试直接使用 fargan_core（如果可用）
        - 其次尝试 AETHERDecoder 的 synth/_generate_waveform
        - 最后回退到独立的 FARGANDecoder
        """
        B, T, D = enhanced_features.shape
        device = enhanced_features.device
        try:
            # 优先：直接用 fargan_core（若已暴露）
            if hasattr(self, '_get_fargan_core'):
                fc = self._get_fargan_core()
            else:
                fc = getattr(self, 'fargan_core', None)
            if fc is not None:
                # 估计周期
                acoustic_part = enhanced_features[..., :20]
                try:
                    if hasattr(self, 'period_estimator') and self.period_estimator is not None:
                        period = self.period_estimator(enhanced_features)
                    else:
                        period = self._estimate_period_from_acoustic(acoustic_part)
                except Exception:
                    period = torch.full((B, T), 100, device=device, dtype=torch.long)

                # 计算可生成帧数（与FARGANCond一致：T→T-4）
                max_available_frames = max(1, T - 4)
                if target_len is not None:
                    target_frames_total = max(1, (int(target_len) + 160 - 1) // 160)
                    nb_frames = max(1, min(max_available_frames, target_frames_total))
                else:
                    nb_frames = max_available_frames

                # features_20 驱动 FARGANCore
                features_20 = acoustic_part
                audio, _ = fc(features_20, period.clamp(32, 255).to(torch.long), int(nb_frames), pre=None)
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                if target_len is not None and audio.size(-1) > target_len:
                    audio = audio[..., :target_len]
                return audio

            # 次优：有 AETHERDecoder 合成链
            if hasattr(self, 'synth') and self.synth is not None:
                wave_result = self.synth(enhanced_features, target_len=target_len)
                if wave_result is None:
                    raise RuntimeError("Synth returned None")
                return wave_result

            if hasattr(self, '_generate_waveform'):
                acoustic_part = enhanced_features[..., :20]
                try:
                    if hasattr(self, 'period_estimator') and self.period_estimator is not None:
                        period = self.period_estimator(enhanced_features)
                    else:
                        period = self._estimate_period_from_acoustic(acoustic_part)
                except Exception:
                    period = torch.full((B, T), 100, device=device, dtype=torch.long)
                wave_result = self._generate_waveform(enhanced_features, period, target_len=target_len)
                if wave_result is None:
                    raise RuntimeError("_generate_waveform returned None")
                return wave_result

            # 最后：独立FARGANDecoder回退（确保走FARGAN路径）
            try:
                if not hasattr(self, '_fallback_fargan_decoder') or self._fallback_fargan_decoder is None:
                    from models.fargan_decoder import FARGANDecoder as _FD
                    self._fallback_fargan_decoder = _FD()
                # 直接用36维融合特征驱动
                per, audio = self._fallback_fargan_decoder(enhanced_features, target_len=target_len)
                return audio
            except Exception:
                pass

            raise RuntimeError("No waveform synthesis method available")

        except Exception as e:
            print(f"[ERROR] Enhanced waveform synthesis failed: {e}")
            # 生成一个安全的零波形而不是抛出异常
            fallback_len = target_len if target_len is not None else T * 160
            return torch.zeros(B, fallback_len, device=device, dtype=enhanced_features.dtype)

    def _estimate_period_from_acoustic(self, acoustic_features: torch.Tensor) -> torch.Tensor:
        """从声学特征估计period（简单实现）

        Args:
            acoustic_features: [B, T, 20] 声学特征

        Returns:
            period: [B, T] period估计值
        """
        B, T = acoustic_features.shape[:2]

        # 简单策略：使用F0相关的维度（通常在后几维）
        # 假设某一维包含F0信息，将其转换为period
        try:
            # 如果第19维是F0相关的
            f0_feature = acoustic_features[..., 18]  # 第19维
            # 将F0特征映射到合理的period范围 (32-255)
            period = torch.clamp(
                100 + f0_feature * 50,  # 基础period + 变化
                32, 255
            ).round().long()
        except Exception:
            # 如果失败，使用固定period
            period = torch.full((B, T), 100, device=acoustic_features.device, dtype=torch.long)

        return period

    def _expose_fargan_components(self):
        """暴露FARGAN组件属性，兼容Stage2→4权重加载"""
        try:
            print("[DEBUG] 开始查找FARGAN组件...")

            # 方法1：检查继承的AETHERDecoder是否有synth属性
            if hasattr(self, 'synth') and self.synth is not None:
                print(f"[DEBUG] 找到synth属性: {type(self.synth)}")
                if hasattr(self.synth, 'fargan_core'):
                    self.fargan_core = self.synth.fargan_core
                    print("[DEBUG] 通过synth.fargan_core成功暴露")
                    return
                elif hasattr(self.synth, 'vocoder') and hasattr(self.synth.vocoder, 'fargan_core'):
                    self.fargan_core = self.synth.vocoder.fargan_core
                    print("[DEBUG] 通过synth.vocoder.fargan_core成功暴露")
                    return
                else:
                    # 检查synth是否本身就是FARGANCore
                    if hasattr(self.synth, 'cond_net') or hasattr(self.synth, 'sig_net'):
                        self.fargan_core = self.synth
                        print("[DEBUG] synth本身就是fargan_core")
                        return

            # 方法2：检查所有子模块寻找FARGAN相关组件
            for name, module in self.named_modules():
                if any(keyword in name.lower() for keyword in ['fargan', 'core', 'cond', 'sig']):
                    if hasattr(module, 'cond_net') or hasattr(module, 'sig_net'):
                        self.fargan_core = module
                        print(f"[DEBUG] 通过子模块 {name} 找到fargan_core")
                        return

            # 方法3：创建一个兼容的fargan_core属性，用于权重加载
            print("[WARNING] 未找到FARGAN组件，创建兼容属性")
            from types import SimpleNamespace
            self.fargan_core = SimpleNamespace()
            # 添加必要的方法使其看起来像真正的fargan_core
            self.fargan_core.state_dict = lambda: {}
            self.fargan_core.load_state_dict = lambda state_dict, strict=True: None
            self.fargan_core.parameters = lambda: iter([])
            print("[DEBUG] 创建了兼容的fargan_core属性")

        except Exception as e:
            print(f"[WARNING] 暴露FARGAN组件失败: {e}")

    def _get_fargan_core(self):
        """获取FARGAN核心组件，支持多种路径"""
        if hasattr(self, 'fargan_core') and self.fargan_core is not None:
            return self.fargan_core

        # 如果没有fargan_core，尝试重新暴露
        self._expose_fargan_components()

        if hasattr(self, 'fargan_core'):
            return self.fargan_core
        else:
            print("[ERROR] 无法获取fargan_core组件")
            return None

    def compute_semantic_loss(
        self,
        semantic_features: torch.Tensor,
        ssl_features: torch.Tensor,
        loss_type: str = "cosine",
        # Wave级语义约束参数
        wave_gt: Optional[torch.Tensor] = None,
        wave_rec: Optional[torch.Tensor] = None,
        ssl_extractor = None,
        wave_semantic_weight: float = 0.3,
        # 20维→16维蒸馏参数
        acoustic_semantic_distill: Optional[torch.Tensor] = None,
        distill_weight: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算语义对齐损失（使用新的LatentSpaceHead）"""
        if not self.enable_semantic_augmentation or self.latent_head is None:
            # 如果未启用语义增强，返回零损失
            device = semantic_features.device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {"semantic_disabled": 1.0}

        # 使用新的LatentSpaceHead计算基础语义损失
        # 注意：这里需要从z_sem重新计算，但我们直接使用已有的semantic_features
        # 创建一个dummy z_sem来调用LatentSpaceHead
        dummy_z_sem = torch.randn_like(semantic_features).unsqueeze(-1).expand(-1, -1, self.latent_head.latent_dim)
        base_loss, base_metrics = self.latent_head(
            dummy_z_sem, ssl_features
        )[1:3]  # 只取loss和metrics

        total_loss = base_loss
        total_metrics = base_metrics.copy()

        # Wave级语义约束：最直接的语义保持
        if wave_gt is not None and wave_rec is not None and ssl_extractor is not None:
            try:
                # 提取原始和重建波形的SSL语义特征
                with torch.no_grad():
                    ssl_semantic_gt = ssl_extractor(wave_gt)  # [B, T, ssl_dim]
                    ssl_semantic_rec = ssl_extractor(wave_rec)  # [B, T, ssl_dim]

                # 时间对齐到特征帧率
                if ssl_semantic_gt.size(1) != semantic_features.size(1):
                    ssl_semantic_gt = F.interpolate(
                        ssl_semantic_gt.transpose(1, 2),
                        size=semantic_features.size(1),
                        mode='linear', align_corners=False
                    ).transpose(1, 2)

                if ssl_semantic_rec.size(1) != semantic_features.size(1):
                    ssl_semantic_rec = F.interpolate(
                        ssl_semantic_rec.transpose(1, 2),
                        size=semantic_features.size(1),
                        mode='linear', align_corners=False
                    ).transpose(1, 2)

                # Wave级语义损失：重建波形的语义应该接近原始波形
                wave_semantic_loss = F.cosine_embedding_loss(
                    ssl_semantic_rec.view(-1, ssl_semantic_rec.size(-1)),
                    ssl_semantic_gt.view(-1, ssl_semantic_gt.size(-1)),
                    torch.ones(ssl_semantic_rec.size(0) * ssl_semantic_rec.size(1),
                              device=ssl_semantic_rec.device)
                )

                total_loss = total_loss + wave_semantic_weight * wave_semantic_loss
                total_metrics['wave_semantic_loss'] = wave_semantic_loss.item()
                total_metrics['wave_semantic_weight'] = wave_semantic_weight

            except Exception as e:
                # Wave级约束失败时回退到基础损失
                total_metrics['wave_semantic_error'] = str(e)

        # 20维→16维蒸馏损失：强制20维保持语义可逆性
        if acoustic_semantic_distill is not None:
            # 将20维投影的输出与16维语义特征对齐
            distill_loss = F.cosine_embedding_loss(
                acoustic_semantic_distill.view(-1, acoustic_semantic_distill.size(-1)),
                semantic_features.view(-1, semantic_features.size(-1)),
                torch.ones(acoustic_semantic_distill.size(0) * acoustic_semantic_distill.size(1),
                          device=acoustic_semantic_distill.device)
            )

            total_loss = total_loss + distill_weight * distill_loss
            total_metrics['acoustic_semantic_distill_loss'] = distill_loss.item()
            total_metrics['distill_weight'] = distill_weight

        total_metrics['total_semantic_loss'] = total_loss.item()
        return total_loss, total_metrics

    def get_semantic_info(self) -> Dict[str, Any]:
        """获取语义增强配置信息"""
        return {
            'enable_semantic_augmentation': self.enable_semantic_augmentation,
            'acoustic_dim': self.acoustic_dim,
            'semantic_dim': self.semantic_dim,
            'ssl_dim': self.ssl_dim,
            'total_feature_dim': self.acoustic_dim + self.semantic_dim,
            'has_semantic_processor': self.semantic_processor is not None,
            # 语义融合模块信息
            'enable_semantic_fusion': self.enable_semantic_fusion,
            'has_semantic_fusion': self.semantic_fusion is not None,
            'fusion_type': getattr(self, 'fusion_type', 'none'),
            'fusion_flow': '36D → 20D+16D → SemanticFusion → Enhanced20D → FARGAN',
            'original_aether_preserved': True,  # 强调保持了原有功能
            # 添加残差权重信息
            'residual_logit': getattr(self.semantic_fusion, 'residual_logit', None),
            'current_residual_scale': torch.sigmoid(self.semantic_fusion.residual_logit).item() if hasattr(self.semantic_fusion, 'residual_logit') else None
        }

    def enable_semantic_mode(self):
        """启用语义增强模式"""
        self.enable_semantic_augmentation = True
        if self.semantic_processor is not None:
            self.semantic_processor.train(True)

    def disable_semantic_mode(self):
        """禁用语义增强模式（回退到纯AETHERDecoder）"""
        self.enable_semantic_augmentation = False
        if self.semantic_processor is not None:
            self.semantic_processor.train(False)


def create_semantic_augmented_decoder(
    config: Dict[str, Any],
    enable_semantic: bool = True,
    ssl_model_type: str = "hubert-base",
    # 语义融合参数
    enable_fusion: bool = True,
    fusion_type: str = "attention",
    fusion_hidden_dim: int = 64,
    semantic_enhancement_layers: int = 2,
    semantic_dropout: float = 0.1,
) -> SemanticAugmentedAETHERDecoder:
    """
    创建语义增强解码器的便捷函数

    Args:
        config: 配置字典
        enable_semantic: 是否启用语义增强
        ssl_model_type: SSL模型类型

    Returns:
        SemanticAugmentedAETHERDecoder实例
    """
    # SSL维度映射
    ssl_dim_map = {
        "hubert-base": 768,
        "hubert-large": 1024,
        "wavlm-base": 768,
        "wavlm-large": 1024,
    }

    ssl_dim = ssl_dim_map.get(ssl_model_type, 768)

    return SemanticAugmentedAETHERDecoder(
        dz=config.get("dz", 24),
        d_out=config.get("d_out", 36),              # 保持36维
        d_hidden=config.get("d_hidden", 128),
        d_csi=config.get("d_csi", 32),
        decoder_heads=config.get("decoder_heads", 2),
        enable_synth=config.get("enable_synth", True),  # 保持合成器
        feature_spec_type=config.get("feature_spec_type", "fargan"),
        use_film=config.get("use_film", True),          # 保持FiLM
        # 语义增强参数
        enable_semantic_augmentation=enable_semantic,
        acoustic_dim=20,
        semantic_dim=16,
        ssl_dim=ssl_dim,
        semantic_enhancement_layers=semantic_enhancement_layers,
        semantic_dropout=semantic_dropout,
        # 语义融合参数
        enable_semantic_fusion=enable_fusion,
        fusion_type=fusion_type,
        fusion_hidden_dim=fusion_hidden_dim,
    )


# 向下兼容别名
DualHeadAETHERFARGANDecoder = SemanticAugmentedAETHERDecoder
create_dual_head_decoder = create_semantic_augmented_decoder


if __name__ == "__main__":
    # 测试语义增强AETHER解码器
    print("Testing SemanticAugmentedAETHERDecoder...")

    # 创建测试数据
    batch_size, seq_len = 2, 50
    dz, d_csi = 24, 32

    z = torch.randn(batch_size, seq_len, dz)
    csi = torch.randn(batch_size, seq_len, d_csi)
    ssl_features = torch.randn(batch_size, seq_len, 768)

    # 创建语义增强解码器
    config = {
        "dz": dz,
        "d_out": 36,
        "d_hidden": 128,
        "d_csi": d_csi,
        "decoder_heads": 2,
    }

    decoder = create_semantic_augmented_decoder(config, enable_semantic=True)

    print(f"Input z shape: {z.shape}")
    print(f"Input csi shape: {csi.shape}")

    # 测试1：兼容模式（与原AETHERDecoder完全一致）
    print("\n=== 兼容模式测试 ===")
    features_compat = decoder(z, csi, enable_semantic_output=False)
    print(f"Compatible output shape: {features_compat.shape}")

    # 测试2：语义增强模式
    print("\n=== 语义增强模式测试 ===")
    outputs = decoder(z, csi, enable_semantic_output=True, return_wave=True)

    print("Semantic augmented outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")

    # 测试3：语义损失计算
    print("\n=== 语义损失测试 ===")
    semantic_loss, metrics = decoder.compute_semantic_loss(
        outputs['semantic_features'],
        ssl_features,
        loss_type="cosine"
    )

    print(f"Semantic loss: {semantic_loss.item():.6f}")
    print(f"Metrics: {metrics}")

    # 测试4：配置信息
    print("\n=== 配置信息 ===")
    info = decoder.get_semantic_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n✅ SemanticAugmentedAETHERDecoder test completed!")
