#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义头模块：从共享latent中提取语义特征，与SSL teacher对齐

核心设计原则：
1. 共享latent表示：声学头(20维) + 语义头(16维) 都从同一个隐状态h提取
2. SSL强绑定：语义头输出直接与SSL teacher表示对齐，不是随意设计的latent
3. 多任务约束：通过梯度反传，语义监督塑造整个decoder的隐空间
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple


class SemanticHead(nn.Module):
    """
    语义头：从decoder隐状态中提取语义特征

    设计思路：
    - 输入：decoder的内部隐状态 h [B, T, hidden_dim]
    - 输出：语义特征 [B, T, semantic_dim] 用于与SSL teacher对齐
    - 约束：通过语义loss，反向传播塑造共享的隐空间表示
    """

    def __init__(
        self,
        hidden_dim: int = 256,           # decoder隐状态维度
        semantic_dim: int = 16,          # 语义特征维度
        ssl_dim: int = 768,              # SSL模型输出维度(HuBERT-base)
        projection_layers: int = 2,      # 投影层数
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.semantic_dim = semantic_dim
        self.ssl_dim = ssl_dim
        self.use_layer_norm = use_layer_norm

        # 隐状态 → 语义特征投影
        if projection_layers == 1:
            self.semantic_projection = nn.Linear(hidden_dim, semantic_dim)
        else:
            mid_dim = hidden_dim // 2
            layers = [
                nn.Linear(hidden_dim, mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, semantic_dim)
            ]
            self.semantic_projection = nn.Sequential(*layers)

        # SSL teacher投影：将SSL高维表示投影到语义空间进行对齐
        self.ssl_to_semantic = nn.Sequential(
            nn.Linear(ssl_dim, ssl_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ssl_dim // 2, semantic_dim)
        )

        # 可选的归一化层
        if use_layer_norm:
            self.output_norm = nn.LayerNorm(semantic_dim)
            self.ssl_norm = nn.LayerNorm(semantic_dim)
        else:
            self.output_norm = nn.Identity()
            self.ssl_norm = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """权重初始化：确保训练初期的稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 小初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, hidden_dim]
    ) -> torch.Tensor:
        """
        从隐状态提取语义特征

        Args:
            hidden_states: decoder的隐状态 [B, T, hidden_dim]

        Returns:
            semantic_features: [B, T, semantic_dim]
        """
        semantic_raw = self.semantic_projection(hidden_states)
        semantic_features = self.output_norm(semantic_raw)
        return semantic_features

    def project_ssl_teacher(
        self,
        ssl_features: torch.Tensor,  # [B, T, ssl_dim]
    ) -> torch.Tensor:
        """
        将SSL teacher特征投影到语义空间

        Args:
            ssl_features: SSL模型输出 [B, T, ssl_dim]

        Returns:
            ssl_semantic: [B, T, semantic_dim]
        """
        ssl_projected = self.ssl_to_semantic(ssl_features)
        ssl_semantic = self.ssl_norm(ssl_projected)
        return ssl_semantic

    def compute_semantic_loss(
        self,
        pred_semantic: torch.Tensor,    # [B, T, semantic_dim]
        ssl_features: torch.Tensor,     # [B, T, ssl_dim]
        loss_type: str = "cosine",
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
        ssl_semantic = self.project_ssl_teacher(ssl_features)  # [B, T, semantic_dim]

        if loss_type == "cosine":
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)  # [B, T]
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
            # InfoNCE对比学习损失
            # 简化版本：frame-level对比
            B, T, D = pred_semantic.shape
            pred_flat = pred_semantic.view(-1, D)      # [B*T, D]
            ssl_flat = ssl_semantic.view(-1, D)        # [B*T, D]

            # 计算相似度矩阵
            logits = torch.matmul(pred_flat, ssl_flat.t()) / 0.1  # [B*T, B*T]
            targets = torch.arange(B * T, device=logits.device)

            loss = F.cross_entropy(logits, targets)

            with torch.no_grad():
                cos_sim = F.cosine_similarity(pred_semantic, ssl_semantic, dim=-1)
                metrics = {
                    "semantic_infonce": loss.item(),
                    "semantic_cos_sim": cos_sim.mean().item(),
                }
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return loss, metrics


class AcousticHead(nn.Module):
    """
    声学头：从decoder隐状态中提取FARGAN需要的20维声学特征
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        acoustic_dim: int = 20,
        use_layer_norm: bool = False,  # 声学特征通常不需要归一化
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.acoustic_dim = acoustic_dim

        # 简单的线性投影即可，声学特征相对稳定
        self.acoustic_projection = nn.Linear(hidden_dim, acoustic_dim)

        if use_layer_norm:
            self.output_norm = nn.LayerNorm(acoustic_dim)
        else:
            self.output_norm = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.acoustic_projection.weight)
        if self.acoustic_projection.bias is not None:
            nn.init.zeros_(self.acoustic_projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从隐状态提取声学特征

        Args:
            hidden_states: [B, T, hidden_dim]

        Returns:
            acoustic_features: [B, T, 20]
        """
        acoustic_raw = self.acoustic_projection(hidden_states)
        acoustic_features = self.output_norm(acoustic_raw)
        return acoustic_features


class DualHeadDecoder(nn.Module):
    """
    双头解码器：共享隐状态，分别输出声学特征和语义特征

    核心设计：
    - 共享的backbone提取隐状态表示
    - 声学头输出20维 → FARGAN
    - 语义头输出16维 → SSL teacher对齐
    - 通过多任务loss约束隐空间既保留声学信息又保留语义信息
    """

    def __init__(
        self,
        input_dim: int = 128,            # 输入维度(如AETHER encoder输出)
        hidden_dim: int = 256,           # 隐状态维度
        acoustic_dim: int = 20,          # 声学特征维度
        semantic_dim: int = 16,          # 语义特征维度
        ssl_dim: int = 768,              # SSL模型维度
        num_layers: int = 2,             # backbone层数
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 共享backbone：提取隐状态表示
        backbone_layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # 双头输出
        self.acoustic_head = AcousticHead(hidden_dim, acoustic_dim)
        self.semantic_head = SemanticHead(hidden_dim, semantic_dim, ssl_dim)

    def forward(
        self,
        inputs: torch.Tensor,           # [B, T, input_dim]
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            inputs: 输入特征 [B, T, input_dim]
            return_hidden: 是否返回隐状态

        Returns:
            outputs: {
                'acoustic_features': [B, T, 20],
                'semantic_features': [B, T, 16],
                'hidden_states': [B, T, hidden_dim] (可选)
            }
        """
        # 共享隐状态
        hidden_states = self.backbone(inputs)  # [B, T, hidden_dim]

        # 双头输出
        acoustic_features = self.acoustic_head(hidden_states)   # [B, T, 20]
        semantic_features = self.semantic_head(hidden_states)   # [B, T, 16]

        outputs = {
            'acoustic_features': acoustic_features,
            'semantic_features': semantic_features,
        }

        if return_hidden:
            outputs['hidden_states'] = hidden_states

        return outputs

    def compute_semantic_loss(
        self,
        semantic_features: torch.Tensor,
        ssl_features: torch.Tensor,
        loss_type: str = "cosine"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算语义对齐损失"""
        return self.semantic_head.compute_semantic_loss(
            semantic_features, ssl_features, loss_type
        )


if __name__ == "__main__":
    # 测试双头解码器
    print("Testing DualHeadDecoder...")

    # 创建测试数据
    batch_size, seq_len = 2, 100
    input_dim, hidden_dim = 128, 256

    inputs = torch.randn(batch_size, seq_len, input_dim)
    ssl_features = torch.randn(batch_size, seq_len, 768)  # HuBERT特征

    # 创建双头解码器
    decoder = DualHeadDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        acoustic_dim=20,
        semantic_dim=16,
        ssl_dim=768,
    )

    # 前向传播
    outputs = decoder(inputs, return_hidden=True)

    print(f"Input shape: {inputs.shape}")
    print(f"Acoustic features shape: {outputs['acoustic_features'].shape}")
    print(f"Semantic features shape: {outputs['semantic_features'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")

    # 测试语义损失
    semantic_loss, metrics = decoder.compute_semantic_loss(
        outputs['semantic_features'],
        ssl_features,
        loss_type="cosine"
    )

    print(f"\nSemantic loss: {semantic_loss.item():.6f}")
    print(f"Metrics: {metrics}")

    print("\nDualHeadDecoder test completed!")