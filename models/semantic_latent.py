import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class SemanticAdapter(nn.Module):
    """
    SemanticAdapter：
    从 decoder 的 semantic_raw + 可选 teacher 语义，构建"我们的统一潜空间 z_sem"。

    - 输入：
        semantic_raw: [B, T, D_sem_raw]，例如 16 维
        teacher_features: [B, T, D_teacher]，例如 SSL 768 / StableCodec latent
        mask: [B, T]，1=有效，0=padding（可选）
    - 输出：
        z_sem: [B, T, D_sem]，我们自己的语义潜空间
        logs: 一些监控指标
    """

    def __init__(
        self,
        semantic_raw_dim: int = 16,
        teacher_dim: int = 768,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.semantic_raw_dim = semantic_raw_dim
        self.teacher_dim = teacher_dim
        self.latent_dim = latent_dim
        self.use_cross_attention = use_cross_attention

        # 1) 学生侧：16 维 semantic_raw → hidden
        self.student_proj = nn.Sequential(
            nn.Linear(semantic_raw_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2) teacher 侧：高维 teacher → hidden
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3) 融合模块：可以是简单的 concat+MLP，或者 cross-attn
        if use_cross_attention:
            # 语义时间序列之间做 cross-attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.cross_attn = None
            self.fuse_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # 4) 最终投影到我们的潜空间 z_sem
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        semantic_raw: torch.Tensor,           # [B, T, D_sem_raw]
        teacher_features: Optional[torch.Tensor] = None,  # [B, T, D_teacher] 或已插值对齐
        mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, _ = semantic_raw.shape
        logs: Dict[str, float] = {}

        # 学生侧
        h_s = self.student_proj(semantic_raw)    # [B, T, H]

        if teacher_features is not None:
            # 假定 teacher_features 时间轴已经和 semantic_raw 对齐
            h_t = self.teacher_proj(teacher_features)   # [B, T, H]

            if self.cross_attn is not None:
                # Cross-Attn：学生 query，teacher key/value
                attn_mask = None
                if mask is not None:
                    # MultiheadAttention 的 key_padding_mask: True=mask
                    key_padding_mask = (mask == 0)      # [B, T]
                else:
                    key_padding_mask = None

                z, attn_weights = self.cross_attn(
                    query=h_s, key=h_t, value=h_t,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )  # [B, T, H]
                fused = z
            else:
                fused = self.fuse_mlp(torch.cat([h_s, h_t], dim=-1))  # [B,T,H]
        else:
            # 无 teacher 时，退化为只用 semantic_raw 的自编码
            fused = h_s

        z_sem = self.latent_proj(fused)          # [B, T, D_sem]
        z_sem = self.output_norm(z_sem)

        # 简单的监控
        with torch.no_grad():
            logs["z_sem_mean"] = z_sem.mean().item()
            logs["z_sem_std"] = z_sem.std().item()

        return z_sem, logs


class LatentSpaceHead(nn.Module):
    """
    LatentSpaceHead：
    从统一潜空间 z_sem 提取用于：
      1) 拼回 36 维 (作为"语义 part")
      2) 和 teacher 特征做语义对齐的表征

    - 输入：
        z_sem: [B, T, D_sem]
        teacher_features: [B, T, D_teacher] (可选，用于loss)
        mask: [B, T] (可选)
    - 输出：
        semantic_features: [B, T, semantic_dim] → 拼回 36 维
        loss, metrics: 语义对齐损失
    """

    def __init__(
        self,
        latent_dim: int = 64,
        semantic_dim: int = 16,
        teacher_dim: int = 768,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        loss_type: str = "cosine+infoce",
        temperature: float = 0.07,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.semantic_dim = semantic_dim
        self.teacher_dim = teacher_dim
        self.loss_type = loss_type
        self.temperature = temperature

        # z_sem → 语义特征
        self.latent_to_semantic = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, semantic_dim),
        )

        # teacher → 语义空间
        self.teacher_to_semantic = nn.Sequential(
            nn.Linear(teacher_dim, teacher_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(teacher_dim // 2, semantic_dim),
        )

        if use_layer_norm:
            self.semantic_norm = nn.LayerNorm(semantic_dim)
        else:
            self.semantic_norm = nn.Identity()

    def forward(
        self,
        z_sem: torch.Tensor,                      # [B, T, D_sem]
        teacher_features: Optional[torch.Tensor] = None,  # [B, T, D_teacher]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        返回：
            semantic_features: [B, T, semantic_dim]
            semantic_loss: 标量张量
            metrics: dict
        """
        semantic_features = self.semantic_norm(
            self.latent_to_semantic(z_sem)
        )  # [B, T, semantic_dim]

        if teacher_features is None:
            loss = z_sem.new_tensor(0.0)
            metrics = {}
            return semantic_features, loss, metrics

        # teacher 投影到语义空间
        teacher_sem = self.teacher_to_semantic(teacher_features)   # [B, T, semantic_dim]

        # 可选 mask：只统计有效帧
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)  # [B,T,1]
            semantic_features_masked = semantic_features * mask_f
            teacher_sem_masked = teacher_sem * mask_f
        else:
            semantic_features_masked = semantic_features
            teacher_sem_masked = teacher_sem

        # ---- 构造语义对齐损失 ----
        loss = 0.0
        metrics: Dict[str, float] = {}

        if "cosine" in self.loss_type:
            cos = F.cosine_similarity(
                semantic_features_masked, teacher_sem_masked, dim=-1
            )
            if mask is not None:
                cos = (cos * mask).sum() / (mask.sum() + 1e-6)
            loss_cos = 1.0 - cos.mean()
            loss = loss + loss_cos
            metrics["sem_cos"] = cos.mean().item()

        if "mse" in self.loss_type:
            if mask is not None:
                diff = (semantic_features - teacher_sem) * mask_f
                loss_mse = (diff ** 2).sum() / (mask_f.sum() * self.semantic_dim + 1e-6)
            else:
                loss_mse = F.mse_loss(semantic_features, teacher_sem)
            loss = loss + loss_mse
            metrics["sem_mse"] = loss_mse.item()

        if "infoce" in self.loss_type or "infonce" in self.loss_type:
            # flatten 到 [N, D]
            try:
                B, T, D = semantic_features_masked.shape
                print(f"🔍 InfoNCE Debug - Input shapes: semantic_masked {semantic_features_masked.shape}, teacher_masked {teacher_sem_masked.shape}")

                q = semantic_features_masked.reshape(-1, D)   # [N,D]
                k = teacher_sem_masked.reshape(-1, D)         # [N,D]
                print(f"🔍 InfoNCE Debug - After reshape: q {q.shape}, k {k.shape}")

                if mask is not None:
                    # 只保留 mask==1 的位置
                    flat_mask = mask.view(-1)
                    print(f"🔍 InfoNCE Debug - Mask shape: {flat_mask.shape}, valid count: {(flat_mask > 0.5).sum()}")
                    q = q[flat_mask > 0.5]
                    k = k[flat_mask > 0.5]
                    print(f"🔍 InfoNCE Debug - After masking: q {q.shape}, k {k.shape}")

                # 检查矩阵形状兼容性
                if q.size(0) != k.size(0):
                    print(f"❌ InfoNCE Error - Shape mismatch: q {q.shape} vs k {k.shape}")
                    raise ValueError(f"InfoNCE shape mismatch: q {q.shape} vs k {k.shape}")

                if q.size(0) == 0:
                    print("⚠️ InfoNCE Warning - Empty tensors after masking, skipping InfoNCE")
                    loss_nce = torch.tensor(0.0, device=q.device, requires_grad=True)
                else:
                    # 正样本对齐，负样本为其他位置
                    q = F.normalize(q, dim=-1)
                    k = F.normalize(k, dim=-1)
                    print(f"🔍 InfoNCE Debug - After normalization: q {q.shape}, k {k.shape}")

                    logits = (q @ k.t()) / self.temperature     # [N,N]
                    print(f"🔍 InfoNCE Debug - Logits shape: {logits.shape}")

                    labels = torch.arange(logits.size(0), device=logits.device)
                    print(f"🔍 InfoNCE Debug - Labels shape: {labels.shape}")

                    loss_nce = F.cross_entropy(logits, labels)
                    print(f"✅ InfoNCE Success - Loss: {loss_nce.item():.6f}")

                loss = loss + loss_nce
                metrics["sem_nce"] = loss_nce.item()

            except Exception as e:
                print(f"❌ InfoNCE Exception: {e}")
                import traceback
                print(f"🔍 InfoNCE Stack Trace: {traceback.format_exc()}")
                # 回退到简单损失
                loss_nce = torch.tensor(0.0, device=semantic_features.device, requires_grad=True)
                loss = loss + loss_nce
                metrics["sem_nce"] = 0.0

        metrics["sem_loss"] = float(loss.item())
        return semantic_features, loss, metrics


class AcousticFusionHead(nn.Module):
    """
    AcousticFusionHead：
    使用统一潜空间 z_sem (或 semantic_features) 指导 20 维声学特征，
    输出给 FARGAN 的 acoustic_enhanced。

    支持几种融合方式：
      - attention: z_sem 作为 query，acoustic 作为 key/value
      - gate: z_sem 生成门控信号，对 acoustic 做逐维调制
      - cross_mlp: 简单 concat(z_sem, acoustic) 送进 MLP
    """

    def __init__(
        self,
        acoustic_dim: int = 20,
        latent_dim: int = 64,
        fusion_type: str = "attention",   # "attention" / "gate" / "cross_mlp"
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.acoustic_dim = acoustic_dim
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            # z_sem 作为 query，acoustic 作为 key/value
            self.q_proj = nn.Linear(latent_dim, hidden_dim)
            self.k_proj = nn.Linear(acoustic_dim, hidden_dim)
            self.v_proj = nn.Linear(acoustic_dim, hidden_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, acoustic_dim),
                nn.GELU(),
                nn.Linear(acoustic_dim, acoustic_dim),
            )

        elif fusion_type == "gate":
            # 从 z_sem 预测出 [0,1] 的 gate，对 acoustic 做逐点放缩
            self.gate_proj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, acoustic_dim),
                nn.Sigmoid(),
            )
            self.residual = nn.Sequential(
                nn.Linear(acoustic_dim, acoustic_dim),
                nn.GELU(),
                nn.Linear(acoustic_dim, acoustic_dim),
            )

        elif fusion_type == "cross_mlp":
            # concat(acoustic, z_sem) → MLP → acoustic
            self.cross_mlp = nn.Sequential(
                nn.Linear(acoustic_dim + latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, acoustic_dim),
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.output_norm = nn.LayerNorm(acoustic_dim)

    def forward(
        self,
        acoustic_raw: torch.Tensor,   # [B, T, 20]
        z_sem: torch.Tensor,          # [B, T, D_sem]
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logs: Dict[str, float] = {}

        if self.fusion_type == "attention":
            Q = self.q_proj(z_sem)       # [B,T,H]
            K = self.k_proj(acoustic_raw)
            V = self.v_proj(acoustic_raw)

            key_padding_mask = None
            if mask is not None:
                key_padding_mask = (mask == 0)   # [B,T]

            attn_out, _ = self.attn(
                query=Q, key=K, value=V,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )  # [B,T,H]
            fused = self.out_proj(attn_out)      # [B,T,20]
            acoustic_enh = acoustic_raw + fused  # 残差

        elif self.fusion_type == "gate":
            gate = self.gate_proj(z_sem)         # [B,T,20], in (0,1)
            residual = self.residual(acoustic_raw)
            acoustic_enh = acoustic_raw + gate * residual

            with torch.no_grad():
                logs["gate_mean"] = gate.mean().item()
                logs["gate_std"] = gate.std().item()

        else:  # cross_mlp
            fused_in = torch.cat([acoustic_raw, z_sem], dim=-1)  # [B,T,20+D_sem]
            delta = self.cross_mlp(fused_in)
            acoustic_enh = acoustic_raw + delta

        acoustic_enh = self.output_norm(acoustic_enh)
        with torch.no_grad():
            logs["acoustic_enh_mean"] = acoustic_enh.mean().item()
            logs["acoustic_enh_std"] = acoustic_enh.std().item()

        return acoustic_enh, logs