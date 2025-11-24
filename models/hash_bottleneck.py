import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any


class GreedyHashFunction(Function):
    """
    GreedyHash with Straight-Through Estimator
    Forward: h = sign(u)
    Backward: gradient passes through unchanged
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class BiHalfHashFunction(Function):
    """
    Bi-half hash layer for maximum bit entropy
    Ensures each bit dimension has 50% +1 and 50% -1 across the batch
    """
    @staticmethod
    def forward(ctx, U: torch.Tensor, gamma: float = 6.0) -> torch.Tensor:
        # U: [B, T, K] - batch, time, hash_bits
        original_shape = U.shape

        # Flatten to [B*T, K] for processing
        U_flat = U.view(-1, U.size(-1))

        # Sort each bit dimension independently
        _, index = U_flat.sort(0, descending=True)
        N, D = U_flat.shape

        # Create balanced binary assignment: top 50% -> +1, bottom 50% -> -1
        B_creat = torch.cat([
            torch.ones([int(N/2), D], device=U.device),
            -torch.ones([N - int(N/2), D], device=U.device)
        ])

        B_flat = torch.zeros_like(U_flat).scatter_(0, index, B_creat)
        B = B_flat.view(original_shape)

        ctx.save_for_backward(U, B)
        ctx.gamma = gamma

        return B

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        U, B = ctx.saved_tensors
        gamma = ctx.gamma

        # Add regularization gradient to encourage U -> B
        add_g = (U - B) / (B.numel())
        grad = grad_output + gamma * add_g

        return grad, None


class HashBottleneck(nn.Module):
    """
    Binary Hash Bottleneck for Speech JSCC

    Combines:
    1. Linear projection to hash logits
    2. Bi-half + GreedyHash binarization
    3. Bit channel simulation (BSC/BPSK+AWGN)
    4. Hash decoder for reconstruction
    """

    def __init__(self,
                 input_dim: int,
                 hash_bits: int,
                 decoder_hidden: int = 128,
                 output_dim: Optional[int] = None,
                 hash_method: str = 'bihalf',
                 gamma: float = 6.0,
                 channel_type: str = 'bsc'):
        super().__init__()

        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.output_dim = output_dim or input_dim
        self.hash_method = hash_method
        self.gamma = gamma
        self.channel_type = channel_type

        # Hash encoder: continuous -> hash logits
        self.hash_encoder = nn.Linear(input_dim, hash_bits)

        # Hash decoder: bits -> continuous latent
        self.hash_decoder = nn.Sequential(
            nn.Linear(hash_bits, decoder_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_hidden, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def hash_layer(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply hash function based on method"""
        if self.hash_method == 'greedy':
            return GreedyHashFunction.apply(logits)
        elif self.hash_method == 'bihalf':
            return BiHalfHashFunction.apply(logits, self.gamma)
        elif self.hash_method == 'sign':
            return torch.sign(logits)
        else:
            raise ValueError(f"Unknown hash method: {self.hash_method}")

    def channel_simulation(self,
                          bits: torch.Tensor,
                          channel_params: Dict[str, float]) -> torch.Tensor:
        """
        Simulate bit-level channel noise

        Args:
            bits: Binary hash codes [B, T, K]
            channel_params: Channel parameters (e.g., {'ber': 0.1, 'snr_db': 10})
        """
        if not self.training:
            return bits

        if self.channel_type == 'bsc':
            # Binary Symmetric Channel
            ber = channel_params.get('ber', 0.1)
            flip_mask = torch.rand_like(bits) < ber
            return torch.where(flip_mask, -bits, bits)

        elif self.channel_type == 'bpsk_awgn':
            # BPSK + AWGN
            snr_db = channel_params.get('snr_db', 10)
            noise_power = 10 ** (-snr_db / 10)
            noise = torch.randn_like(bits) * np.sqrt(noise_power)
            return bits + noise

        else:
            return bits

    def forward(self,
                x: torch.Tensor,
                channel_params: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hash bottleneck

        Args:
            x: Input latent [B, T, D]
            channel_params: Channel simulation parameters

        Returns:
            Dictionary with:
            - hash_logits: Pre-binarization logits [B, T, K]
            - hash_bits_clean: Clean binary hash [B, T, K]
            - hash_bits_noisy: Channel-corrupted hash [B, T, K]
            - reconstructed: Decoded continuous latent [B, T, D_out]
        """
        # Encode to hash logits
        hash_logits = self.hash_encoder(x)  # [B, T, K]

        # Binarize using selected hash method
        hash_bits_clean = self.hash_layer(hash_logits)

        # Channel simulation
        if channel_params is not None:
            hash_bits_noisy = self.channel_simulation(hash_bits_clean, channel_params)
        else:
            hash_bits_noisy = hash_bits_clean

        # Decode back to continuous space
        reconstructed = self.hash_decoder(hash_bits_noisy)

        return {
            'hash_logits': hash_logits,
            'hash_bits_clean': hash_bits_clean,
            'hash_bits_noisy': hash_bits_noisy,
            'reconstructed': reconstructed
        }

    def compute_hash_regularization(self,
                                   hash_logits: torch.Tensor,
                                   hash_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hash-specific regularization losses
        """
        losses = {}

        # Bit balance: encourage each bit to be 50% +1, 50% -1
        bit_means = hash_bits.mean(dim=[0, 1])  # [K]
        balance_target = torch.zeros_like(bit_means)
        losses['bit_balance'] = F.mse_loss(bit_means, balance_target)

        # Bit decorrelation: minimize correlation between different bits
        hash_flat = hash_bits.view(-1, hash_bits.size(-1))  # [B*T, K]
        if hash_flat.size(0) > 1:
            correlation_matrix = torch.corrcoef(hash_flat.t())  # [K, K]
            # Penalize off-diagonal elements
            mask = ~torch.eye(correlation_matrix.size(0), dtype=torch.bool, device=correlation_matrix.device)
            losses['bit_decorrelation'] = correlation_matrix[mask].abs().mean()
        else:
            losses['bit_decorrelation'] = torch.tensor(0.0, device=hash_bits.device)

        # Quantization loss: encourage logits to be close to ±1
        losses['quantization'] = torch.mean(torch.abs(torch.abs(hash_logits) - 1.0))

        # Entropy regularization: maximum entropy per bit
        probs = torch.sigmoid(hash_logits)
        entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
        # Negative because we want to maximize entropy when used as a loss term
        losses['entropy'] = -entropy.mean()

        # Rate loss: Bernoulli KL(q(b|x) || Bernoulli(0.5))
        # This provides an explicit "rate" term that can be weighted separately.
        p = torch.clamp(probs, 1e-6, 1.0 - 1e-6)
        prior_p = 0.5
        log_prior = math.log(prior_p)
        log_prior_comp = math.log(1.0 - prior_p)
        rate_kl = p * (torch.log(p) - log_prior) + (1.0 - p) * (torch.log(1.0 - p) - log_prior_comp)
        losses['rate_kl'] = rate_kl.mean()

        return losses

    def get_bitrate(self, frame_rate: float = 50.0) -> float:
        """Calculate nominal bitrate in bps"""
        return self.hash_bits * frame_rate

    @torch.no_grad()
    def analyze_bit_statistics(self, hash_bits: torch.Tensor) -> Dict[str, float]:
        """Analyze hash bit statistics for monitoring"""
        hash_flat = hash_bits.view(-1, hash_bits.size(-1))

        stats = {}
        stats['bit_balance'] = hash_flat.mean(dim=0).abs().mean().item()
        stats['bit_utilization'] = (hash_flat.abs().mean()).item()

        if hash_flat.size(0) > 1:
            corr_matrix = torch.corrcoef(hash_flat.t())
            mask = ~torch.eye(
                corr_matrix.size(0),
                dtype=torch.bool,
                device=corr_matrix.device,
            )
            stats['bit_correlation'] = corr_matrix[mask].abs().mean().item()
        else:
            stats['bit_correlation'] = 0.0

        return stats


class TeacherDistillationModule(nn.Module):
    """
    Teacher distillation for hash bottleneck learning
    Supports various teacher models (StableCodec, HuBERT, etc.)
    """

    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 temperature: float = 1.0):
        super().__init__()

        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.temperature = temperature

        # Alignment projection if dimensions differ
        if teacher_dim != student_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()

    def forward(self,
                student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute distillation losses

        Args:
            student_features: Student latent [B, T, D_s]
            teacher_features: Teacher latent [B, T, D_t]
        """
        projected_student = self.projection(student_features)

        losses = {}

        # Feature alignment loss
        losses['feature_mse'] = F.mse_loss(projected_student, teacher_features)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(projected_student, teacher_features, dim=-1)
        losses['cosine_similarity'] = 1 - cos_sim.mean()

        # InfoNCE-style contrastive loss (optional)
        if teacher_features.size(1) > 1:  # Temporal dimension > 1
            # Positive pairs: same time step
            # Negative pairs: different time steps
            pos_sim = F.cosine_similarity(projected_student, teacher_features, dim=-1)

            # Create negative pairs by shifting
            neg_teacher = torch.roll(teacher_features, shifts=1, dims=1)
            neg_sim = F.cosine_similarity(projected_student, neg_teacher, dim=-1)

            logits = torch.stack([pos_sim, neg_sim], dim=-1) / self.temperature
            targets = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=logits.device)

            losses['contrastive'] = F.cross_entropy(logits.view(-1, 2), targets.view(-1))
        else:
            losses['contrastive'] = torch.tensor(0.0, device=student_features.device)

        return losses


if __name__ == "__main__":
    # Test the hash bottleneck
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create test data
    B, T, D = 2, 100, 36  # batch=2, time=100frames, features=36D
    K = 16  # 16 bits per frame -> ~800 bps at 50Hz

    x = torch.randn(B, T, D).to(device)

    # Initialize hash bottleneck
    hash_bottleneck = HashBottleneck(
        input_dim=D,
        hash_bits=K,
        hash_method='bihalf',
        channel_type='bsc'
    ).to(device)

    # Forward pass with channel simulation
    channel_params = {'ber': 0.1}  # 10% bit error rate

    results = hash_bottleneck(x, channel_params)

    print("Hash Bottleneck Test:")
    print(f"Input shape: {x.shape}")
    print(f"Hash logits shape: {results['hash_logits'].shape}")
    print(f"Hash bits shape: {results['hash_bits_clean'].shape}")
    print(f"Reconstructed shape: {results['reconstructed'].shape}")
    print(f"Nominal bitrate: {hash_bottleneck.get_bitrate():.1f} bps")

    # Test regularization
    reg_losses = hash_bottleneck.compute_hash_regularization(
        results['hash_logits'], results['hash_bits_clean']
    )
    print(f"Regularization losses: {reg_losses}")

    # Test bit statistics
    stats = hash_bottleneck.analyze_bit_statistics(results['hash_bits_clean'])
    print(f"Bit statistics: {stats}")

    print("✅ Hash bottleneck implementation complete!")
