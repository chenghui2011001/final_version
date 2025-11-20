# -*- coding: utf-8 -*-
"""
KAN-Lite FiLM generator that maps CSI metadata to affine modulation curves.
Includes B-spline basis functions for interpretable channel adaptation.
"""

from __future__ import annotations

import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class KANLiteFiLM(nn.Module):
    """
    Lightweight module that converts channel state information into FiLM
    parameters ``alpha`` and ``beta`` with optional temporal variation.
    """

    def __init__(
        self,
        d_csi: int,
        d_feat: int,
        time_dependent: bool = False,
        hidden_multiplier: int = 2,
    ) -> None:
        super().__init__()
        self.time_dependent = time_dependent
        hidden = max(32, d_csi * hidden_multiplier)

        self.fc1 = nn.Linear(d_csi, hidden)
        self.act = nn.SiLU()
        self.fc_alpha = nn.Linear(hidden, d_feat)
        self.fc_beta = nn.Linear(hidden, d_feat)
        self.alpha_gate = nn.Sigmoid()

        if time_dependent:
            self.temporal_mod = nn.GRU(
                input_size=d_feat,
                hidden_size=d_feat,
                batch_first=True,
            )

    def _expand_temporal(self, params: torch.Tensor, T: int) -> torch.Tensor:
        # —— 防止上游传入二维导致后面 permute 报错 —— #
        if params.dim() == 2:              # [B, D] -> [B, 1, D]
            params = params.unsqueeze(1)

        if T <= 1 or params.size(1) == T:
            return params
        if not self.time_dependent:
            return params.expand(-1, T, -1)
        repeated = params.expand(-1, T, -1).contiguous()
        out, _ = self.temporal_mod(repeated)
        return out


    def forward(self, csi_vec: torch.Tensor, T: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            csi_vec: Tensor with shape ``[B, d_csi]``.
            T: Desired temporal length for the FiLM parameters.

        Returns:
            Tuple ``(alpha, beta)`` where both tensors have shape ``[B, T, D]``.
        """
        if csi_vec.dim() != 2:
            raise ValueError("csi_vec must be of shape [B, d_csi]")

        # Ensure all parameters are on the same device as input
        device = csi_vec.device
        if self.fc1.weight.device != device:
            self.to(device)

        h = self.act(self.fc1(csi_vec))
        alpha = self.alpha_gate(self.fc_alpha(h)).unsqueeze(1)
        beta = self.fc_beta(h).unsqueeze(1)

        alpha = self._expand_temporal(alpha, T)
        beta = self._expand_temporal(beta, T)

        return alpha, beta


class BSplineBasis(nn.Module):
    """
    B-spline basis functions for KAN-field interpolation.
    """

    def __init__(self, degree: int = 3, num_knots: int = 8):
        super().__init__()
        self.degree = degree
        self.num_knots = num_knots

        # Create uniform knot vector
        knots = torch.linspace(-1, 1, num_knots)
        # Extend knots for B-spline evaluation
        extended_knots = torch.cat([
            torch.full((degree,), knots[0]),
            knots,
            torch.full((degree,), knots[-1])
        ])
        self.register_buffer('knots', extended_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions using simplified RBF-like approach.
        Args:
            x: Input tensor [B, ...] in range [-1, 1]
        Returns:
            Basis evaluations [B, ..., num_basis]
        """
        # Ensure knots are on the same device as input
        if self.knots.device != x.device:
            self.knots = self.knots.to(x.device)

        # Clamp input to valid range
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)

        batch_shape = x.shape
        x_flat = x.reshape(-1, 1)  # [N, 1]

        # Use simplified Gaussian-like basis functions centered at knot positions
        internal_knots = self.knots[self.degree:-self.degree]  # Remove boundary knots
        centers = internal_knots.view(1, -1)  # [1, num_basis]

        # Compute distances and apply Gaussian-like basis
        width = 2.0 / (len(internal_knots) - 1) if len(internal_knots) > 1 else 1.0
        distances = (x_flat - centers) / width  # [N, num_basis]
        basis = torch.exp(-0.5 * distances ** 2)  # Gaussian basis

        # Normalize to ensure partition of unity
        basis_sum = basis.sum(dim=1, keepdim=True) + 1e-8
        basis = basis / basis_sum

        return basis.reshape(*batch_shape, -1)

    def _cox_de_boor(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """Cox-de Boor recursion for B-spline basis functions."""
        if k == 0:
            # Handle boundary case for last knot
            if i == len(self.knots) - 2:
                return ((x >= self.knots[i]) & (x <= self.knots[i + 1])).float()
            else:
                return ((x >= self.knots[i]) & (x < self.knots[i + 1])).float()
        else:
            # Recursive evaluation
            left_term = torch.zeros_like(x)
            right_term = torch.zeros_like(x)

            # Avoid division by zero
            if abs(self.knots[i + k] - self.knots[i]) > 1e-8:
                left_term = (x - self.knots[i]) / (self.knots[i + k] - self.knots[i]) * \
                           self._cox_de_boor(x, i, k - 1)

            if abs(self.knots[i + k + 1] - self.knots[i + 1]) > 1e-8:
                right_term = (self.knots[i + k + 1] - x) / (self.knots[i + k + 1] - self.knots[i + 1]) * \
                            self._cox_de_boor(x, i + 1, k - 1)

            return left_term + right_term


class KANFieldEnhanced(nn.Module):
    """
    Enhanced KAN-Field with B-spline basis functions and CSI encoding.
    """

    def __init__(
        self,
        d_csi: int,
        d_feat: int,
        num_knots: int = 8,
        spline_degree: int = 3,
        time_dependent: bool = False,
    ):
        super().__init__()
        self.d_csi = d_csi
        self.d_feat = d_feat
        self.time_dependent = time_dependent

        # B-spline basis
        self.bspline = BSplineBasis(degree=spline_degree, num_knots=num_knots)
        # For simplified Gaussian basis, we actually use all internal knots
        # Create a dummy input to get actual basis dimension
        dummy_input = torch.tensor([0.0])
        with torch.no_grad():
            dummy_basis = self.bspline(dummy_input)
            num_basis = dummy_basis.shape[-1]

        # CSI encoder: maps channel state to spline coefficients
        self.csi_encoder = nn.Sequential(
            nn.Linear(d_csi, 64),
            nn.SiLU(),
            nn.Linear(64, num_basis * d_feat * 2)  # For both alpha and beta
        )

        # Optional temporal modulation
        if time_dependent:
            self.temporal_mod = nn.GRU(
                input_size=d_feat,
                hidden_size=d_feat,
                batch_first=True,
            )

    def forward(
        self,
        csi_vec: torch.Tensor,
        snr_normalized: Optional[torch.Tensor] = None,
        T: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            csi_vec: CSI vector [B, d_csi]
            snr_normalized: Normalized SNR values [B] in range [-1, 1]
            T: Temporal length
        Returns:
            (alpha, beta) modulation parameters [B, T, d_feat]
        """
        batch_size = csi_vec.shape[0]

        # Ensure all parameters are on the same device as input
        device = csi_vec.device
        if next(self.csi_encoder.parameters()).device != device:
            self.to(device)

        if snr_normalized is None:
            # Extract SNR from CSI if available, otherwise use default
            snr_normalized = torch.zeros(batch_size, device=device)

        # Get spline coefficients from CSI
        coeffs = self.csi_encoder(csi_vec)  # [B, num_basis * d_feat * 2]
        coeffs = coeffs.reshape(batch_size, -1, self.d_feat, 2)  # [B, num_basis, d_feat, 2]

        # Evaluate B-spline basis functions
        basis = self.bspline(snr_normalized)  # [B, num_basis]

        # Compute alpha and beta using B-spline interpolation
        alpha_coeffs = coeffs[..., 0]  # [B, num_basis, d_feat]
        beta_coeffs = coeffs[..., 1]   # [B, num_basis, d_feat]

        alpha = torch.einsum('bn,bnf->bf', basis, alpha_coeffs).unsqueeze(1)  # [B, 1, d_feat]
        beta = torch.einsum('bn,bnf->bf', basis, beta_coeffs).unsqueeze(1)   # [B, 1, d_feat]

        # Apply sigmoid to alpha for stability
        alpha = torch.sigmoid(alpha)

        # Expand to temporal dimension if needed
        if T > 1:
            alpha = alpha.expand(-1, T, -1)
            beta = beta.expand(-1, T, -1)

            if self.time_dependent and hasattr(self, 'temporal_mod'):
                alpha, _ = self.temporal_mod(alpha)
                beta, _ = self.temporal_mod(beta)

        return alpha, beta


def visualize_bspline_curves(
    kan_field: KANFieldEnhanced,
    csi_samples: torch.Tensor,
    snr_range: Tuple[float, float] = (-10, 20),
    save_path: str = "kan_field_curves.png"
):
    """
    Visualize KAN-field modulation curves across SNR range.
    """
    snr_values = torch.linspace(snr_range[0], snr_range[1], 100)
    snr_normalized = 2 * (snr_values - snr_range[0]) / (snr_range[1] - snr_range[0]) - 1

    plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for i, csi in enumerate(csi_samples):
            csi_batch = csi.unsqueeze(0)  # [1, d_csi]

            alphas = []
            betas = []

            for snr_norm in snr_normalized:
                alpha, beta = kan_field(csi_batch, snr_norm.unsqueeze(0))
                alphas.append(alpha[0, 0, :5])  # First 5 features
                betas.append(beta[0, 0, :5])

            alphas = torch.stack(alphas)  # [100, 5]
            betas = torch.stack(betas)

            # Plot alpha curves
            plt.subplot(2, 2, 1)
            for j in range(5):
                plt.plot(snr_values, alphas[:, j], label=f'CSI{i}_feat{j}', alpha=0.7)
            plt.title('Alpha Modulation Curves')
            plt.xlabel('SNR (dB)')
            plt.ylabel('Alpha')
            plt.grid(True)

            # Plot beta curves
            plt.subplot(2, 2, 2)
            for j in range(5):
                plt.plot(snr_values, betas[:, j], label=f'CSI{i}_feat{j}', alpha=0.7)
            plt.title('Beta Modulation Curves')
            plt.xlabel('SNR (dB)')
            plt.ylabel('Beta')
            plt.grid(True)

    # Plot B-spline basis functions
    plt.subplot(2, 2, 3)
    basis_vals = kan_field.bspline(snr_normalized)  # [100, num_basis]
    for i in range(min(8, basis_vals.shape[1])):
        plt.plot(snr_values, basis_vals[:, i], label=f'Basis {i}')
    plt.title('B-spline Basis Functions')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Basis Value')
    plt.legend()
    plt.grid(True)

    # Plot CSI impact
    plt.subplot(2, 2, 4)
    for i, csi in enumerate(csi_samples):
        csi_energy = torch.norm(csi).item()
        plt.bar(i, csi_energy, label=f'CSI {i}')
    plt.title('CSI Energy Distribution')
    plt.xlabel('CSI Sample')
    plt.ylabel('L2 Norm')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def test_bspline_functionality():
    """Test B-spline basis functions."""
    print("Testing B-spline basis functions...")

    bspline = BSplineBasis(degree=3, num_knots=8)
    x = torch.linspace(-1, 1, 50)
    basis = bspline(x)

    print(f"B-spline test: input {x.shape} -> basis {basis.shape}")

    # Test partition of unity (basis functions should sum to 1)
    basis_sum = basis.sum(dim=1)
    max_deviation = (basis_sum - 1.0).abs().max()
    print(f"Partition of unity: max deviation = {max_deviation:.6f}")

    return max_deviation < 1e-4


def test_kan_field_enhanced():
    """Test enhanced KAN-field with B-splines."""
    print("Testing enhanced KAN-field...")

    kan_field = KANFieldEnhanced(d_csi=32, d_feat=128, num_knots=6)

    # Test data
    csi = torch.randn(4, 32)
    snr_norm = torch.linspace(-1, 1, 4)

    alpha, beta = kan_field(csi, snr_norm, T=100)

    print(f"KAN-field test: CSI {csi.shape} -> alpha {alpha.shape}, beta {beta.shape}")

    # Test CSI responsiveness
    csi_low = torch.full((1, 32), -1.0)  # Low quality channel
    csi_high = torch.full((1, 32), 1.0)  # High quality channel

    alpha_low, _ = kan_field(csi_low, torch.tensor([-1.0]))
    alpha_high, _ = kan_field(csi_high, torch.tensor([1.0]))

    responsiveness = (alpha_high - alpha_low).abs().mean()
    print(f"CSI responsiveness: {responsiveness:.4f}")

    return alpha.shape == (4, 100, 128)


def main():
    """Main function for testing and visualization."""
    parser = argparse.ArgumentParser(description="KAN-field testing and visualization")
    parser.add_argument("--test-bspline", action="store_true", help="Test B-spline basis functions")
    parser.add_argument("--test-kan", action="store_true", help="Test KAN-field functionality")
    parser.add_argument("--visualize-curves", action="store_true", help="Visualize modulation curves")
    parser.add_argument("--save-path", default="kan_field_visualization.png", help="Save path for visualizations")

    args = parser.parse_args()

    if args.test_bspline:
        success = test_bspline_functionality()
        print(f"B-spline test: {'PASSED' if success else 'FAILED'}")

    if args.test_kan:
        success = test_kan_field_enhanced()
        print(f"KAN-field test: {'PASSED' if success else 'FAILED'}")

    if args.visualize_curves:
        kan_field = KANFieldEnhanced(d_csi=32, d_feat=128, num_knots=8, spline_degree=3)
        csi_samples = torch.randn(3, 32)  # 3 different channel conditions
        visualize_bspline_curves(kan_field, csi_samples, save_path=args.save_path)

    if not any([args.test_bspline, args.test_kan, args.visualize_curves]):
        print("Available options: --test-bspline, --test-kan, --visualize-curves")


if __name__ == "__main__":
    main()


__all__ = ["KANLiteFiLM", "KANFieldEnhanced", "BSplineBasis"]
