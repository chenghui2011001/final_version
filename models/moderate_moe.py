# -*- coding: utf-8 -*-
"""
ModerateMoE: A moderately complex, non-simplified Mixture-of-Experts layer.

Key features
- Token-level routing with Noisy Top-k (Top-1/Top-2)
- Capacity factor with overflow handling (Switch/Top2 gating style)
- Load balancing auxiliary loss and router z-loss
- Expert dropout and temperature for exploration
- Pluggable experts (default GLU-FFN)

Shape conventions
- Input: h [B, T, D]
- Output: y [B, T, D]
- Experts process token embeddings [N_sel, D] where N_sel ≤ capacity per expert

Notes
- This module focuses on a stronger routing core, not domain-specific experts.
- You can pass in domain experts if they accept [N, D] token batches.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUFeedForwardExpert(nn.Module):
    """Default token-level expert: PreNorm + GLU-FFN with dropout.

    Expects input as [N, D] (a packed set of tokens assigned to this expert).
    """
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_hidden or (4 * d_model)

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 2 * d_hidden)
        self.act = nn.GLU(dim=-1)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class NoisyTopKRouter(nn.Module):
    """Noisy Top-k router with capacity and auxiliary losses.

    - Computes logits per token → adds noise during training (NoisyTopK)
    - Selects top-k experts with per-token gates
    - Enforces per-expert capacity with deterministic positions
    - Produces dispatch and combine tensors for expert execution/merge

    Based on ideas from Switch Transformer and Top-2 Router (GShard/T5X).
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 2,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
        noisy_gating: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert k in (1, 2), "Only Top-1 or Top-2 routing is supported."
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.noisy_gating = noisy_gating
        self.temperature = temperature

        self.router = nn.Linear(d_model, n_experts, bias=True)
        # Learned noise scale per expert for NoisyTopK
        self.noise_scale = nn.Parameter(torch.zeros(n_experts))

    @staticmethod
    def _cv_squared(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        # Coefficient of variation squared: (std/mean)^2, robust to scale
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean.clamp_min(eps) ** 2 + eps)

    def forward(self, tokens: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            tokens: [N, D] flattened token embeddings (N=B*T)

        Returns:
            dispatch: dict with
              - "dispatch_mask": [N, E, C] one-hot to send tokens to experts
              - "combine_weights": [N, E, C] weights to merge expert outputs
              - "expert_capacity": scalar int, capacity per expert
              - "topk_indices": [N, k] expert indices for top-k
              - "gates": [N, k] normalized gate weights
            aux: dict with auxiliary losses (balance_loss, z_loss, importance, load)
        """
        device = tokens.device
        N, D = tokens.shape

        logits = self.router(tokens)  # [N, E]
        logits = logits / self.temperature

        if self.training and self.noisy_gating:
            noise_std = torch.exp(self.noise_scale).unsqueeze(0)  # [1, E]
            logits = logits + torch.randn_like(logits) * noise_std

        # z-loss for router logits (stabilization)
        z_loss = (logits ** 2).mean() * 1e-4

        # Softmax for importance statistics (not used for selection directly)
        probs = F.softmax(logits, dim=-1)  # [N, E]
        importance = probs.sum(dim=0)  # [E]

        # Top-k selection (deterministic)
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)  # [N, k]
        topk_gates = F.softmax(topk_vals, dim=-1)  # normalize within top-k

        # Compute capacity per expert
        # Total assignments is N*k, split across E; scaled by capacity_factor
        capacity = math.ceil(self.capacity_factor * (N * self.k) / self.n_experts)
        C = capacity

        # Build a [N, E] binary assignment for each of k
        # Also compute positional indices (location within expert) with cumsum
        dispatch_mask = tokens.new_zeros((N, self.n_experts, C))
        combine_weights = tokens.new_zeros((N, self.n_experts, C))

        load = tokens.new_zeros(self.n_experts)  # actual load (selected tokens)

        # For each expert, we assign tokens whose position < C
        # We do this by iterating k passes to respect Top-2 composition
        # Accumulate positions per expert
        positions = tokens.new_zeros(self.n_experts, dtype=torch.long)

        # To compute positions per expert efficiently, gather indices per expert
        # We process k=1 and optionally k=2 in sequence
        for pass_k in range(self.k):
            expert_idx = topk_idx[:, pass_k]  # [N]
            gate = topk_gates[:, pass_k]      # [N]

            # Sort tokens by expert to cumsum positions per expert
            # Build per-expert lists of token indices
            for e in range(self.n_experts):
                mask_e = (expert_idx == e)
                if not mask_e.any():
                    continue
                idx_e = torch.nonzero(mask_e, as_tuple=False).squeeze(-1)  # [Ne]

                # Assign positions for these tokens
                start = positions[e].item()
                end = start + idx_e.numel()
                # Allowed range within capacity
                allowed = max(0, C - start)

                if allowed <= 0:
                    # Expert overflow handling: drop or skip remaining tokens for this expert
                    continue

                # Tokens that fit in capacity window
                idx_fit = idx_e[:allowed]
                pos = torch.arange(start, start + idx_fit.numel(), device=device)

                # Write dispatch one-hot and combine weights
                dispatch_mask[idx_fit, e, pos] = 1.0
                combine_weights[idx_fit, e, pos] = gate[idx_fit]

                # Update position and load
                positions[e] = positions[e] + idx_fit.numel()
                load[e] = load[e] + idx_fit.numel()

        # Normalize combine weights per token (sum over [E, C])
        denom = combine_weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        combine_weights = combine_weights / denom

        # Auxiliary: balance loss (Switch Transformer style)
        # Encourage equal importance and equal load
        # Using coefficient of variation squared on both terms
        balance_importance = self._cv_squared(importance)
        balance_load = self._cv_squared(load)
        balance_loss = (balance_importance + balance_load) * 0.5

        aux = {
            'balance_loss': balance_loss,
            'z_loss': z_loss,
            'importance': importance.detach(),
            'load': load.detach(),
            'expert_capacity': torch.tensor(C, device=device),
        }

        out = {
            'dispatch_mask': dispatch_mask,      # [N, E, C]
            'combine_weights': combine_weights,  # [N, E, C]
            'expert_capacity': torch.tensor(C, device=device),
            'topk_indices': topk_idx,
            'gates': topk_gates,
        }
        return out, aux


class ModerateMoE(nn.Module):
    """Moderately complex token-level MoE with Noisy Top-k routing.

    This layer performs routing at token level and only computes experts
    for the tokens assigned to them (capacity-aware), avoiding the naive
    "compute-all-experts" approach.

    Args:
      d_model: token embedding size
      n_experts: number of experts
      k: top-k routing (1 or 2)
      capacity_factor: expert capacity multiplier
      expert_cls: constructor for expert module taking (d_model) -> nn.Module
      expert_kwargs: optional kwargs passed to expert constructor
      dropout_expert: probability of dropping selected expert assignment
      temperature: router temperature
      noisy_gating: whether to add noise during training
    """

    def __init__(
        self,
        d_model: int = 128,
        n_experts: int = 4,
        k: int = 2,
        capacity_factor: float = 1.25,
        expert_cls=GLUFeedForwardExpert,
        expert_kwargs: Optional[dict] = None,
        dropout_expert: float = 0.0,
        temperature: float = 1.0,
        noisy_gating: bool = True,
        router_side_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.dropout_expert = dropout_expert

        # Optional side features for routing (e.g., CSI or acoustic priors)
        self.router_side_dim = router_side_dim
        if router_side_dim > 0:
            self.side_proj = nn.Linear(d_model + router_side_dim, d_model)
        else:
            self.side_proj = None

        # Router operates on tokens [N, D] (after optional side projection)
        self.router = NoisyTopKRouter(
            d_model=d_model,
            n_experts=n_experts,
            k=k,
            capacity_factor=capacity_factor,
            noisy_gating=noisy_gating,
            temperature=temperature,
        )

        # Instantiate experts
        expert_kwargs = expert_kwargs or {}
        self.experts = nn.ModuleList([
            expert_cls(d_model, **expert_kwargs) for _ in range(n_experts)
        ])

        # Track utilization (EMA)
        self.register_buffer('expert_counts', torch.zeros(n_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def _update_utilization(self, load: torch.Tensor):
        # load: [E] number of tokens actually processed per expert
        momentum = 0.99
        self.expert_counts = momentum * self.expert_counts + (1 - momentum) * load
        self.total_tokens = momentum * self.total_tokens + (1 - momentum) * load.sum()

    def get_expert_utilization(self) -> torch.Tensor:
        if self.total_tokens.item() > 0:
            return self.expert_counts / (self.total_tokens + 1e-8)
        else:
            return torch.ones_like(self.expert_counts) / self.n_experts

    def forward(self, h: torch.Tensor, router_side: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, D = h.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        # Flatten to tokens [N, D]
        x = h.reshape(B * T, D)

        # Optionally fuse side information into router input
        if self.side_proj is not None and router_side is not None:
            assert router_side.shape[0] == B, "router_side must be [B, R]"
            assert router_side.shape[1] == self.router_side_dim, f"router_side_dim={self.router_side_dim}"
            side_tokens = router_side.unsqueeze(1).expand(B, T, self.router_side_dim).reshape(B * T, self.router_side_dim)
            x_router = torch.cat([x, side_tokens], dim=-1)
            x_router = self.side_proj(x_router)
        else:
            x_router = x

        # Route tokens
        routing, aux = self.router(x_router)  # dispatch/combine and losses
        dispatch = routing['dispatch_mask']     # [N, E, C]
        combine = routing['combine_weights']    # [N, E, C]
        C = routing['expert_capacity'].item()

        # Optionally apply expert dropout on combine weights (during training)
        if self.training and self.dropout_expert > 0:
            drop_mask = (torch.rand_like(combine) > self.dropout_expert).float()
            combine = combine * drop_mask
            denom = combine.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            combine = combine / denom

        # Dispatch: build expert inputs of shape [E, C, D]
        # einsum: [N, E, C]T x [N, D] -> [E, C, D]
        expert_inputs = torch.einsum('nec,nd->ecd', dispatch, x)

        # Apply experts independently per expert over [C, D] tokens
        expert_outputs: List[torch.Tensor] = []
        loads = []
        for e, expert in enumerate(self.experts):
            inp_e = expert_inputs[e]  # [C, D]
            # Some capacity rows may be zeros; expert should handle it fine
            out_e = expert(inp_e)
            expert_outputs.append(out_e)
            loads.append((inp_e.abs().sum(dim=-1) > 0).float().sum())

        expert_outputs = torch.stack(expert_outputs, dim=0)  # [E, C, D]
        loads = torch.stack(loads)  # [E]

        # Combine back to tokens: [N, E, C] x [E, C, D] -> [N, D]
        y_tokens = torch.einsum('nec,ecd->nd', combine, expert_outputs)

        # Reshape to [B, T, D]
        y = y_tokens.view(B, T, D)

        # Update utilization stats
        if self.training:
            with torch.no_grad():
                self._update_utilization(loads.detach())

        # Collect aux losses and metrics
        aux_losses = {
            'moe_balance_loss': aux['balance_loss'],
            'moe_router_z_loss': aux['z_loss'],
        }
        aux_metrics = {
            'expert_importance': aux['importance'],
            'expert_load': aux['load'],
            'expert_capacity': aux['expert_capacity'],
            'utilization': self.get_expert_utilization(),
        }
        aux_losses.update(aux_metrics)

        return y, aux_losses


if __name__ == "__main__":
    # Quick sanity check
    B, T, D = 2, 16, 128
    x = torch.randn(B, T, D)
    moe = ModerateMoE(d_model=D, n_experts=4, k=2, capacity_factor=1.25)
    y, aux = moe(x)
    print(y.shape, {k: (v.shape if torch.is_tensor(v) else v) for k, v in aux.items()})
