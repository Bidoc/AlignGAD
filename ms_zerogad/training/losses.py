"""
Loss functions for MS-ZeroGAD.

Per-pass loss = E[(1 - cos(X_rec, X_gen))^alpha] + beta * Var(Z_latent)
Total loss = sum over passes (with weights, equal in Phase 1)
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..utils.stable_ops import cosine_similarity_safe


def per_pass_loss(
    X_rec: torch.Tensor,
    X_gen: torch.Tensor,
    Z: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Compute Zero-GAD loss for one pass.
    
    L = E[(1 - cos(X_rec, X_gen))^alpha] + beta * Var(Z)
    
    Args:
        X_rec: (N, d) input feature
        X_gen: (N, d) generated feature
        Z: (N, d_latent) latent representation
        alpha: sparsity-aware exponent (>1 emphasizes large discrepancies)
        beta: neutralization weight
    
    Returns:
        loss: scalar
    """
    # Reconstruction loss: E[(1 - cos)^alpha]
    cos_sim = cosine_similarity_safe(X_rec, X_gen, dim=-1)  # (N,)
    discrepancy = (1.0 - cos_sim).clamp(min=0)              # (N,)
    L_recon = (discrepancy ** alpha).mean()
    
    # Neutralization loss: variance of Z (encourage flat latent for normal nodes)
    # Use per-feature variance, then mean
    Z_mean = Z.mean(dim=0, keepdim=True)
    L_neu = ((Z - Z_mean) ** 2).mean()
    
    return L_recon + beta * L_neu


def multipass_total_loss(
    features_list: List[Tuple[torch.Tensor, torch.Tensor]],
    Z_list: List[torch.Tensor],
    alpha: float = 2.0,
    beta: float = 0.5,
    weights: List[float] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Total loss across all passes.
    
    Args:
        features_list: list of (X_rec, X_gen) tuples per pass
        Z_list: list of latent tensors per pass
        alpha: per-pass alpha
        beta: per-pass beta
        weights: per-pass loss weights (defaults to all 1.0)
    
    Returns:
        total_loss: scalar
        per_pass_losses: list of scalar losses (for logging)
    """
    if weights is None:
        weights = [1.0] * len(features_list)
    
    assert len(weights) == len(features_list) == len(Z_list)
    
    per_pass_losses = []
    total = 0.0
    
    for w, (X_rec, X_gen), Z in zip(weights, features_list, Z_list):
        L = per_pass_loss(X_rec, X_gen, Z, alpha=alpha, beta=beta)
        per_pass_losses.append(L)
        total = total + w * L
    
    return total, per_pass_losses
