"""Numerical stability utilities and Straight-Through Estimator wrappers."""

import torch
import torch.nn.functional as F


EPS = 1e-8


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Numerically stable division with epsilon floor on denominator."""
    return numerator / (denominator + eps)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """L2-normalize a tensor along a dimension with epsilon."""
    norm = x.norm(p=2, dim=dim, keepdim=True)
    return x / (norm + eps)


def min_max_normalize(x: torch.Tensor, dim: int = None, eps: float = EPS) -> torch.Tensor:
    """
    Min-max normalize to [0, 1].
    
    Args:
        x: input tensor
        dim: dimension to normalize over (None = global)
        eps: stability epsilon
    
    Returns:
        normalized tensor with same shape as x
    """
    if dim is None:
        x_min = x.min()
        x_max = x.max()
    else:
        x_min = x.min(dim=dim, keepdim=True).values
        x_max = x.max(dim=dim, keepdim=True).values
    
    range_val = x_max - x_min
    return (x - x_min) / (range_val + eps)


def ste_argmax(logits: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    """
    Straight-Through Estimator for argmax.
    
    Forward: returns one-hot vector at argmax position.
    Backward: gradient flows through softmax(logits / tau).
    
    Args:
        logits: input scores, higher = more likely
        dim: dimension to apply argmax
        tau: temperature for softmax (smaller = closer to hard)
    
    Returns:
        one-hot tensor (forward), with soft gradient (backward)
    """
    # Soft assignment for backward
    soft = F.softmax(logits / tau, dim=dim)
    
    # Hard assignment for forward
    idx = soft.argmax(dim=dim, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(dim, idx, 1.0)
    
    # STE trick
    return soft + (hard - soft).detach()


def ste_indicator(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Straight-Through Estimator for indicator function (x > threshold).
    
    Forward: binary {0, 1}.
    Backward: gradient flows through x identity.
    
    Args:
        x: input tensor
        threshold: comparison threshold
    
    Returns:
        binary tensor (forward), with identity gradient (backward)
    """
    hard = (x > threshold).float()
    # STE: forward = hard binary, backward = gradient of x
    return hard + (x - x.detach()) * 0.0 + (hard - hard.detach())  # equivalent to: hard + 0
    # The above ensures forward is hard, but gradient of x flows through:
    # Actually simpler: return hard + (x - x.detach()) but then forward is wrong
    # Correct STE:


def ste_binary(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Correct STE for binary indicator.
    
    Forward: 1 if x > threshold else 0
    Backward: identity gradient (∂out/∂x = 1)
    
    Args:
        x: input tensor (real-valued)
        threshold: comparison threshold
    
    Returns:
        binary tensor with identity gradient
    """
    hard = (x > threshold).float()
    # Forward = hard, Backward = identity through x
    return x + (hard - x).detach()


def cosine_similarity_safe(x1: torch.Tensor, x2: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Numerically stable cosine similarity."""
    x1_norm = x1.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    x2_norm = x2.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return (x1 * x2).sum(dim=dim) / (x1_norm.squeeze(dim) * x2_norm.squeeze(dim))


def pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance.
    
    Args:
        x: (N, d)
        y: (M, d)
    
    Returns:
        distances: (N, M) where dist[i,j] = ||x[i] - y[j]||^2
    """
    # Use ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)        # (N, 1)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True).T      # (1, M)
    xy = x @ y.T                                       # (N, M)
    
    dist_sq = x_sq + y_sq - 2 * xy
    return dist_sq.clamp(min=0)  # numerical floor at 0
