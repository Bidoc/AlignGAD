"""
Module 1: Global Information Unification with Asymmetric Spectral Normalization.

Two stages:
1. Feature Dimension Alignment via SVD (project to common d_prime)
2. Asymmetric Spectral Normalization (3 frequency bands with different alphas)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..utils.stable_ops import EPS


class GlobalUnification(nn.Module):
    """
    Module 1: Global Information Unification.
    
    Implements:
    - Step 1: SVD-based dimension alignment
    - Step 2: Asymmetric Spectral Normalization across frequency bands
    
    Note: This module has NO trainable parameters in Phase 1.
    All operations are deterministic given fixed hyperparameters.
    """
    
    def __init__(
        self,
        d_prime: int = 8,
        band_low: float = 0.5,
        band_high: float = 1.5,
        band_low_percentile=0.33,        # ← new
        band_high_percentile=0.67,       # ← new
        adaptive_bands=False,   
        alpha_low: float = 1.0,
        alpha_mid: float = 0.95,
        alpha_high: float = 0.9,
    ):
        """
        Args:
            d_prime: target dimension after SVD alignment
            band_low: boundary between low and mid bands (in normalized Laplacian eigenvalues)
            band_high: boundary between mid and high bands
            alpha_low/mid/high: normalization strength for each band [0, 1]
                                1.0 = full normalization, 0.0 = no normalization
        """
        super().__init__()
        self.d_prime = d_prime
        self.band_low = band_low
        self.band_high = band_high
        self.alpha_low = alpha_low
        self.alpha_mid = alpha_mid
        self.alpha_high = alpha_high
        self.adaptive_bands = adaptive_bands              # ← phải có dòng này
        self.band_low_percentile = band_low_percentile    # ← phải có dòng này
        self.band_high_percentile = band_high_percentile  
    
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Apply global unification to (X, A).
        
        Args:
            X: (n, f) feature matrix
            A: (n, n) adjacency matrix (dense)
            return_components: if True, also return intermediate values for inspection
        
        Returns:
            X_unified: (n, d_prime) unified features
            (optional) components dict with intermediate tensors
        """
        # Step 1: Feature Dimension Alignment via SVD
        X_aligned = self._dimension_alignment(X)  # (n, d_prime)
        
        # Step 2: Asymmetric Spectral Normalization
        X_unified, components = self._spectral_normalization(X_aligned, A)
        
        if return_components:
            return X_unified, components
        return X_unified
    
    def _dimension_alignment(self, X: torch.Tensor) -> torch.Tensor:
        """
        SVD-based dimension alignment to d_prime.
        
        X = U S V^T
        Keep top-d_prime components: X_aligned = U[:, :d'] @ S[:d'] @ Q^T
        
        For Phase 1, set Q = I (identity).
        
        Args:
            X: (n, f) feature matrix
        
        Returns:
            X_aligned: (n, d_prime)
        """
        n, f = X.shape
        d_prime = min(self.d_prime, min(n, f))
        
        # Use truncated SVD for efficiency
        # torch.svd_lowrank gives U (n,k), S (k,), V (f,k) where X ≈ U @ diag(S) @ V.T
        # k slightly larger than d_prime for numerical accuracy
        q = min(d_prime + 5, min(n, f))
        U, S, V = torch.svd_lowrank(X, q=q)
        
        # Truncate to d_prime
        U_d = U[:, :d_prime]                                # (n, d_prime)
        S_d = S[:d_prime]                                   # (d_prime,)
        
        # X_aligned = U @ diag(S) (with Q = I for Phase 1)
        X_aligned = U_d * S_d.unsqueeze(0)                  # (n, d_prime)
        
        return X_aligned
    
    def _spectral_normalization(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Asymmetric spectral normalization in graph Fourier domain.
        
        Args:
            X: (n, d_prime) aligned features
            A: (n, n) adjacency
        
        Returns:
            X_unified: (n, d_prime)
            components: dict with intermediate tensors
        """
        n = X.shape[0]
        device = X.device
        
        # Step 2a: Compute normalized Laplacian
        # L = I - D^(-1/2) A D^(-1/2)
        degree = A.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        L = torch.eye(n, device=device) - A_norm
        
        # Symmetrize numerically (small floating-point asymmetries can cause eigh to fail)
        L = 0.5 * (L + L.T)
        
        # Step 2b: Eigendecomposition
        eigenvalues, U_L = torch.linalg.eigh(L)             # ascending order
        # eigenvalues: (n,), U_L: (n, n)
        
        # Step 2c: Project to spectral domain
        F = U_L.T @ X                                        # (n, d_prime)
        
        # Step 2d: Compute mean and std across all frequencies
        mu = F.mean(dim=0, keepdim=True)                    # (1, d_prime)
        sigma = F.std(dim=0, keepdim=True) + EPS            # (1, d_prime)
        F_normalized = (F - mu) / sigma                     # fully normalized
        
        # Step 2e: Identify bands
        # low_mask = eigenvalues < self.band_low
        # mid_mask = (eigenvalues >= self.band_low) & (eigenvalues < self.band_high)
        # high_mask = eigenvalues >= self.band_high
        if self.adaptive_bands:
            band_low_val = torch.quantile(eigenvalues, self.band_low_percentile)
            band_high_val = torch.quantile(eigenvalues, self.band_high_percentile)
        else:
            band_low_val = self.band_low
            band_high_val = self.band_high
        
        low_mask = eigenvalues < band_low_val
        mid_mask = (eigenvalues >= band_low_val) & (eigenvalues < band_high_val)
        high_mask = eigenvalues >= band_high_val    
                
        # Step 2f: Apply asymmetric normalization
        # For each row k of F (corresponding to eigenvalue lambda_k):
        #   F_hat[k] = alpha_band(k) * F_normalized[k] + (1 - alpha_band(k)) * F[k]
        
        # Build a per-frequency alpha vector
        alpha_vec = torch.empty(n, device=device)
        alpha_vec[low_mask] = self.alpha_low
        alpha_vec[mid_mask] = self.alpha_mid
        alpha_vec[high_mask] = self.alpha_high
        alpha_vec = alpha_vec.unsqueeze(1)                  # (n, 1)
        
        F_hat = alpha_vec * F_normalized + (1 - alpha_vec) * F
        
        # Step 2g: Inverse Graph Fourier Transform
        X_unified = U_L @ F_hat                              # (n, d_prime)
        
        components = {
            'eigenvalues': eigenvalues,
            'eigenvectors': U_L,
            'F_before': F,
            'F_after': F_hat,
            'low_mask': low_mask,
            'mid_mask': mid_mask,
            'high_mask': high_mask,
            'num_low': int(low_mask.sum().item()),
            'num_mid': int(mid_mask.sum().item()),
            'num_high': int(high_mask.sum().item()),
            'band_low_used': band_low_val,    # ← thêm
            'band_high_used': band_high_val,
        }
        
        return X_unified, components
    
    def get_band_info(self, eigenvalues: torch.Tensor) -> dict:
        """Helper to inspect band statistics for a given eigenvalue spectrum."""
        low = (eigenvalues < self.band_low).sum().item()
        mid = ((eigenvalues >= self.band_low) & (eigenvalues < self.band_high)).sum().item()
        high = (eigenvalues >= self.band_high).sum().item()
        n = eigenvalues.numel()
        
        return {
            'low': low,
            'mid': mid,
            'high': high,
            'low_pct': low / n * 100,
            'mid_pct': mid / n * 100,
            'high_pct': high / n * 100,
            'total': n,
        }
