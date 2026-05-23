"""
Module 2: RFF-based Differentiable Hierarchical Clustering.

Pipeline per pass:
    Input → Light Smoothing → T-SVD → RFF → Spectral Embedding → STE K-means → Soft Assignment

Inspired by SASE (Scalable Adaptive Spectral Embedding) but:
- Adds Straight-Through Estimator for gradient flow through hard assignment
- Used as intermediate clustering for super-graph construction
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.stable_ops import EPS, pairwise_squared_distance, ste_argmax


class RFFClustering(nn.Module):
    """
    Hierarchical clustering using:
    1. Light SGC smoothing (k=1 or 2)
    2. Truncated SVD dimension reduction
    3. Random Fourier Features projection
    4. Implicit spectral embedding (no n×n similarity matrix)
    5. Differentiable K-means via Straight-Through Estimator
    
    Has NO trainable parameters (deterministic given fixed RNG seed for omega).
    """
    
    def __init__(
        self,
        k_smoothing: int = 1,
        sigma: float = 1.0,
        D_rff: int = 50,
        d_svd: int = 32,
        tau: float = 0.5,
        kmeans_max_iter: int = 20,
        seed: int = 42,
    ):
        """
        Args:
            k_smoothing: smoothing order (number of times to apply A_norm)
            sigma: RFF Gaussian kernel bandwidth
            D_rff: number of random Fourier features (final dim = 2*D_rff)
            d_svd: T-SVD reduced dimension
            tau: STE temperature for soft assignment
            kmeans_max_iter: K-means iterations
            seed: random seed for RFF omega sampling
        """
        super().__init__()
        self.k_smoothing = k_smoothing
        self.sigma = sigma
        self.D_rff = D_rff
        self.d_svd = d_svd
        self.tau = tau
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        
        # omega will be sampled lazily based on input dimension
        # IMPORTANT: For Phase 1, we sample once and reuse across passes (cùng set ω)
        # Stored as a buffer (not parameter, no gradient)
        self._omega = None
        self._omega_input_dim = None
    
    def _get_omega(self, input_dim: int, device: torch.device) -> torch.Tensor:
        """Lazily sample omega for RFF, cache for reuse."""
        if self._omega is not None and self._omega_input_dim == input_dim:
            return self._omega.to(device)
        
        # Sample omega ~ N(0, (1/sigma)^2 I)
        # In RFF, omega scale relates to kernel bandwidth: omega ~ N(0, 1/sigma^2 I)
        # so omega^T x has appropriate variance for kernel.
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        omega = torch.randn(input_dim, self.D_rff, generator=gen) / self.sigma
        
        self._omega = omega
        self._omega_input_dim = input_dim
        return omega.to(device)
    
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """
        Forward pass: cluster nodes into num_clusters using RFF + STE K-means.
        
        Args:
            X: (N, d) input features
            A: (N, N) adjacency (dense, binary, no self-loops)
            num_clusters: target number of clusters
        
        Returns:
            P: (N, num_clusters) assignment matrix
                Forward: one-hot (hard)
                Backward: gradient flows through soft softmax
        """
        N = X.shape[0]
        device = X.device
        
        # Step 1: Light Smoothing
        X_smoothed = self._smooth(X, A)
        
        # Step 2: T-SVD reduction
        Z = self._truncated_svd(X_smoothed, k=min(self.d_svd, X_smoothed.shape[1], N))
        
        # Step 3: RFF projection
        Z_rff = self._rff_project(Z)
        
        # Step 4: Spectral embedding (without building n×n matrix)
        U = self._spectral_embedding(Z_rff, k=min(self.d_svd, Z_rff.shape[1], N))
        
        # Step 5: STE K-means
        P = self._differentiable_kmeans(U, num_clusters)
        
        return P
    
    def _smooth(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Light SGC smoothing: X^(k) = (D̂^(-1/2) Â D̂^(-1/2))^k X.
        
        If k_smoothing == 0, returns X unchanged.
        """
        if self.k_smoothing == 0:
            return X
        
        N = X.shape[0]
        device = X.device
        
        # Add self-loops and normalize
        A_hat = A + torch.eye(N, device=device)
        degree = A_hat.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # Apply k-th power
        X_smoothed = X
        for _ in range(self.k_smoothing):
            X_smoothed = A_norm @ X_smoothed
        
        return X_smoothed
    
    def _truncated_svd(self, X: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute top-k singular vectors via randomized SVD.
        
        Returns: U @ diag(S), shape (N, k)
        """
        # Use slightly larger q for better accuracy
        q = min(k + 5, min(X.shape))
        U, S, V = torch.svd_lowrank(X, q=q)
        # Project: X_reduced = U_top * S_top
        return U[:, :k] * S[:k].unsqueeze(0)
    
    def _rff_project(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Random Fourier Features projection.
        
        phi(z) = (1/sqrt(D)) [cos(omega^T z), sin(omega^T z)]
        
        Args:
            Z: (N, d) input
        
        Returns:
            Z_rff: (N, 2*D_rff)
        """
        d = Z.shape[1]
        omega = self._get_omega(d, Z.device)              # (d, D_rff)
        
        projection = Z @ omega                              # (N, D_rff)
        cos_part = torch.cos(projection)
        sin_part = torch.sin(projection)
        
        Z_rff = torch.cat([cos_part, sin_part], dim=-1) / (self.D_rff ** 0.5)
        # shape (N, 2*D_rff)
        
        return Z_rff
    
    def _spectral_embedding(self, Z_rff: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute spectral embedding without building the n×n similarity matrix.
        
        Trick: 
            - Implicit similarity W ≈ Z_rff @ Z_rff.T
            - Degree D = diag(W @ 1) = diag(Z_rff @ (Z_rff.T @ 1))
            - Normalized: Ẑ = D^(-1/2) Z_rff
            - Top eigenvectors of D^(-1/2) W D^(-1/2) = top left singular vectors of Ẑ
        
        Args:
            Z_rff: (N, 2*D_rff)
            k: number of top components
        
        Returns:
            U: (N, k) L2-normalized spectral embedding
        """
        N = Z_rff.shape[0]
        
        # Compute degree without forming W
        ones = torch.ones(N, device=Z_rff.device)
        z_sum = Z_rff.T @ ones                              # (2*D_rff,)
        degree = Z_rff @ z_sum                              # (N,)
        degree = degree.clamp(min=EPS)
        
        d_inv_sqrt = degree.pow(-0.5)                       # (N,)
        Z_hat = d_inv_sqrt.unsqueeze(1) * Z_rff             # (N, 2*D_rff)
        
        # Top-k singular vectors of Z_hat
        q = min(k + 5, min(Z_hat.shape))
        U, S, V = torch.svd_lowrank(Z_hat, q=q)
        U_top = U[:, :k]                                    # (N, k)
        
        # L2 normalize rows
        norms = U_top.norm(dim=1, keepdim=True).clamp(min=EPS)
        U_norm = U_top / norms
        
        return U_norm
    
    def _differentiable_kmeans(self, X: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        K-means with K-means++ init and Straight-Through Estimator.
        
        Args:
            X: (N, d) embeddings
            num_clusters: target K
        
        Returns:
            P: (N, num_clusters) assignment
                Forward: one-hot
                Backward: soft softmax(-distance / tau) gradient
        """
        N, d = X.shape
        K = num_clusters
        
        # K-means++ initialization (no gradient needed for centroids)
        with torch.no_grad():
            centroids = self._kmeans_pp_init(X.detach(), K)
            
            # K-means iterations (forward only)
            for _ in range(self.kmeans_max_iter):
                # Compute distances
                dists = pairwise_squared_distance(X.detach(), centroids)  # (N, K)
                # Hard assignment
                c_hard = dists.argmin(dim=-1)              # (N,)
                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(K, device=X.device)
                new_centroids.index_add_(0, c_hard, X.detach())
                counts.index_add_(0, c_hard, torch.ones(N, device=X.device))
                # Avoid empty cluster: keep old centroid
                mask = counts > 0
                new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
                new_centroids[~mask] = centroids[~mask]
                
                # Check convergence (small change)
                if torch.allclose(centroids, new_centroids, atol=1e-6):
                    centroids = new_centroids
                    break
                centroids = new_centroids
        
        # Final assignment with STE (gradient flows through here)
        # X is differentiable, centroids are detached
        dists = pairwise_squared_distance(X, centroids)    # (N, K)
        
        # ste_argmax: forward = one-hot at argmin, backward = softmax gradient
        # Note: argmin of distance = argmax of -distance
        P = ste_argmax(-dists, dim=-1, tau=self.tau)
        
        return P
    
    def _kmeans_pp_init(self, X: torch.Tensor, K: int) -> torch.Tensor:
        """
        K-means++ initialization.
        
        Args:
            X: (N, d) data
            K: number of centroids
        
        Returns:
            centroids: (K, d)
        """
        N = X.shape[0]
        device = X.device
        
        # First centroid: random
        idx = torch.randint(0, N, (1,), device=device).item()
        centroids = [X[idx]]
        chosen_indices = {idx}
        
        for _ in range(K - 1):
            # Distance to nearest existing centroid
            cents_tensor = torch.stack(centroids)           # (k, d)
            dists = pairwise_squared_distance(X, cents_tensor)  # (N, k)
            min_dists = dists.min(dim=-1).values             # (N,)
            
            total = min_dists.sum().item()
            
            if total < EPS:
                # Degenerate case: all points are identical or already covered
                # Fallback: pick a random point not already chosen
                available = [i for i in range(N) if i not in chosen_indices]
                if not available:
                    # All points already chosen; pick any
                    idx = torch.randint(0, N, (1,), device=device).item()
                else:
                    idx = available[torch.randint(0, len(available), (1,)).item()]
            else:
                # Probability proportional to squared distance
                probs = min_dists / total
                # Ensure no negative due to numerical issues
                probs = probs.clamp(min=0)
                if probs.sum().item() < EPS:
                    idx = torch.randint(0, N, (1,), device=device).item()
                else:
                    idx = torch.multinomial(probs, 1).item()
            
            centroids.append(X[idx])
            chosen_indices.add(idx)
        
        return torch.stack(centroids)                       # (K, d)
