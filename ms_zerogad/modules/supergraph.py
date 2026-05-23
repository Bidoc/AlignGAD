"""
Module 3: Super-graph Construction.

Given an assignment matrix P, build:
1. Compressed feature: X_super[j] = weighted mean of nodes in cluster j
2. Super-adjacency: 
   - Off-diagonal binary: A_super[i,j] = 1 if any edge between clusters i,j
   - Self-loop: A_super[i,i] = 1 if any internal edge in cluster i (Quy ước 2)
   - All from ORIGINAL graph (Cách α — no accumulation across passes)
"""

from typing import Tuple, Dict, List, Optional

import torch

from ..utils.stable_ops import EPS, ste_binary


def build_super_graph(
    X: torch.Tensor,
    A: torch.Tensor,
    P: torch.Tensor,
    membership_for_self_loop: Dict[int, List[int]],
    A_original: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build super-graph from cluster assignment.
    
    Args:
        X: (N, d) features at current scale (input to clustering)
        A: (N, N) adjacency at current scale (input to clustering)
        P: (N, K) assignment matrix (one-hot in forward, soft gradient backward)
        membership_for_self_loop: dict mapping super_cluster_idx -> list of ORIGINAL node indices
                                   (for computing self-loops from A_original)
        A_original: (n_orig, n_orig) ORIGINAL graph adjacency (for self-loop computation)
    
    Returns:
        X_super: (K, d) compressed feature (differentiable)
        A_super_hat: (K, K) normalized super-adjacency = A_super_raw + I
                     (forward: binary off-diag + binary self-loop + I)
                     (backward: gradient through soft P @ A @ P)
    """
    K = P.shape[1]
    
    # === Step 1: Feature compression (weighted mean) ===
    # X_super[j] = sum_i P[i,j] X[i] / sum_i P[i,j]
    cluster_sizes = P.sum(dim=0, keepdim=True).T + EPS    # (K, 1)
    X_super = (P.T @ X) / cluster_sizes                    # (K, d)
    
    # === Step 2: Off-diagonal — binary indicator with STE ===
    # Compute "real" weighted adjacency using soft P
    # A_real[i,j] = sum_{u,v} P[u,i] A[u,v] P[v,j] = (P^T A P)[i,j]
    A_real = P.T @ A @ P                                   # (K, K) — differentiable
    
    # Zero out diagonal of A_real (we'll fill it from original graph below)
    diag_mask = torch.eye(K, device=A.device, dtype=torch.bool)
    A_real_offdiag = A_real.masked_fill(diag_mask, 0.0)
    
    # Binary indicator with STE: forward = (>0), backward = identity
    A_offdiag_binary = ste_binary(A_real_offdiag, threshold=0.0)
    # Note: this still has 0 on diagonal (we set it to 0 above)
    
    # === Step 3: Self-loop — binary, from ORIGINAL graph (Cách α) ===
    # For each super-cluster j, check if any pair of ORIGINAL nodes 
    # in membership_for_self_loop[j] has an edge in A_original.
    # 
    # This is a discrete topology check; no gradient flow through this part.
    self_loop = torch.zeros(K, device=A.device)
    
    for j in range(K):
        original_nodes = membership_for_self_loop.get(j, [])
        if len(original_nodes) >= 2:
            # Index a sub-block of A_original
            idx = torch.tensor(original_nodes, device=A_original.device, dtype=torch.long)
            sub_A = A_original[idx][:, idx]
            # Sum upper triangular (exclude diagonal which should be 0 anyway)
            num_internal_edges = torch.triu(sub_A, diagonal=1).sum()
            if num_internal_edges.item() > 0:
                self_loop[j] = 1.0
    
    # === Step 4: Combine ===
    # A_super_raw = off-diag binary + diag(self_loop)
    A_super_raw = A_offdiag_binary + torch.diag(self_loop)
    
    # === Step 5: Standard GCN normalization (add I) ===
    # Note: We do NOT compute D^(-1/2) here because the scoring module 
    # will normalize when needed. We return the raw + I version.
    # But for consistency with how the scoring module expects, we add I and let 
    # downstream normalize.
    A_hat = A_super_raw + torch.eye(K, device=A.device)
    
    return X_super, A_hat


def normalize_super_adjacency(A_hat: torch.Tensor) -> torch.Tensor:
    """
    Apply D^(-1/2) A_hat D^(-1/2) normalization.
    
    Args:
        A_hat: (K, K) super-adjacency (already includes self-loops)
    
    Returns:
        A_norm: (K, K) normalized
    """
    degree = A_hat.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt
