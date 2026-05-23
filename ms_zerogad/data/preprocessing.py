"""Preprocessing utilities for graph datasets."""

from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch


def sparse_to_torch(A: sp.csr_matrix) -> torch.Tensor:
    """Convert scipy sparse to torch sparse tensor."""
    A = A.tocoo()
    indices = torch.from_numpy(np.vstack([A.row, A.col])).long()
    values = torch.from_numpy(A.data).float()
    shape = torch.Size(A.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def sparse_to_torch_dense(A: sp.csr_matrix) -> torch.Tensor:
    """Convert scipy sparse to torch dense tensor (only for small matrices)."""
    return torch.from_numpy(A.toarray()).float()


def feature_to_torch(X: sp.csr_matrix, dense: bool = True) -> torch.Tensor:
    """Convert feature matrix to torch tensor."""
    if dense:
        return torch.from_numpy(X.toarray()).float()
    return sparse_to_torch(X)


def normalize_adjacency(A: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Standard GCN adjacency normalization: D^(-1/2) (A + I) D^(-1/2).
    
    Args:
        A: (n, n) adjacency matrix (dense or sparse)
        add_self_loops: whether to add identity matrix
    
    Returns:
        Normalized adjacency
    """
    if A.is_sparse:
        A = A.to_dense()  # for simplicity in Phase 1; optimize later
    
    n = A.shape[0]
    
    if add_self_loops:
        A = A + torch.eye(n, device=A.device)
    
    # Compute D^(-1/2)
    degree = A.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # handle isolated nodes
    
    # D^(-1/2) A D^(-1/2)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    return A_norm


def feature_row_normalize(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L1 row-normalize feature matrix (each row sums to 1)."""
    row_sum = X.sum(dim=1, keepdim=True)
    return X / (row_sum + eps)


def standardize_features(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score standardize feature columns (mean 0, std 1)."""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    return (X - mean) / (std + eps)
