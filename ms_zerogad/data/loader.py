"""Loader for .mat graph anomaly detection datasets."""

import os
from typing import Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat


# Common variable names found in .mat files
ADJACENCY_KEYS = ['Network', 'A', 'adj', 'Adj', 'network', 'adjacency']
FEATURE_KEYS = ['Attributes', 'X', 'features', 'Features', 'attributes', 'attribute']
LABEL_KEYS = ['Label', 'gnd', 'label', 'labels', 'class', 'Class']


def _load_mat_file(path: str) -> Dict[str, Any]:
    """Load a .mat file, trying scipy first, falling back to mat73 for v7.3."""
    try:
        data = loadmat(path)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError(
                "File appears to be MATLAB v7.3 (HDF5). "
                "Install mat73: pip install mat73"
            )
        data = mat73.loadmat(path)
    return data


def _find_key(data: Dict[str, Any], candidates: list, kind: str) -> str:
    """Find first matching key in data."""
    for key in candidates:
        if key in data:
            return key
    available = [k for k in data.keys() if not k.startswith('__')]
    raise KeyError(
        f"Cannot find {kind} in .mat file. "
        f"Looked for {candidates}. Available: {available}"
    )


def load_graph_dataset(
    path: str,
    symmetrize: bool = True,
    remove_self_loops: bool = True,
    binarize: bool = True,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Load a graph anomaly detection dataset from .mat file.
    
    Args:
        path: path to .mat file
        symmetrize: ensure adjacency is symmetric (for undirected graphs)
        remove_self_loops: remove diagonal entries
        binarize: convert adjacency to binary {0, 1}
    
    Returns:
        A: (n, n) sparse adjacency matrix in CSR format, float32
        X: (n, f) feature matrix in CSR format, float32
        y: (n,) anomaly labels, int64 (0 = normal, 1 = anomaly)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    data = _load_mat_file(path)
    
    # Find required arrays
    a_key = _find_key(data, ADJACENCY_KEYS, "adjacency matrix")
    x_key = _find_key(data, FEATURE_KEYS, "feature matrix")
    y_key = _find_key(data, LABEL_KEYS, "label vector")
    
    A = data[a_key]
    X = data[x_key]
    y = data[y_key]
    
    # Standardize adjacency
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A = A.tocsr().astype(np.float32)
    
    # Standardize features
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr().astype(np.float32)
    
    # Standardize labels
    y = np.asarray(y).flatten().astype(np.int64)
    
    # Sanity checks
    n = A.shape[0]
    assert A.shape == (n, n), f"Adjacency is not square: {A.shape}"
    assert X.shape[0] == n, f"Feature has {X.shape[0]} rows, expected {n}"
    assert y.shape == (n,), f"Labels shape {y.shape}, expected ({n},)"
    
    # Postprocessing
    if symmetrize:
        A = A.maximum(A.T)
    
    if remove_self_loops:
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()
        A.eliminate_zeros()
    
    if binarize:
        A.data = np.ones_like(A.data, dtype=np.float32)
    
    return A, X, y


def dataset_info(A: sp.csr_matrix, X: sp.csr_matrix, y: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics of a loaded dataset."""
    n = A.shape[0]
    num_edges = A.nnz // 2  # undirected
    num_anomalies = int(y.sum())
    
    return {
        'num_nodes': n,
        'num_edges': num_edges,
        'num_features': X.shape[1],
        'feature_density': X.nnz / (X.shape[0] * X.shape[1]),
        'num_anomalies': num_anomalies,
        'anomaly_ratio': num_anomalies / n,
        'avg_degree': 2 * num_edges / n,
    }


def print_dataset_info(name: str, info: Dict[str, Any]):
    """Pretty print dataset info."""
    print(f"\n=== {name} ===")
    print(f"  Nodes:           {info['num_nodes']:,}")
    print(f"  Edges:           {info['num_edges']:,}")
    print(f"  Features:        {info['num_features']:,}")
    print(f"  Feature density: {info['feature_density']:.4f}")
    print(f"  Anomalies:       {info['num_anomalies']:,} ({info['anomaly_ratio']*100:.2f}%)")
    print(f"  Avg degree:      {info['avg_degree']:.2f}")
