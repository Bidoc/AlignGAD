"""
Evaluation metrics for anomaly detection.

Standard metrics: AUROC, AUPRC.
"""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC.
    
    Args:
        scores: (n,) per-node anomaly scores (higher = more anomalous)
        labels: (n,) binary labels (1 = anomaly, 0 = normal)
    
    Returns:
        dict with 'auroc' and 'auprc'
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Make sure dtypes are right
    scores = np.asarray(scores).astype(np.float64).flatten()
    labels = np.asarray(labels).astype(np.int64).flatten()
    
    # Edge case: only one class present
    if len(np.unique(labels)) < 2:
        return {'auroc': 0.5, 'auprc': float(labels.mean())}
    
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
    }


@torch.no_grad()
def evaluate_pipeline(
    pipeline,
    X: torch.Tensor,
    A: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate pipeline on a single target graph.
    
    Args:
        pipeline: trained MS-ZeroGAD pipeline
        X: (n, f) features
        A: (n, n) adjacency
        y: (n,) labels
        device: torch device
    
    Returns:
        metrics dict
    """
    from ..pipeline.aggregation import aggregate_scores
    
    if device is None:
        device = next(pipeline.parameters()).device
    
    pipeline.eval()
    
    X = X.to(device)
    A = A.to(device)
    
    scores_list, _, tracker, _ = pipeline(X, A, is_training=False)
    n = X.shape[0]
    
    final_scores = aggregate_scores(scores_list, tracker, n)
    
    return compute_metrics(final_scores.cpu(), y.cpu())


@torch.no_grad()
def evaluate_per_pass(
    pipeline,
    X: torch.Tensor,
    A: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate each pass individually + aggregated. Useful for ablation.
    
    Returns:
        dict like:
            {
                'pass1': {'auroc': ..., 'auprc': ...},
                'pass2': {...},
                'pass3': {...},
                'aggregated': {...}
            }
    """
    from ..pipeline.aggregation import aggregate_scores_with_breakdown
    
    if device is None:
        device = next(pipeline.parameters()).device
    
    pipeline.eval()
    
    X = X.to(device)
    A = A.to(device)
    
    scores_list, _, tracker, _ = pipeline(X, A, is_training=False)
    n = X.shape[0]
    
    breakdown = aggregate_scores_with_breakdown(scores_list, tracker, n)
    y_cpu = y.cpu()
    
    return {
        'pass1': compute_metrics(breakdown['pass1'].cpu(), y_cpu),
        'pass2': compute_metrics(breakdown['pass2'].cpu(), y_cpu),
        'pass3': compute_metrics(breakdown['pass3'].cpu(), y_cpu),
        'aggregated': compute_metrics(breakdown['final'].cpu(), y_cpu),
    }
