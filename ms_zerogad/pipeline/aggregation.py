"""
Score aggregation across multi-pass results.

Strategy: MAX of min-max normalized scores per pass.
- Min-max normalize each pass's scores to [0, 1]
- Map super-cluster scores back to original nodes via membership tracker
- Take per-node MAX across all passes
"""

from typing import List

import torch

from ..utils.stable_ops import min_max_normalize, EPS
from ..utils.tracking import MembershipTracker


def aggregate_scores(
    scores_list: List[torch.Tensor],
    tracker: MembershipTracker,
    n_original: int,
) -> torch.Tensor:
    """
    Aggregate per-pass scores into per-original-node final scores.
    
    Args:
        scores_list: list of 3 tensors
            [scores_1 (n,), scores_2 (m_pass1,), scores_3 (m_pass2,)]
        tracker: MembershipTracker with 2 levels of history
            - tracker.get_membership(1): pass-1 super-clusters → original nodes
            - tracker.get_membership(2): pass-2 super-clusters → original nodes
        n_original: number of original nodes
    
    Returns:
        final_scores: (n_original,) per-node aggregated anomaly scores
    """
    assert len(scores_list) >= 1, "Need at least one pass"
    
    device = scores_list[0].device
    
    # Normalize each pass's scores to [0, 1]
    normalized_scores = [min_max_normalize(s) for s in scores_list]
    
    # Pass 1: directly per-original-node
    final = normalized_scores[0].clone()  # (n_original,)
    
    # Pass 2: map super-cluster scores back to original nodes
    if len(scores_list) >= 2:
        node_to_cluster_p1 = tracker.get_node_to_cluster_at_level(1).to(device)  # (n_original,)
        scores_p2_per_node = normalized_scores[1][node_to_cluster_p1]  # (n_original,)
        final = torch.maximum(final, scores_p2_per_node)
    
    # Pass 3: map pass-3 super-cluster scores back to original nodes
    if len(scores_list) >= 3:
        node_to_cluster_p2 = tracker.get_node_to_cluster_at_level(2).to(device)  # (n_original,)
        scores_p3_per_node = normalized_scores[2][node_to_cluster_p2]  # (n_original,)
        final = torch.maximum(final, scores_p3_per_node)
    
    return final


def aggregate_scores_with_breakdown(
    scores_list: List[torch.Tensor],
    tracker: MembershipTracker,
    n_original: int,
) -> dict:
    """
    Same as aggregate_scores but also returns per-pass scores mapped to original nodes.
    Useful for debugging and analysis.
    
    Returns:
        dict with keys:
            'final': (n_original,) MAX-aggregated scores
            'pass1': (n_original,) pass 1 normalized scores
            'pass2': (n_original,) pass 2 scores mapped to original
            'pass3': (n_original,) pass 3 scores mapped to original
    """
    device = scores_list[0].device
    
    normalized_scores = [min_max_normalize(s) for s in scores_list]
    
    result = {
        'pass1': normalized_scores[0].clone(),
    }
    
    if len(scores_list) >= 2:
        node_to_cluster_p1 = tracker.get_node_to_cluster_at_level(1).to(device)
        result['pass2'] = normalized_scores[1][node_to_cluster_p1]
    else:
        result['pass2'] = torch.zeros(n_original, device=device)
    
    if len(scores_list) >= 3:
        node_to_cluster_p2 = tracker.get_node_to_cluster_at_level(2).to(device)
        result['pass3'] = normalized_scores[2][node_to_cluster_p2]
    else:
        result['pass3'] = torch.zeros(n_original, device=device)
    
    final = result['pass1']
    if len(scores_list) >= 2:
        final = torch.maximum(final, result['pass2'])
    if len(scores_list) >= 3:
        final = torch.maximum(final, result['pass3'])
    result['final'] = final
    
    return result
