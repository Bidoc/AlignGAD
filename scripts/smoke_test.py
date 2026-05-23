"""
Smoke test for MS-ZeroGAD pipeline.

Creates a synthetic small graph and runs the full pipeline (forward + loss + backward).
Just verifies that nothing crashes — does NOT verify correctness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from ms_zerogad.pipeline.multipass import MultiPassPipeline
from ms_zerogad.pipeline.aggregation import aggregate_scores, aggregate_scores_with_breakdown
from ms_zerogad.training.losses import multipass_total_loss
from ms_zerogad.evaluation.metrics import compute_metrics


def make_synthetic_graph(n=200, f=50, p_edge=0.05, anomaly_ratio=0.05, seed=42):
    """Create a synthetic graph with planted anomalies."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Random adjacency
    A = (torch.rand(n, n) < p_edge).float()
    A = torch.triu(A, diagonal=1)
    A = A + A.T  # symmetric
    
    # Random feature
    X = torch.randn(n, f)
    
    # Plant anomalies: nodes with shuffled feature
    num_anomalies = int(n * anomaly_ratio)
    anomaly_idx = torch.randperm(n)[:num_anomalies]
    y = torch.zeros(n, dtype=torch.long)
    y[anomaly_idx] = 1
    
    # Make anomalies' features very different
    X[anomaly_idx] = X[anomaly_idx] * 5 + 10
    
    return X, A, y


def main():
    print("=" * 60)
    print("MS-ZeroGAD Smoke Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create synthetic graph
    X, A, y = make_synthetic_graph(n=200, f=50)
    print(f"\nSynthetic graph: n={X.shape[0]}, f={X.shape[1]}, edges={int(A.sum().item()) // 2}")
    print(f"Anomalies: {int(y.sum().item())} ({y.float().mean().item()*100:.1f}%)")
    
    # Build pipeline
    pipeline = MultiPassPipeline(
        d_prime=8,
        d_hidden=32,
        d_latent=16,
        D_rff=20,    # smaller for speed
        d_svd=8,
        kmeans_max_iter=10,
    ).to(device)
    
    print(f"\nPipeline created. Trainable params: {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,}")
    
    X = X.to(device)
    A = A.to(device)
    y = y.to(device)
    
    # === Forward pass (training mode) ===
    print("\n--- Forward (training mode) ---")
    pipeline.train()
    scores_list, features_list, tracker, extras = pipeline(X, A, is_training=True)
    
    print(f"  Pass 1 scores shape: {scores_list[0].shape}")
    print(f"  Pass 2 scores shape: {scores_list[1].shape} (m_pass1={extras['m_pass1']})")
    print(f"  Pass 3 scores shape: {scores_list[2].shape} (m_pass2={extras['m_pass2']})")
    print(f"  Tracker has {len(tracker)} levels")
    
    # === Loss computation ===
    print("\n--- Loss ---")
    Z_list = [extras['Z_1'], extras['Z_2'], extras['Z_3']]
    total_loss, per_pass_losses = multipass_total_loss(features_list, Z_list)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Pass 1: {per_pass_losses[0].item():.4f}")
    print(f"  Pass 2: {per_pass_losses[1].item():.4f}")
    print(f"  Pass 3: {per_pass_losses[2].item():.4f}")
    
    # === Backward pass ===
    print("\n--- Backward ---")
    total_loss.backward()
    
    # Check gradients exist
    has_grad = sum(1 for p in pipeline.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in pipeline.parameters())
    print(f"  Parameters with non-zero gradient: {has_grad}/{total_params}")
    
    # === Aggregation ===
    print("\n--- Aggregation ---")
    n = X.shape[0]
    final_scores = aggregate_scores(scores_list, tracker, n)
    print(f"  Final scores shape: {final_scores.shape}")
    print(f"  Score range: [{final_scores.min().item():.4f}, {final_scores.max().item():.4f}]")
    
    # === Metrics ===
    print("\n--- Metrics ---")
    metrics = compute_metrics(final_scores.detach(), y)
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    
    # === Per-pass breakdown ===
    print("\n--- Per-pass evaluation ---")
    breakdown = aggregate_scores_with_breakdown(scores_list, tracker, n)
    for key in ['pass1', 'pass2', 'pass3', 'final']:
        m = compute_metrics(breakdown[key].detach(), y)
        print(f"  {key}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}")
    
    print("\n" + "=" * 60)
    print("Smoke test PASSED.")
    print("=" * 60)


if __name__ == '__main__':
    main()
