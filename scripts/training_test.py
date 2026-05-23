"""
Quick training test: train on synthetic graphs and verify loss decreases + AUROC improves.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scripts.smoke_test import make_synthetic_graph

from ms_zerogad.pipeline.multipass import MultiPassPipeline
from ms_zerogad.training.train import train
from ms_zerogad.evaluation.metrics import evaluate_pipeline


def main():
    print("=" * 60)
    print("MS-ZeroGAD Quick Training Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create source graphs (training)
    source_graphs = []
    for i in range(3):
        X, A, y = make_synthetic_graph(n=150, f=30, seed=i)
        source_graphs.append((X, A))
    
    print(f"\nCreated {len(source_graphs)} source graphs (each n=150)")
    
    # Create target graph (test)
    X_test, A_test, y_test = make_synthetic_graph(n=200, f=30, seed=99)
    print(f"Created test graph: n={X_test.shape[0]}")
    
    # Build pipeline (small for speed)
    pipeline = MultiPassPipeline(
        d_prime=8,
        d_hidden=16,
        d_latent=8,
        D_rff=20,
        d_svd=8,
        kmeans_max_iter=10,
    ).to(device)
    
    # Eval BEFORE training
    pipeline.eval()
    metrics_before = evaluate_pipeline(pipeline, X_test, A_test, y_test, device=device)
    print(f"\n[Before training] AUROC={metrics_before['auroc']:.4f}, AUPRC={metrics_before['auprc']:.4f}")
    
    # Train
    print("\n--- Training ---")
    history = train(
        pipeline,
        source_graphs,
        num_epochs=30,
        lr=1e-3,
        cache_unified=True,
        device=device,
        verbose=False,  # we'll print summary
    )
    
    # Print summary
    print(f"  Initial loss: {history['total_loss'][0]:.4f}")
    print(f"  Final loss:   {history['total_loss'][-1]:.4f}")
    print(f"  Loss decrease: {history['total_loss'][0] - history['total_loss'][-1]:.4f}")
    
    # Eval AFTER training
    metrics_after = evaluate_pipeline(pipeline, X_test, A_test, y_test, device=device)
    print(f"\n[After training] AUROC={metrics_after['auroc']:.4f}, AUPRC={metrics_after['auprc']:.4f}")
    print(f"AUROC change: {metrics_after['auroc'] - metrics_before['auroc']:+.4f}")
    
    # Sanity checks
    loss_decreased = history['total_loss'][-1] < history['total_loss'][0]
    auroc_improved = metrics_after['auroc'] > metrics_before['auroc']
    
    print(f"\n--- Sanity Checks ---")
    print(f"  Loss decreased: {loss_decreased}")
    print(f"  AUROC improved: {auroc_improved}")
    
    if loss_decreased:
        print("\nQUICK TRAINING TEST PASSED.")
    else:
        print("\nWARNING: Loss did not decrease — investigate.")


if __name__ == '__main__':
    main()
