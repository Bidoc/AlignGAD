"""
Training loop for MS-ZeroGAD.

- Iterates over multiple source graphs (multi-source domain training)
- Computes multi-pass loss per graph
- Standard Adam optimizer with gradient clipping (for STE stability)
"""

from typing import List, Tuple, Dict, Optional, Callable
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from ..pipeline.multipass import MultiPassPipeline
from ..training.losses import multipass_total_loss


def train_one_epoch(
    pipeline: MultiPassPipeline,
    optimizer: Adam,
    source_graphs: List[Tuple[torch.Tensor, torch.Tensor]],
    cached_unified: Optional[List[torch.Tensor]] = None,
    alpha: float = 2.0,
    beta: float = 0.5,
    loss_weights: List[float] = None,
    grad_clip_norm: float = 1.0,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Train pipeline for one epoch over all source graphs.
    
    Args:
        pipeline: MS-ZeroGAD pipeline
        optimizer: Adam optimizer
        source_graphs: list of (X, A) tuples — features and adjacency for each source graph
        cached_unified: optional list of pre-computed Module 1 outputs
        alpha, beta, loss_weights: loss hyperparameters
        grad_clip_norm: gradient clipping norm (for STE stability)
        device: torch device
    
    Returns:
        metrics dict with per-pass losses and total
    """
    if device is None:
        device = next(pipeline.parameters()).device
    
    pipeline.train()
    
    metrics = {
        'total_loss': 0.0,
        'pass1_loss': 0.0,
        'pass2_loss': 0.0,
        'pass3_loss': 0.0,
        'num_graphs': 0,
    }
    
    # Random shuffle source graphs
    perm = torch.randperm(len(source_graphs))
    
    for i in perm.tolist():
        X, A = source_graphs[i]
        X = X.to(device)
        A = A.to(device)
        
        precomputed = None
        if cached_unified is not None:
            precomputed = cached_unified[i].to(device)
        
        # Forward
        scores_list, features_list, tracker, extras = pipeline(
            X, A, is_training=True, precomputed_X_unified=precomputed
        )
        
        # Collect Z's from extras
        Z_list = [extras['Z_1'], extras['Z_2'], extras['Z_3']]
        
        # Compute loss
        total_loss, per_pass_losses = multipass_total_loss(
            features_list, Z_list, alpha=alpha, beta=beta, weights=loss_weights
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for STE stability
        torch.nn.utils.clip_grad_norm_(pipeline.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        # Logging
        metrics['total_loss'] += total_loss.item()
        metrics['pass1_loss'] += per_pass_losses[0].item()
        metrics['pass2_loss'] += per_pass_losses[1].item()
        metrics['pass3_loss'] += per_pass_losses[2].item()
        metrics['num_graphs'] += 1
    
    # Average over graphs
    n = max(1, metrics['num_graphs'])
    for key in ['total_loss', 'pass1_loss', 'pass2_loss', 'pass3_loss']:
        metrics[key] /= n
    
    return metrics


def precompute_unified_features(
    pipeline: MultiPassPipeline,
    source_graphs: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device = None,
) -> List[torch.Tensor]:
    """
    Precompute X_unified for each source graph (Module 1 output).
    
    Module 1 has no trainable params, so we can compute once and cache.
    Saves a lot of time during training (eigendecomposition is O(n^3)).
    
    Args:
        pipeline: pipeline (only its unification module is used)
        source_graphs: list of (X, A) tuples
        device: torch device
    
    Returns:
        list of X_unified tensors (on CPU to save GPU memory; move to GPU when used)
    """
    if device is None:
        device = next(pipeline.parameters()).device
    
    pipeline.eval()
    cached = []
    
    print(f"Precomputing unified features for {len(source_graphs)} graphs...")
    
    with torch.no_grad():
        for i, (X, A) in enumerate(source_graphs):
            t_start = time.time()
            X_dev = X.to(device)
            A_dev = A.to(device)
            X_unified = pipeline.unification(X_dev, A_dev)
            cached.append(X_unified.cpu())
            elapsed = time.time() - t_start
            print(f"  Graph {i+1}/{len(source_graphs)}: n={X.shape[0]}, elapsed={elapsed:.2f}s")
    
    return cached


def train(
    pipeline: MultiPassPipeline,
    source_graphs: List[Tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    alpha: float = 2.0,
    beta: float = 0.5,
    loss_weights: List[float] = None,
    grad_clip_norm: float = 1.0,
    cache_unified: bool = True,
    eval_callback: Optional[Callable] = None,
    eval_interval: int = 5,
    save_best_path=None,         # ← thêm
    cfg_for_save=None, 
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Full training loop.
    
    Args:
        pipeline: MS-ZeroGAD pipeline
        source_graphs: list of (X, A) source graphs
        num_epochs: training epochs
        lr, weight_decay: optimizer parameters
        alpha, beta, loss_weights: loss parameters
        grad_clip_norm: gradient clipping
        cache_unified: precompute Module 1 outputs
        eval_callback: optional function called every eval_interval, signature f(pipeline, epoch) -> dict
        eval_interval: epochs between evaluations
        device: torch device
        verbose: print progress
    
    Returns:
        history dict with per-epoch metrics
    """
    if device is None:
        device = next(pipeline.parameters()).device
    
    optimizer = Adam(pipeline.parameters(), lr=lr, weight_decay=weight_decay)
    
    cached_unified = None
    if cache_unified:
        cached_unified = precompute_unified_features(pipeline, source_graphs, device)
    
    history = {
        'epoch': [],
        'total_loss': [],
        'pass1_loss': [],
        'pass2_loss': [],
        'pass3_loss': [],
        'eval': [],
    }
    
    best_auroc = -1.0
    best_epoch = -1
    for epoch in range(1, num_epochs + 1):
        t_start = time.time()
        
        metrics = train_one_epoch(
            pipeline, optimizer, source_graphs,
            cached_unified=cached_unified,
            alpha=alpha, beta=beta, loss_weights=loss_weights,
            grad_clip_norm=grad_clip_norm, device=device,
        )
        
        elapsed = time.time() - t_start
        
        history['epoch'].append(epoch)
        history['total_loss'].append(metrics['total_loss'])
        history['pass1_loss'].append(metrics['pass1_loss'])
        history['pass2_loss'].append(metrics['pass2_loss'])
        history['pass3_loss'].append(metrics['pass3_loss'])
        
        if verbose:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"L_total={metrics['total_loss']:.4f} | "
                f"L1={metrics['pass1_loss']:.4f} L2={metrics['pass2_loss']:.4f} L3={metrics['pass3_loss']:.4f} | "
                f"{elapsed:.1f}s"
            )
        
        # Evaluation
        if eval_callback is not None and epoch % eval_interval == 0:
            eval_result = eval_callback(pipeline, epoch)
            history['eval'].append({'epoch': epoch, **eval_result})
            if verbose:
                print(f"  [Eval] {eval_result}")
            
            # Save best checkpoint
            if save_best_path is not None:
                # Lấy AUROC trung bình hoặc của một dataset cụ thể
                aurocs = [v for k, v in eval_result.items() if k.endswith('_auroc')]
                avg_auroc = sum(aurocs) / len(aurocs) if aurocs else 0
                
                if avg_auroc > best_auroc:
                    best_auroc = avg_auroc
                    best_epoch = epoch
                    torch.save({
                        'state_dict': pipeline.state_dict(),
                        'config': cfg_for_save,
                        'history': history,
                        'best_epoch': epoch,
                        'best_auroc': avg_auroc,
                    }, save_best_path)
                    if verbose:
                        print(f"  [Best] New best AUROC={avg_auroc:.4f} at epoch {epoch} → saved")
    
    if save_best_path is not None and verbose:
        print(f"\nBest checkpoint: epoch {best_epoch}, AUROC={best_auroc:.4f}")
    
    return history