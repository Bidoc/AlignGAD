"""
Multi-pass orchestration: combine all modules into the 3-level hierarchical pipeline.

Pass 1: Score on original graph (n nodes)
Pass 2: Cluster to n/2 super-nodes, score
Pass 3: Cluster to n/4 super-nodes, score
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from ..modules.unification import GlobalUnification
from ..modules.clustering import RFFClustering
from ..modules.supergraph import build_super_graph
from ..modules.scoring import NodeNeutralizedScoringModule, compute_anomaly_score
from ..data.preprocessing import normalize_adjacency
from ..utils.tracking import MembershipTracker


class MultiPassPipeline(nn.Module):
    """
    Full hierarchical pipeline.
    
    Components:
    - Module 1 (GlobalUnification): applied ONCE before Pass 1
    - Module 2 (RFFClustering): used after Pass 1 and Pass 2 for super-graph construction
    - Module 3 (build_super_graph): used after Pass 1 and Pass 2
    - Module 4 (NodeNeutralizedScoringModule): SHARED across all 3 passes
    """
    
    def __init__(
        self,
        # Module 1
        d_prime: int = 8,
        band_low: float = 0.5,
        band_high: float = 1.5,
        alpha_low: float = 1.0,
        alpha_mid: float = 0.95,
        alpha_high: float = 0.9,
        adaptive_bands: bool = False,           # ← thêm
        band_low_percentile: float = 0.33,      # ← thêm
        band_high_percentile: float = 0.67,     # ← thêm
        # Module 2
        k_smoothing: int = 1,
        sigma: float = 1.0,
        D_rff: int = 50,
        d_svd: int = 32,
        tau: float = 0.5,
        kmeans_max_iter: int = 20,
        seed: int = 42,
        # Module 4
        d_hidden: int = 64,
        d_latent: int = 32,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.0,
        # Multi-pass
        num_passes: int = 3,
        cluster_ratios: Tuple[float, ...] = (0.5, 0.25, 0.125),
    ):
        super().__init__()
        
        self.num_passes = num_passes
        self.cluster_ratios = cluster_ratios
        
        # Module 1 (no params, but module-style for consistency)
        self.unification = GlobalUnification(
            d_prime=d_prime,
            band_low=band_low,
            band_high=band_high,
            alpha_low=alpha_low,
            alpha_mid=alpha_mid,
            alpha_high=alpha_high,
            adaptive_bands=adaptive_bands,
            band_low_percentile=band_low_percentile,
            band_high_percentile=band_high_percentile,
        )
        # Module 2 (no params)
        self.clustering = RFFClustering(
            k_smoothing=k_smoothing,
            sigma=sigma,
            D_rff=D_rff,
            d_svd=d_svd,
            tau=tau,
            kmeans_max_iter=kmeans_max_iter,
            seed=seed,
        )
        
        # Module 4 (TRAINABLE, shared across passes)
        self.scoring = NodeNeutralizedScoringModule(
            d_input=d_prime,
            d_hidden=d_hidden,
            d_latent=d_latent,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
    
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        is_training: bool = True,
        precomputed_X_unified: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], MembershipTracker, dict]:
        """
        Full forward through 3-level hierarchy.
        
        Args:
            X: (n, f) original features
            A: (n, n) original adjacency (dense, binary)
            is_training: if True, returns per-pass losses requirement intermediates
            precomputed_X_unified: (n, d_prime) optional pre-computed Module 1 output
                                    (for caching across epochs)
        
        Returns:
            scores_list: list of 3 raw score tensors
                [scores_1 (n,), scores_2 (n/2,), scores_3 (n/4,)]
            features_list: list of (X_rec, X_gen) pairs per pass
                [(X_rec_1, X_gen_1), (X_rec_2, X_gen_2), (X_rec_3, X_gen_3)]
                Useful for computing losses
            tracker: MembershipTracker with cluster history
            extras: dict with intermediate tensors for debugging/inspection
        """
        n = X.shape[0]
        device = X.device
        
        # === Module 1: Global Information Unification (once) ===
        if precomputed_X_unified is not None:
            X_unified = precomputed_X_unified
        else:
            X_unified = self.unification(X, A)
        
        scores_list: List[torch.Tensor] = []
        features_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        tracker = MembershipTracker(n_original=n)
        extras = {}
        
        # === PASS 1: Score on original graph ===
        A_norm_1 = normalize_adjacency(A, add_self_loops=True)
        X_gen_1, Z_1 = self.scoring(X_unified, A_norm_1)
        scores_1 = compute_anomaly_score(X_unified, X_gen_1)  # (n,)
        scores_list.append(scores_1)
        features_list.append((X_unified, X_gen_1))
        extras['Z_1'] = Z_1
        
        # === PASS 1 → PASS 2: Cluster and build super-graph ===
        m_pass1 = max(2, int(n * self.cluster_ratios[0]))  # n/2, but at least 2
        P_1 = self.clustering(X_unified, A, num_clusters=m_pass1)  # (n, m_pass1)
        membership_1 = tracker.update(P_1)
        
        X_super_1, A_super_1_hat = build_super_graph(
            X=X_unified,
            A=A,
            P=P_1,
            membership_for_self_loop=membership_1,
            A_original=A,
        )
        # X_super_1: (m_pass1, d_prime), A_super_1_hat: (m_pass1, m_pass1) — already includes I
        
        # === PASS 2: Score on super-graph 1 ===
        A_norm_2 = self._normalize_with_existing_self_loops(A_super_1_hat)
        X_gen_2, Z_2 = self.scoring(X_super_1, A_norm_2)
        scores_2 = compute_anomaly_score(X_super_1, X_gen_2)  # (m_pass1,)
        scores_list.append(scores_2)
        features_list.append((X_super_1, X_gen_2))
        extras['Z_2'] = Z_2
        extras['m_pass1'] = m_pass1
        
        # === PASS 2 → PASS 3: Cluster super-graph 1, build super-graph 2 ===
        m_pass2 = max(2, m_pass1 // 2)  # m_pass1 / 2
        
        # Use A_super_1_hat WITHOUT identity for clustering input (clustering doesn't need self-loops here)
        # but the binary off-diagonal does include the self-loops we set in Module 3
        # Actually safer: use A_super_1_hat as is (includes self-loops + I)
        # The clustering module will add its own self-loops in smoothing
        # Better: pass the raw A (without the +I we added in Module 3)
        A_super_1_raw = A_super_1_hat - torch.eye(m_pass1, device=device)
        
        P_2 = self.clustering(X_super_1, A_super_1_raw, num_clusters=m_pass2)  # (m_pass1, m_pass2)
        membership_2 = tracker.update(P_2)
        
        X_super_2, A_super_2_hat = build_super_graph(
            X=X_super_1,
            A=A_super_1_raw,
            P=P_2,
            membership_for_self_loop=membership_2,
            A_original=A,  # always trace back to original (Cách α)
        )
        
        # === PASS 3: Score on super-graph 2 ===
        A_norm_3 = self._normalize_with_existing_self_loops(A_super_2_hat)
        X_gen_3, Z_3 = self.scoring(X_super_2, A_norm_3)
        scores_3 = compute_anomaly_score(X_super_2, X_gen_3)  # (m_pass2,)
        scores_list.append(scores_3)
        features_list.append((X_super_2, X_gen_3))
        extras['Z_3'] = Z_3
        extras['m_pass2'] = m_pass2
        
        return scores_list, features_list, tracker, extras
    
    def _normalize_with_existing_self_loops(self, A_hat: torch.Tensor) -> torch.Tensor:
        """
        Normalize an adjacency that ALREADY has self-loops (so don't add I again).
        
        D^(-1/2) A_hat D^(-1/2)
        """
        degree = A_hat.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_hat @ D_inv_sqrt
