"""
Module 4: Node-Neutralized Discrepancy Scoring.

GCN encoder-decoder that reconstructs features. Anomaly score = 1 - cos(input, output).
Shared parameters across all 3 passes.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.stable_ops import cosine_similarity_safe, EPS


class GCNLayer(nn.Module):
    """
    Single GCN layer: H' = activation(A_norm @ H @ W).
    
    Operates on dense adjacency for simplicity in Phase 1.
    """
    
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, A_norm: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_norm: (n, n) normalized adjacency (typically D^(-1/2)(A+I)D^(-1/2))
            H: (n, in_dim) input features
        
        Returns:
            H_out: (n, out_dim)
        """
        # Aggregation: A_norm @ H
        H_agg = A_norm @ H
        # Linear transformation
        H_out = self.linear(H_agg)
        if self.activation:
            H_out = F.relu(H_out)
        return H_out


class NodeNeutralizedScoringModule(nn.Module):
    """
    Module 4: GCN encoder-decoder with shared parameters across passes.
    
    Architecture:
        Input → [3-layer GCN encoder with residuals] → projection → [2-layer GCN decoder] → output
        
        Anomaly score per node = 1 - cos(input, output)
        
        Loss = E[(1 - cos)^alpha] + beta * Var(latent)
    """
    
    def __init__(
        self,
        d_input: int,
        d_hidden: int = 64,
        d_latent: int = 32,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.dropout = dropout
        
        # Encoder
        encoder_dims = [d_input] + [d_hidden] * (num_encoder_layers - 1) + [d_latent]
        self.encoder = nn.ModuleList([
            GCNLayer(encoder_dims[i], encoder_dims[i + 1], activation=(i < num_encoder_layers - 1))
            for i in range(num_encoder_layers)
        ])
        
        # Residual projections (when dim mismatch in encoder)
        self.encoder_residual_proj = nn.ModuleList([
            nn.Linear(encoder_dims[i], encoder_dims[i + 1])
            if encoder_dims[i] != encoder_dims[i + 1] else nn.Identity()
            for i in range(num_encoder_layers)
        ])
        
        # Bottleneck projection
        self.projection = nn.Linear(d_latent, d_latent)
        
        # Decoder
        decoder_dims = [d_latent] + [d_hidden] * (num_decoder_layers - 1) + [d_input]
        self.decoder = nn.ModuleList([
            GCNLayer(
                decoder_dims[i], decoder_dims[i + 1],
                activation=(i < num_decoder_layers - 1)
            )
            for i in range(num_decoder_layers)
        ])
        
        # Decoder residual projections
        self.decoder_residual_proj = nn.ModuleList([
            nn.Linear(decoder_dims[i], decoder_dims[i + 1])
            if decoder_dims[i] != decoder_dims[i + 1] else nn.Identity()
            for i in range(num_decoder_layers)
        ])
    
    def forward(
        self,
        X: torch.Tensor,
        A_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (n, d_input) input features (X^rec in Zero-GAD notation)
            A_norm: (n, n) normalized adjacency
        
        Returns:
            X_gen: (n, d_input) reconstructed features
            Z: (n, d_latent) latent representation
        """
        # Encoder with residual connections
        h = X
        for layer, res_proj in zip(self.encoder, self.encoder_residual_proj):
            h_new = layer(A_norm, h)
            # Apply dropout
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual
            h_residual = res_proj(h)
            h = h_new + h_residual
        
        # Bottleneck
        Z = self.projection(h)  # (n, d_latent)
        
        # Decoder with residual connections
        h = Z
        for layer, res_proj in zip(self.decoder, self.decoder_residual_proj):
            h_new = layer(A_norm, h)
            if self.dropout > 0:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h_residual = res_proj(h)
            h = h_new + h_residual
        
        X_gen = h  # (n, d_input)
        
        return X_gen, Z


def compute_anomaly_score(X_rec: torch.Tensor, X_gen: torch.Tensor) -> torch.Tensor:
    """
    Per-node anomaly score (raw, not normalized).
    
    Args:
        X_rec: (n, d) input
        X_gen: (n, d) generated
    
    Returns:
        scores: (n,) raw anomaly scores in [0, 2]
    """
    cos_sim = cosine_similarity_safe(X_rec, X_gen, dim=-1)
    return 1.0 - cos_sim
