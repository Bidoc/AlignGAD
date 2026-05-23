"""
Membership tracking utility for multi-pass hierarchical clustering.

Tracks the mapping from super-cluster indices at each pass back to ORIGINAL node indices.
"""

from typing import Dict, List, Optional
import torch


class MembershipTracker:
    """
    Tracks original node membership through hierarchical clustering passes.
    
    Usage:
        tracker = MembershipTracker(n_original=100)
        
        # Pass 1: cluster original nodes into super-clusters
        P_1 = ...  # (100, 50) assignment
        tracker.update(P_1)  # now tracker[j] = list of original nodes in super-cluster j
        
        # Pass 2: cluster super-clusters from Pass 1
        P_2 = ...  # (50, 25)
        tracker.update(P_2)  # now tracker[j] = list of original nodes in pass-2 super-cluster j
        
        # Get mapping for any pass
        membership_pass1 = tracker.get_membership(level=1)
        membership_pass2 = tracker.get_membership(level=2)
    """
    
    def __init__(self, n_original: int):
        """
        Args:
            n_original: number of original nodes
        """
        self.n_original = n_original
        self.history: List[Dict[int, List[int]]] = []  # one mapping per pass
    
    def update(self, P: torch.Tensor) -> Dict[int, List[int]]:
        """
        Update membership using a new assignment matrix P.
        
        Args:
            P: (N_input, N_output) assignment matrix from current pass.
               In forward pass, P should be one-hot (hard assignment).
        
        Returns:
            new_membership: dict {super_cluster_idx -> list of original node indices}
        """
        # Get hard assignment from P (argmax along output dim)
        hard_assignment = P.argmax(dim=-1).cpu().numpy()  # (N_input,)
        N_input = hard_assignment.shape[0]
        N_output = P.shape[1]
        
        new_membership: Dict[int, List[int]] = {}
        
        if len(self.history) == 0:
            # First pass: input indices ARE original node indices
            assert N_input == self.n_original, \
                f"First pass should have N_input={self.n_original}, got {N_input}"
            
            for input_idx in range(N_input):
                cluster_idx = int(hard_assignment[input_idx])
                if cluster_idx not in new_membership:
                    new_membership[cluster_idx] = []
                new_membership[cluster_idx].append(input_idx)
        
        else:
            # Subsequent pass: input indices are super-clusters from previous pass
            prev_membership = self.history[-1]
            
            for input_idx in range(N_input):
                cluster_idx = int(hard_assignment[input_idx])
                
                # Get original nodes belonging to this input super-cluster
                original_nodes = prev_membership.get(input_idx, [])
                
                if cluster_idx not in new_membership:
                    new_membership[cluster_idx] = []
                new_membership[cluster_idx].extend(original_nodes)
        
        # Ensure all output clusters have entries (even if empty)
        for j in range(N_output):
            if j not in new_membership:
                new_membership[j] = []
        
        self.history.append(new_membership)
        return new_membership
    
    def get_membership(self, level: int) -> Dict[int, List[int]]:
        """
        Get membership at a specific pass level.
        
        Args:
            level: 1 = after first clustering, 2 = after second, etc.
        
        Returns:
            dict {super_cluster_idx -> list of original node indices}
        """
        if level < 1 or level > len(self.history):
            raise IndexError(f"Level {level} out of range (have {len(self.history)} passes)")
        return self.history[level - 1]
    
    def __len__(self) -> int:
        """Number of passes recorded."""
        return len(self.history)
    
    def reset(self):
        """Clear all history (for reuse on a different graph)."""
        self.history = []
    
    def get_node_to_cluster_at_level(self, level: int) -> torch.Tensor:
        """
        For each ORIGINAL node, get the super-cluster index at a given level.
        
        Args:
            level: pass level (1, 2, 3, ...)
        
        Returns:
            tensor (n_original,) where output[v] = super-cluster index of node v at this level
        """
        membership = self.get_membership(level)
        result = torch.full((self.n_original,), -1, dtype=torch.long)
        
        for cluster_idx, original_nodes in membership.items():
            for v in original_nodes:
                result[v] = cluster_idx
        
        # Sanity check: all nodes should be assigned
        assert (result >= 0).all(), \
            f"Some nodes have no cluster assignment at level {level}"
        
        return result
