"""
Raw AGNIS — src/agnis/sequence/recurrent_state.py

RecurrentState: Temporal state management for sequential prediction tasks.

Phase 2 implementation. Manages z_prev for all layers in the hierarchy
and handles sequence boundary resets.
"""

import torch
from typing import List, Optional


class RecurrentStateManager:
    """
    Manages recurrent (temporal) state for a PredictiveHierarchy.

    Tracks z_prev for each layer and provides reset functionality
    at sequence boundaries.

    Parameters
    ----------
    layer_sizes : list of int
        Latent dimension for each layer in the hierarchy.
    """

    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self._z_prevs: List[torch.Tensor] = [
            torch.zeros(d_z) for d_z in layer_sizes
        ]

    def get_z_prevs(self) -> List[torch.Tensor]:
        """Return current z_prev for all layers."""
        return [z.clone() for z in self._z_prevs]

    def update(self, current_z_list: List[torch.Tensor]):
        """Update z_prev with current latent states from a forward pass."""
        for i, z in enumerate(current_z_list):
            if i < self.n_layers:
                self._z_prevs[i] = z.detach().clone()

    def reset(self):
        """Reset all z_prev to zero (call at sequence boundaries)."""
        self._z_prevs = [torch.zeros(d_z) for d_z in self.layer_sizes]
