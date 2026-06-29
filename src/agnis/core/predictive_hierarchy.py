"""
Raw AGNIS — src/agnis/core/predictive_hierarchy.py

PredictiveHierarchy: A stack of PredictiveCell layers forming a
multi-level predictive coding hierarchy.

Design:
  - Layer 0 receives raw input s
  - Each layer l predicts layer l-1's activation
  - Prediction error at layer l drives updates for layer l
  - Higher layers learn more abstract, compressed representations
  - For Phase 1, a single-layer hierarchy suffices.
  - Multi-layer hierarchies are intended for Phase 2+.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict

from agnis.core.predictive_cell import PredictiveCell


class PredictiveHierarchy(nn.Module):
    """
    A stacked predictive coding hierarchy of PredictiveCells.

    Parameters
    ----------
    layer_dims : list of int
        Sizes of each layer, starting with input dimension.
        E.g., [64, 32, 16] creates two layers:
          Layer 0: d_in=64, d_z=32
          Layer 1: d_in=32, d_z=16
    k_sparse_per_layer : list of int or None
        kWTA k values per layer. If None, all layers use default.
    cell_kwargs : dict
        Additional keyword arguments passed to each PredictiveCell.
    """

    def __init__(
        self,
        layer_dims: List[int],
        k_sparse_per_layer: Optional[List[int]] = None,
        cell_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        assert len(layer_dims) >= 2, "Need at least 2 dims (input + one latent)"

        cell_kwargs = cell_kwargs or {}
        if k_sparse_per_layer is None:
            k_sparse_per_layer = [0] * (len(layer_dims) - 1)  # 0 = dense

        self.n_layers = len(layer_dims) - 1
        self.cells = nn.ModuleList()

        for l in range(self.n_layers):
            d_in = layer_dims[l]
            d_z = layer_dims[l + 1]
            k = k_sparse_per_layer[l]
            cell = PredictiveCell(
                d_in=d_in,
                d_z=d_z,
                k_sparse=k,
                **cell_kwargs,
            )
            self.cells.append(cell)

    def forward(
        self,
        s: torch.Tensor,
        z_prevs: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass through the hierarchy.

        Parameters
        ----------
        s : torch.Tensor of shape (d_in,)
            Raw input stimulus for layer 0.
        z_prevs : list of tensors, optional
            Previous latent states for each layer (for recurrent drive).

        Returns
        -------
        activations : list of torch.Tensor
            Sparse activations [a_0, a_1, ..., a_{L-1}].
        """
        if z_prevs is None:
            z_prevs = [None] * self.n_layers

        activations = []
        current_input = s

        for l, cell in enumerate(self.cells):
            a = cell.forward(current_input, z_prev=z_prevs[l])
            activations.append(a)
            current_input = a  # higher layer gets lower layer's activation as input

        return activations

    def update_all_weights(
        self,
        s: torch.Tensor,
        activations: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Apply Hebbian weight updates to all layers.

        Parameters
        ----------
        s : torch.Tensor of shape (d_in,)
            Raw input.
        activations : list of torch.Tensor
            Activations from forward pass.

        Returns
        -------
        dict
            Aggregated weight delta norms per layer for logging.
        """
        metrics = {}
        inputs = [s] + activations[:-1]  # input to each layer

        for l, (cell, inp, act) in enumerate(zip(self.cells, inputs, activations)):
            layer_metrics = cell.update_weights(inp, act)
            for k, v in layer_metrics.items():
                metrics[f"layer{l}/{k}"] = v

        return metrics

    def reset_state(self):
        """Reset all recurrent states across layers."""
        for cell in self.cells:
            cell.reset_state()

    @property
    def total_prediction_error(self) -> float:
        """Sum of prediction errors across all layers."""
        errors = [c.prediction_error for c in self.cells if c.prediction_error is not None]
        return sum(errors)

    @property
    def per_layer_prediction_error(self) -> List[Optional[float]]:
        """Prediction error per layer."""
        return [c.prediction_error for c in self.cells]

    @property
    def per_layer_sparsity(self) -> List[float]:
        """Sparsity level per layer."""
        return [c.sparsity_level for c in self.cells]

    def get_config(self) -> dict:
        """Return full hierarchy configuration."""
        return {
            "n_layers": self.n_layers,
            "cells": [cell.get_config() for cell in self.cells],
        }
