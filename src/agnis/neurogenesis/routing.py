"""
Raw AGNIS — src/agnis/neurogenesis/routing.py

Hierarchical Neurogenesis Routing: Assigns growth to appropriate layers
using novelty attribution ratios.

Phase 6 implementation.
Routes growth decisions based on where the unexplained variance bottleneck
lives in the hierarchy:
  - High e^{l-1}, low rho^l: detail novelty -> grow at layer l
  - High e^{l-1}, high rho^l: abstraction missing -> escalate to layer l+1
  - Top-down suppression: stable parent requires persistent trigger

See: Phase 6 design (Fable 5) for full specification.
"""

import math
from typing import List, Optional, Tuple


class NoveltyAttributor:
    """
    Tracks per-layer novelty attribution ratios rho^l = |e^l| / |e^{l-1}|
    using exponential moving averages, and routes growth decisions.

    Parameters
    ----------
    n_layers : int
        Number of layers in the hierarchy.
    ema_alpha : float
        EMA decay for error tracking.
    suppression_window : int
        Number of consecutive windows the trigger must persist before
        a birth is allowed when the parent layer is stable.
    stable_gate_threshold : float
        Parent precision gate value below which the parent is considered
        "stable" (low error relative to its own history).
    max_units_per_layer : list of int or None
        Maximum unit count cap per layer. None means no cap.
    """

    def __init__(
        self,
        n_layers: int,
        ema_alpha: float = 0.02,
        suppression_window: int = 50,
        stable_gate_threshold: float = 0.3,
        max_units_per_layer: Optional[List[int]] = None,
    ):
        self.n_layers = n_layers
        self.ema_alpha = ema_alpha
        self.suppression_window = suppression_window
        self.stable_gate_threshold = stable_gate_threshold
        self.max_units_per_layer = max_units_per_layer or [128] * n_layers

        # EMA of |e^l| per layer
        self._ema_error: List[float] = [0.0] * n_layers
        # EMA of |e^l|^2 (precision estimate)
        self._ema_error_sq: List[float] = [1.0] * n_layers
        # Consecutive trigger counts per layer (for suppression window)
        self._trigger_persistence: List[int] = [0] * n_layers
        # Per-layer precision gate g^l (for top-down suppression check)
        self._gate_values: List[float] = [0.5] * n_layers

    def update(
        self,
        per_layer_errors: List[float],
        per_layer_gates: Optional[List[float]] = None,
    ):
        """
        Update EMA error tracking for all layers.

        Parameters
        ----------
        per_layer_errors : list of float
            |e^l| for each layer (bottom-up prediction error norm).
        per_layer_gates : list of float or None
            Precision gate values g^l per layer. If None, uses 0.5 for all.
        """
        alpha = self.ema_alpha
        for l in range(self.n_layers):
            err = per_layer_errors[l] if l < len(per_layer_errors) else 0.0
            self._ema_error[l] = (1 - alpha) * self._ema_error[l] + alpha * err
            self._ema_error_sq[l] = (1 - alpha) * self._ema_error_sq[l] + alpha * (err ** 2)

        if per_layer_gates is not None:
            for l in range(min(len(per_layer_gates), self.n_layers)):
                self._gate_values[l] = per_layer_gates[l]

    def compute_novelty_ratios(self) -> List[float]:
        """
        Compute per-layer novelty attribution ratios rho^l = |e^l| / |e^{l-1}|.

        Returns
        -------
        list of float
            rho^l for each layer l >= 1. rho[0] is always 1.0 (no layer below).
        """
        ratios = [1.0]  # rho^0 = 1.0 (bottom layer, no parent reference)
        for l in range(1, self.n_layers):
            below = max(self._ema_error[l - 1], 1e-8)
            above = self._ema_error[l]
            ratios.append(above / below)
        return ratios

    def route_growth(
        self,
        per_layer_trigger_active: List[bool],
        per_layer_current_dims: List[int],
    ) -> int:
        """
        Determine which layer should receive a new unit.

        Logic:
        1. Find the lowest layer where the local trigger is active.
        2. Compute rho^l at that layer.
           - If rho^l is LOW (< 0.5): detail novelty -> grow at layer l.
           - If rho^l is HIGH (>= 0.5): abstraction missing -> escalate to l+1.
        3. Apply top-down suppression: if the parent layer is stable (low gate),
           the trigger must persist for suppression_window steps.
        4. Respect per-layer capacity caps.

        Parameters
        ----------
        per_layer_trigger_active : list of bool
            Whether the growth criterion is met at each layer.
        per_layer_current_dims : list of int
            Current latent dimension at each layer.

        Returns
        -------
        int
            Layer index to grow at, or -1 if no growth should occur.
        """
        ratios = self.compute_novelty_ratios()

        for l in range(self.n_layers):
            if not per_layer_trigger_active[l]:
                self._trigger_persistence[l] = 0
                continue

            self._trigger_persistence[l] += 1

            # Check capacity cap
            if per_layer_current_dims[l] >= self.max_units_per_layer[l]:
                # Try escalating to layer above
                if l + 1 < self.n_layers and per_layer_current_dims[l + 1] < self.max_units_per_layer[l + 1]:
                    return l + 1
                continue

            # Check rho: should we grow here or escalate?
            target_layer = l
            if l + 1 < self.n_layers and ratios[l] >= 0.5:
                # Abstraction above is also struggling -> escalate
                if per_layer_current_dims[l + 1] < self.max_units_per_layer[l + 1]:
                    target_layer = l + 1

            # Top-down suppression check
            if target_layer + 1 < self.n_layers:
                parent_gate = self._gate_values[target_layer + 1]
                if parent_gate < self.stable_gate_threshold:
                    # Parent is stable -> require persistent trigger
                    if self._trigger_persistence[l] < self.suppression_window:
                        continue  # Suppress: trigger hasn't persisted long enough

            return target_layer

        return -1  # No growth


def select_growth_layer(
    per_layer_growth_scores: List[float],
) -> int:
    """
    Select which layer to grow a new unit in.

    Strategy: grow in the layer with the highest growth score.

    Parameters
    ----------
    per_layer_growth_scores : list of float
        Current growth score for each layer.

    Returns
    -------
    int
        Index of the layer to add a new unit to.
    """
    if not per_layer_growth_scores:
        return 0
    return max(range(len(per_layer_growth_scores)), key=lambda i: per_layer_growth_scores[i])
