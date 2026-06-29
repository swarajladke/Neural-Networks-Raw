"""
Raw AGNIS — src/agnis/neurogenesis/routing.py

Routing: Assigns new units to appropriate layers in a hierarchy.

Phase 3 implementation stub.
In Phase 3, growth is single-layer. Multi-layer column growth is Phase 3+.
"""

from typing import List


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
