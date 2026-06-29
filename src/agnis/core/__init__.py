"""
Raw AGNIS — src/agnis/core/__init__.py

Core predictive coding components:
  - PredictiveCell: single-layer predictive coding unit
  - PredictiveHierarchy: multi-layer hierarchy
  - Settling: iterative state refinement
  - Hebbian rules: local weight update rules
  - Sparsity: kWTA and lateral inhibition
"""

from agnis.core.predictive_cell import PredictiveCell
from agnis.core.sparsity import kwta, compute_sparsity_level
from agnis.core.hebbian_rules import (
    hebbian_generative_update,
    hebbian_recognition_update,
    hebbian_recurrent_update,
    plasticity_gated_update,
)

__all__ = [
    "PredictiveCell",
    "kwta",
    "compute_sparsity_level",
    "hebbian_generative_update",
    "hebbian_recognition_update",
    "hebbian_recurrent_update",
    "plasticity_gated_update",
]
