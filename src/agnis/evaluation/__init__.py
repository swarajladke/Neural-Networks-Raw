"""
Raw AGNIS — src/agnis/evaluation/__init__.py

Evaluation subpackage:
  - ForgettingTracker: task accuracy tracking and forgetting computation
  - Metrics: aggregated continual learning metrics
  - Baselines: naive MLP, dense Hebbian, simple RNN baselines
  - Probes: representation analysis probes
"""

from agnis.evaluation.forgetting import ForgettingTracker, compute_forgetting
from agnis.evaluation.metrics import ContinualLearningMetrics

__all__ = [
    "ForgettingTracker",
    "compute_forgetting",
    "ContinualLearningMetrics",
]
