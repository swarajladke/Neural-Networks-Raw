"""
Raw AGNIS — Autonomous Generative Neuroplastic Intelligence System
src/agnis/__init__.py

Top-level package init. Exposes version and core imports.
"""

__version__ = "0.1.0"
__project__ = "Raw AGNIS"
__description__ = "Standalone continual learning neural architecture"

# Lazy imports — only import what's needed to avoid circular dependencies
# from agnis.core import PredictiveCell, PredictiveHierarchy
# from agnis.memory import FastMemory, ReplayBuffer
# from agnis.evaluation import ForgettingTracker

__all__ = [
    "__version__",
    "__project__",
    "__description__",
]
