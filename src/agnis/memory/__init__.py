"""
Raw AGNIS — src/agnis/memory/__init__.py

Memory subpackage:
  - FastMemory: rapid episodic prototype store (key-value with cosine retrieval)
  - ReplayBuffer: importance-weighted sampling for sleep-phase replay
  - SemanticPrototypes: slow stable prototype store (post-consolidation)
  - Consolidation: utilities for migrating fast → slow memory
"""

from agnis.memory.fast_memory import FastMemory
from agnis.memory.replay_buffer import ReplayBuffer

__all__ = [
    "FastMemory",
    "ReplayBuffer",
]
