"""
Raw AGNIS — src/agnis/memory/consolidation.py

Consolidation utilities: migrating learned patterns from fast memory
to stable slow weights during sleep/replay phases.
"""

import torch
from typing import Optional
from agnis.memory.fast_memory import FastMemory
from agnis.memory.replay_buffer import ReplayBuffer


def consolidate_to_buffer(
    fast_memory: FastMemory,
    replay_buffer: ReplayBuffer,
    n_entries: int = 32,
    min_importance: float = 0.1,
) -> int:
    """
    Transfer high-importance entries from fast memory to the replay buffer.

    Parameters
    ----------
    fast_memory : FastMemory
    replay_buffer : ReplayBuffer
    n_entries : int
        Number of entries to transfer.
    min_importance : float
        Only transfer entries above this importance threshold.

    Returns
    -------
    int
        Number of entries actually transferred.
    """
    candidates = [e for e in fast_memory.get_all_entries() if e.importance >= min_importance]
    candidates.sort(key=lambda e: e.importance, reverse=True)
    transferred = 0
    for entry in candidates[:n_entries]:
        replay_buffer.add(entry)
        transferred += 1
    return transferred
