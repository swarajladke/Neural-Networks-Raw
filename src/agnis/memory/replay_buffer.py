"""
Raw AGNIS — src/agnis/memory/replay_buffer.py

ReplayBuffer: Importance-weighted experience store for sleep-phase replay.

The replay buffer is the bridge between fast episodic memory and the
sleep/consolidation trainer. It:
  - Receives important entries from FastMemory
  - Provides importance-weighted sampling for the SleepTrainer
  - Tracks which entries have been replayed and how many times

Separate from FastMemory: FastMemory is the write-on-surprise store.
ReplayBuffer is the curated replay set used during offline consolidation.
"""

import torch
from typing import List, Optional, Tuple
from agnis.memory.fast_memory import MemoryEntry


class ReplayBuffer:
    """
    Curated replay buffer for sleep-phase consolidation.

    Parameters
    ----------
    max_size : int
        Maximum number of entries to store in the replay buffer.
    """

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._entries: List[MemoryEntry] = []
        self._replay_counts: List[int] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, entry: MemoryEntry):
        """Add an entry to the replay buffer. Evicts lowest-importance if full."""
        if len(self._entries) >= self.max_size:
            # Evict lowest-importance
            min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].importance)
            self._entries.pop(min_idx)
            self._replay_counts.pop(min_idx)

        self._entries.append(entry)
        self._replay_counts.append(0)

    def add_from_memory(self, fast_memory, n: int = 32):
        """
        Populate replay buffer from FastMemory's top-importance entries.

        Parameters
        ----------
        fast_memory : FastMemory
            Source of entries.
        n : int
            Number of entries to pull.
        """
        sampled = fast_memory.sample_by_importance(n)
        for entry in sampled:
            self.add(entry)

    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Sample a batch for replay.

        Returns list of (key, value, entry_idx) tuples,
        weighted by importance (high-importance entries replayed more).

        Parameters
        ----------
        batch_size : int
            Number of entries to sample.

        Returns
        -------
        list of (key, value, entry_idx)
        """
        if not self._entries:
            return []

        batch_size = min(batch_size, len(self._entries))
        importances = torch.tensor(
            [e.importance for e in self._entries], dtype=torch.float
        ).clamp(min=1e-8)
        probs = importances / importances.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False).tolist()

        results = []
        for idx in indices:
            entry = self._entries[idx]
            self._replay_counts[idx] += 1
            results.append((entry.key.clone(), entry.value.clone(), idx))

        return results

    def stats(self) -> dict:
        """Replay buffer statistics."""
        if not self._entries:
            return {"size": 0, "mean_replay_count": 0.0}
        return {
            "size": len(self._entries),
            "mean_replay_count": sum(self._replay_counts) / len(self._replay_counts),
            "max_replay_count": max(self._replay_counts),
        }


class LatentReplayBuffer:
    """
    Stores latent state snapshots per domain using reservoir sampling
    to prevent recency bias and catastrophic forgetting.
    """

    def __init__(self, max_per_domain: int = 64):
        self.max_per_domain = max_per_domain
        self.buffers = {}  # task_id -> list of [z_0, z_1, ...]
        self.counts = {}   # task_id -> count of items seen

    def add(self, task_id: int, latent_states: List[torch.Tensor]):
        """
        Add a latent snapshot [z_0, z_1, ...] for a given task_id
        using reservoir sampling.
        """
        # Detach and clone to avoid reference memory leaks
        cloned_states = [z.detach().clone() for z in latent_states]

        if task_id not in self.buffers:
            self.buffers[task_id] = []
            self.counts[task_id] = 0

        self.counts[task_id] += 1
        n_seen = self.counts[task_id]

        if len(self.buffers[task_id]) < self.max_per_domain:
            self.buffers[task_id].append(cloned_states)
        else:
            # Reservoir sampling: choose a random index in 0..n_seen-1
            import random
            idx = random.randint(0, n_seen - 1)
            if idx < self.max_per_domain:
                self.buffers[task_id][idx] = cloned_states

    def sample(self, batch_size: int) -> List[List[torch.Tensor]]:
        """
        Sample a random batch of snapshots from all accumulated tasks.
        """
        all_snapshots = []
        for task_id, snapshots in self.buffers.items():
            all_snapshots.extend(snapshots)

        if not all_snapshots:
            return []

        import random
        k = min(batch_size, len(all_snapshots))
        return random.sample(all_snapshots, k)

    @property
    def size(self) -> int:
        """Total number of snapshots stored across all domains."""
        return sum(len(s) for s in self.buffers.values())

    def stats(self) -> dict:
        """Retrieve buffer diagnostics."""
        return {
            "total_snapshots": self.size,
            "domains": list(self.buffers.keys()),
            "snapshots_per_domain": {task_id: len(s) for task_id, s in self.buffers.items()},
        }


