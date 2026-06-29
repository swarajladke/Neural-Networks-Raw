"""
Raw AGNIS — src/agnis/memory/fast_memory.py

FastMemory: Rapid episodic prototype store.

Stores compressed representations of novel or high-error experiences.
Supports:
  - Write: store (key, value, metadata) when novelty/error is high
  - Retrieve: find nearest prototype using cosine similarity
  - Importance update: increment importance of frequently retrieved entries
  - Capacity management: evict oldest low-importance entries when full

Design:
  - key   = sparse latent activation a (compact and distinctive)
  - value = full input s (for reconstruction during replay)
  - entry metadata: timestamp, importance, error_at_write, usage_count, task_id

This is NOT a gradient-based memory. Writes and reads are deterministic
nearest-neighbor operations. This is an explicit episodic store.
"""

import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """A single stored memory prototype."""
    key: torch.Tensor         # sparse latent activation (d_z,)
    value: torch.Tensor       # original input (d_in,)
    timestamp: int            # step at which this was written
    error_at_write: float     # prediction error at write time
    importance: float = 0.0   # updated based on retrieval frequency
    usage_count: int = 0      # number of times this entry was retrieved
    task_id: Optional[int] = None  # task during which this was written


class FastMemory:
    """
    Rapid episodic prototype store with cosine-similarity retrieval.

    Parameters
    ----------
    capacity : int
        Maximum number of stored entries.
    write_error_threshold : float
        Minimum prediction error to trigger a write (high-error → write).
    write_novelty_threshold : float
        Minimum novelty to trigger a write.
    importance_decay : float
        Decay factor for importance scores (prevents all entries staying equally important).
    min_similarity_to_skip_write : float
        If the nearest existing entry has cosine sim > this, skip the write
        (avoid storing near-duplicates).
    """

    def __init__(
        self,
        capacity: int = 256,
        write_error_threshold: float = 0.3,
        write_novelty_threshold: float = 0.2,
        importance_decay: float = 0.999,
        min_similarity_to_skip_write: float = 0.95,
    ):
        self.capacity = capacity
        self.write_error_threshold = write_error_threshold
        self.write_novelty_threshold = write_novelty_threshold
        self.importance_decay = importance_decay
        self.min_similarity_to_skip_write = min_similarity_to_skip_write

        self._entries: List[MemoryEntry] = []
        self._step: int = 0

        # EMA of prediction error for novelty computation
        self._ema_error: float = 0.0
        self._ema_alpha: float = 0.1

    @property
    def size(self) -> int:
        """Number of stored entries."""
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        return len(self._entries) >= self.capacity

    @property
    def utilization(self) -> float:
        """Fraction of capacity used."""
        return len(self._entries) / self.capacity

    def tick(self):
        """Increment internal step counter."""
        self._step += 1

    def _update_ema_error(self, error_val: float):
        """Update EMA of prediction error (novelty signal)."""
        self._ema_error = (1 - self._ema_alpha) * self._ema_error + self._ema_alpha * error_val

    def _compute_novelty(self, error_val: float) -> float:
        """
        Novelty = current error relative to EMA.
        High when current error significantly exceeds recent average.
        """
        return max(0.0, error_val - self._ema_error)

    def _cosine_similarity(
        self, query: torch.Tensor, key: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """Cosine similarity between two vectors."""
        q_norm = query.norm().item()
        k_norm = key.norm().item()
        if q_norm < eps or k_norm < eps:
            return 0.0
        return (query @ key).item() / (q_norm * k_norm)

    def should_write(self, error_val: float, novelty: float) -> bool:
        """Determine whether to write based on error and novelty thresholds."""
        return (
            error_val > self.write_error_threshold
            or novelty > self.write_novelty_threshold
        )

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        error_val: float,
        task_id: Optional[int] = None,
    ) -> bool:
        """
        Attempt to write a new memory entry.

        Parameters
        ----------
        key : torch.Tensor of shape (d_z,)
            Sparse latent activation used as the retrieval key.
        value : torch.Tensor of shape (d_in,)
            Original input stored as the memory content.
        error_val : float
            Prediction error magnitude at write time.
        task_id : int, optional
            Task identifier for tracking.

        Returns
        -------
        bool
            True if the entry was written, False if skipped.
        """
        novelty = self._compute_novelty(error_val)
        self._update_ema_error(error_val)

        if not self.should_write(error_val, novelty):
            return False

        # Check for near-duplicates — skip if too similar to existing entry
        if len(self._entries) > 0:
            nearest, sim = self._find_nearest(key)
            if sim > self.min_similarity_to_skip_write:
                # Update the existing entry's importance instead of writing a duplicate
                nearest.importance += 0.1
                nearest.usage_count += 1
                return False

        entry = MemoryEntry(
            key=key.detach().clone(),
            value=value.detach().clone(),
            timestamp=self._step,
            error_at_write=error_val,
            importance=error_val,   # initialize importance = write-time error
            usage_count=0,
            task_id=task_id,
        )

        if self.is_full:
            self._evict_one()

        self._entries.append(entry)
        return True

    def _find_nearest(self, query: torch.Tensor) -> Tuple[MemoryEntry, float]:
        """
        Find the nearest stored entry by cosine similarity.

        Returns
        -------
        (entry, similarity) : tuple
        """
        best_entry = self._entries[0]
        best_sim = self._cosine_similarity(query, self._entries[0].key)

        for entry in self._entries[1:]:
            sim = self._cosine_similarity(query, entry.key)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        return best_entry, best_sim

    def retrieve(
        self, query: torch.Tensor, update_importance: bool = True
    ) -> Optional[Tuple[torch.Tensor, float, MemoryEntry]]:
        """
        Retrieve the nearest memory entry for a given query activation.

        Parameters
        ----------
        query : torch.Tensor of shape (d_z,)
            Current sparse activation used as the retrieval key.
        update_importance : bool
            Whether to increment the retrieved entry's importance and usage.

        Returns
        -------
        (value, similarity, entry) : tuple or None
            value = stored input tensor (d_in,)
            similarity = cosine similarity score
            entry = the full MemoryEntry object
        Returns None if memory is empty.
        """
        if len(self._entries) == 0:
            return None

        entry, sim = self._find_nearest(query)

        if update_importance:
            entry.usage_count += 1
            entry.importance += 0.05

        return entry.value.clone(), sim, entry

    def _evict_one(self):
        """Evict the entry with the lowest importance score."""
        if not self._entries:
            return
        min_idx = min(
            range(len(self._entries)),
            key=lambda i: self._entries[i].importance,
        )
        self._entries.pop(min_idx)

    def decay_importance(self):
        """Apply importance decay to all entries. Call periodically (e.g., each task boundary)."""
        for entry in self._entries:
            entry.importance *= self.importance_decay

    def get_all_entries(self) -> List[MemoryEntry]:
        """Return all stored entries (read-only reference)."""
        return self._entries

    def sample_by_importance(self, n: int) -> List[MemoryEntry]:
        """
        Sample n entries weighted by importance score.

        Parameters
        ----------
        n : int
            Number of entries to sample.

        Returns
        -------
        list of MemoryEntry
        """
        if len(self._entries) == 0:
            return []
        n = min(n, len(self._entries))
        importances = torch.tensor([e.importance for e in self._entries], dtype=torch.float)
        importances = importances.clamp(min=1e-8)
        probs = importances / importances.sum()
        indices = torch.multinomial(probs, n, replacement=False)
        return [self._entries[i] for i in indices.tolist()]

    def task_entries(self, task_id: int) -> List[MemoryEntry]:
        """Return all entries written during a specific task."""
        return [e for e in self._entries if e.task_id == task_id]

    def stats(self) -> Dict:
        """Return memory statistics for logging."""
        if not self._entries:
            return {
                "size": 0,
                "utilization": 0.0,
                "mean_importance": 0.0,
                "mean_error_at_write": 0.0,
                "mean_usage": 0.0,
            }
        importances = [e.importance for e in self._entries]
        errors = [e.error_at_write for e in self._entries]
        usages = [e.usage_count for e in self._entries]
        return {
            "size": len(self._entries),
            "utilization": self.utilization,
            "mean_importance": sum(importances) / len(importances),
            "mean_error_at_write": sum(errors) / len(errors),
            "mean_usage": sum(usages) / len(usages),
        }

    def __repr__(self) -> str:
        return (
            f"FastMemory(size={self.size}/{self.capacity}, "
            f"utilization={self.utilization:.2%})"
        )
