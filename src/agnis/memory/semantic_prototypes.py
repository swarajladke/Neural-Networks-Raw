"""
Raw AGNIS — src/agnis/memory/semantic_prototypes.py

SemanticPrototypes: Slow stable prototype store.

Unlike FastMemory (rapid write-on-surprise), semantic prototypes represent
patterns that have been observed repeatedly and consolidated from fast memory.
They are updated slowly and infrequently — they represent stable, generalizable
abstractions rather than specific episodic experiences.

Phase 1–2: Not actively used. FastMemory + ReplayBuffer are sufficient.
Phase 3+: SemanticPrototypes accumulate stable abstractions from repeated replay.
"""

import torch
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Prototype:
    """A semantic prototype: a stable, consolidated representation."""
    centroid: torch.Tensor      # mean activation pattern (d_z,)
    reconstruction: torch.Tensor  # mean input pattern (d_in,)
    n_observations: int = 0      # how many experiences contributed
    stability: float = 0.0       # how stable this prototype is (convergence score)
    task_ids: List[int] = field(default_factory=list)


class SemanticPrototypes:
    """
    Slow stable prototype store. Updated during consolidation phases.

    Parameters
    ----------
    capacity : int
        Maximum number of prototypes.
    merge_threshold : float
        Cosine similarity threshold above which two prototypes are merged.
    """

    def __init__(self, capacity: int = 64, merge_threshold: float = 0.9):
        self.capacity = capacity
        self.merge_threshold = merge_threshold
        self._prototypes: List[Prototype] = []

    @property
    def size(self) -> int:
        return len(self._prototypes)

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        eta: float = 0.1,
        task_id: Optional[int] = None,
    ):
        """
        Update or create a prototype matching the given key.

        If a sufficiently similar prototype exists, update it (running average).
        Otherwise, create a new prototype.

        Parameters
        ----------
        key : torch.Tensor of shape (d_z,)
            Activation key.
        value : torch.Tensor of shape (d_in,)
            Input value.
        eta : float
            Learning rate for prototype centroid update (slow).
        task_id : int, optional
        """
        if self._prototypes:
            nearest, sim = self._find_nearest(key)
            if sim > self.merge_threshold:
                # Update existing prototype with running average
                nearest.centroid = (1 - eta) * nearest.centroid + eta * key
                nearest.reconstruction = (1 - eta) * nearest.reconstruction + eta * value
                nearest.n_observations += 1
                if task_id is not None and task_id not in nearest.task_ids:
                    nearest.task_ids.append(task_id)
                return

        # Create new prototype
        if len(self._prototypes) >= self.capacity:
            # Evict least-observed prototype
            min_idx = min(range(len(self._prototypes)), key=lambda i: self._prototypes[i].n_observations)
            self._prototypes.pop(min_idx)

        proto = Prototype(
            centroid=key.detach().clone(),
            reconstruction=value.detach().clone(),
            n_observations=1,
            stability=0.0,
            task_ids=[task_id] if task_id is not None else [],
        )
        self._prototypes.append(proto)

    def _find_nearest(self, query: torch.Tensor, eps: float = 1e-8) -> Tuple[Prototype, float]:
        """Find the nearest prototype by cosine similarity."""
        best_proto = self._prototypes[0]
        q_norm = query.norm().item()
        best_sim = -1.0

        for proto in self._prototypes:
            k_norm = proto.centroid.norm().item()
            if q_norm < eps or k_norm < eps:
                sim = 0.0
            else:
                sim = (query @ proto.centroid).item() / (q_norm * k_norm)
            if sim > best_sim:
                best_sim = sim
                best_proto = proto

        return best_proto, best_sim

    def retrieve(self, query: torch.Tensor) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve the nearest prototype reconstruction."""
        if not self._prototypes:
            return None
        proto, sim = self._find_nearest(query)
        return proto.reconstruction.clone(), sim

    def stats(self) -> Dict:
        """Return prototype store statistics."""
        if not self._prototypes:
            return {"size": 0}
        obs = [p.n_observations for p in self._prototypes]
        return {
            "size": len(self._prototypes),
            "mean_observations": sum(obs) / len(obs),
            "max_observations": max(obs),
        }
