"""
Raw AGNIS — src/agnis/evaluation/probes.py

Representation analysis probes.

Used to analyze what Raw AGNIS has learned:
- RepresentationOverlapProbe: cosine similarity between task-specific representations
- ActivationSelectivityProbe: how selective are individual units to specific tasks/inputs

These are read-only diagnostic tools. They do not modify the model.
"""

import torch
from typing import List, Dict, Optional
from agnis.core.predictive_cell import PredictiveCell


class RepresentationOverlapProbe:
    """
    Measures the overlap between sparse representations of different tasks.

    High overlap = high interference risk.
    Low overlap = good separation = sparsity is working.

    Usage:
        probe = RepresentationOverlapProbe(n_tasks=3)
        # Collect activations per task
        probe.record_task(task_id=0, activations=task0_acts)
        probe.record_task(task_id=1, activations=task1_acts)
        print(probe.pairwise_overlap())
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        self._task_activations: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_tasks)}

    def record_task(self, task_id: int, activations: List[torch.Tensor]):
        """Record a list of activation vectors for a task."""
        self._task_activations[task_id].extend(activations)

    def mean_activation(self, task_id: int) -> Optional[torch.Tensor]:
        """Compute mean activation vector for a task."""
        acts = self._task_activations.get(task_id, [])
        if not acts:
            return None
        return torch.stack(acts).mean(dim=0)

    def pairwise_overlap(self) -> Dict[str, float]:
        """
        Compute pairwise cosine similarity between task mean activations.

        Returns
        -------
        dict with keys like "task0_vs_task1" → float
        """
        result = {}
        mean_acts = {}
        for task_id in range(self.n_tasks):
            mean_act = self.mean_activation(task_id)
            if mean_act is not None:
                mean_acts[task_id] = mean_act

        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                if i in mean_acts and j in mean_acts:
                    a_i = mean_acts[i]
                    a_j = mean_acts[j]
                    sim = (a_i @ a_j) / (a_i.norm() * a_j.norm() + 1e-8)
                    result[f"task{i}_vs_task{j}"] = sim.item()

        return result


class ActivationSelectivityProbe:
    """
    Measures how selective individual units are to specific tasks.

    Selectivity_j = |mean_task_A[j] - mean_task_B[j]| / (mean_task_A[j] + mean_task_B[j] + eps)

    High selectivity = unit activates preferentially on one task (good — specialized).
    Low selectivity = unit activates similarly on all tasks (bad — not specialized).
    """

    def __init__(self):
        self._task_mean_acts: Dict[int, torch.Tensor] = {}

    def record(self, task_id: int, mean_activation: torch.Tensor):
        """Record mean activation for a task."""
        self._task_mean_acts[task_id] = mean_activation

    def selectivity(self, task_a: int, task_b: int, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute per-unit selectivity between task_a and task_b.

        Returns
        -------
        torch.Tensor of shape (d_z,)
            Per-unit selectivity scores in [0, 1].
        """
        if task_a not in self._task_mean_acts or task_b not in self._task_mean_acts:
            return torch.tensor([0.0])

        a = self._task_mean_acts[task_a].abs()
        b = self._task_mean_acts[task_b].abs()
        return (a - b).abs() / (a + b + eps)

    def mean_selectivity(self, task_a: int, task_b: int) -> float:
        """Mean selectivity score across all units."""
        sel = self.selectivity(task_a, task_b)
        return sel.mean().item()


class WordBoundaryProbe:
    """Linear probe testing whether z^l encodes word boundary (space) prediction.

    Trains a simple ridge regression classifier on collected (z, is_space) pairs.
    Higher accuracy at layer l means that layer has learned word-level structure.
    This is the direct test that hierarchy produces abstract representations.
    """

    def __init__(self):
        self._features: List[torch.Tensor] = []
        self._labels: List[int] = []

    def record(self, z: torch.Tensor, next_char_is_space: bool):
        """Record a (latent_state, is_next_char_space) pair."""
        self._features.append(z.detach().clone())
        self._labels.append(1 if next_char_is_space else 0)

    def evaluate(self) -> float:
        """Train linear probe via ridge regression, return test accuracy."""
        if len(self._features) < 20:
            return 0.0
        X = torch.stack(self._features)
        y = torch.tensor(self._labels, dtype=torch.float32)
        n = X.shape[0]
        split = int(0.8 * n)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        lam = 1.0
        XtX = X_train.T @ X_train + lam * torch.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        w = torch.linalg.solve(XtX, Xty)
        preds = (X_test @ w > 0.5).float()
        accuracy = (preds == y_test).float().mean().item()
        return accuracy

    def reset(self):
        """Clear all recorded features and labels."""
        self._features.clear()
        self._labels.clear()
