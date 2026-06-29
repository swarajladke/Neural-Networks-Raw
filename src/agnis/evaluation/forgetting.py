"""
Raw AGNIS — src/agnis/evaluation/forgetting.py

Forgetting metrics for continual learning evaluation.

Core metric:
  F_i = max_{t <= T_i} A_{i,t} - A_{i,T}

Where:
  - A_{i,t} = accuracy of model on task i, evaluated after training on task t
  - T_i     = last time task i was trained on
  - T       = final evaluation checkpoint

A positive F_i means forgetting occurred. Zero = no forgetting.
Negative F_i = backward transfer (learning new tasks helped old ones — rare).
"""

import torch
from typing import Dict, List, Optional


def compute_forgetting(
    accuracy_matrix: List[List[float]],
    task_train_order: List[int],
) -> List[float]:
    """
    Compute per-task forgetting from an accuracy matrix.

    Parameters
    ----------
    accuracy_matrix : list of list of float
        accuracy_matrix[eval_checkpoint][task_id] = accuracy on task_id
        at evaluation checkpoint eval_checkpoint.
        Shape: (n_checkpoints, n_tasks)
    task_train_order : list of int
        task_train_order[checkpoint] = task_id trained up to that checkpoint.
        Used to determine T_i (last training checkpoint for each task).

    Returns
    -------
    list of float
        Per-task forgetting values F_i for tasks 0..N-2
        (last task has no post-training evaluation, so excluded).

    Example
    -------
    # 3 tasks, 3 checkpoints (evaluated after each task):
    # checkpoint 0: trained task 0 → eval [0.9, -, -]
    # checkpoint 1: trained task 1 → eval [0.7, 0.85, -]
    # checkpoint 2: trained task 2 → eval [0.65, 0.80, 0.92]
    # F_0 = max(0.9, 0.7, 0.65) - 0.65 = 0.9 - 0.65 = 0.25
    # F_1 = max(0.85, 0.80) - 0.80 = 0.85 - 0.80 = 0.05
    """
    if not accuracy_matrix or not task_train_order:
        return []

    n_checkpoints = len(accuracy_matrix)
    n_tasks = len(accuracy_matrix[0]) if accuracy_matrix else 0

    forgetting = []
    # Compute forgetting for all tasks except the last one
    for task_id in range(n_tasks - 1):
        # Find checkpoints where this task was evaluated (not None/-1)
        task_accuracies = []
        for checkpoint in range(n_checkpoints):
            acc = accuracy_matrix[checkpoint][task_id]
            if acc is not None and acc >= 0:
                task_accuracies.append(acc)

        if len(task_accuracies) < 2:
            forgetting.append(0.0)
            continue

        # Peak accuracy for this task
        peak_acc = max(task_accuracies)
        # Final accuracy for this task
        final_acc = task_accuracies[-1]

        F_i = peak_acc - final_acc
        forgetting.append(F_i)

    return forgetting


def compute_average_forgetting(forgetting: List[float]) -> float:
    """Average forgetting across all tasks (excluding last)."""
    if not forgetting:
        return 0.0
    return sum(forgetting) / len(forgetting)


def compute_backward_transfer(accuracy_matrix: List[List[float]]) -> float:
    """
    Compute backward transfer (BWT).

    BWT = (1/(N-1)) * sum_{i=1}^{N-1} (A_{i,T} - A_{i,T_i})

    Negative = forgetting (learning later tasks hurts older ones).
    Positive = backward transfer (rare, means later training somehow helps earlier tasks).
    """
    if not accuracy_matrix or len(accuracy_matrix) < 2:
        return 0.0

    n_checkpoints = len(accuracy_matrix)
    n_tasks = len(accuracy_matrix[0])

    bwt_values = []
    for task_id in range(n_tasks - 1):
        # A_{i, T_i}: accuracy right after task i was trained
        # (approximately: the checkpoint where task_id was trained)
        # For simplicity, assume task i was trained at checkpoint i
        if task_id < n_checkpoints:
            acc_at_train_time = accuracy_matrix[task_id][task_id]
        else:
            continue

        # A_{i, T}: final accuracy
        acc_final = accuracy_matrix[-1][task_id]

        if acc_at_train_time is not None and acc_final is not None:
            bwt_values.append(acc_final - acc_at_train_time)

    if not bwt_values:
        return 0.0
    return sum(bwt_values) / len(bwt_values)


class ForgettingTracker:
    """
    Tracks task accuracy over time and computes continual learning metrics.

    Usage:
        tracker = ForgettingTracker(n_tasks=3)
        # After training task 0:
        tracker.record(task_id=0, checkpoint=0, accuracy=0.9)
        tracker.record(task_id=1, checkpoint=0, accuracy=None)  # not yet trained
        tracker.record(task_id=2, checkpoint=0, accuracy=None)
        # After training task 1:
        tracker.record(task_id=0, checkpoint=1, accuracy=0.7)
        tracker.record(task_id=1, checkpoint=1, accuracy=0.85)
        tracker.record(task_id=2, checkpoint=1, accuracy=None)
        # ...
        print(tracker.forgetting)      # per-task forgetting
        print(tracker.avg_forgetting)  # average forgetting
    """

    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks
        # accuracy_matrix[checkpoint][task_id] = accuracy or None
        self._matrix: List[List[Optional[float]]] = []
        self._current_checkpoint: int = 0
        self._n_checkpoints: int = 0
        self._task_first_trained: List[int] = [-1] * n_tasks

    def new_checkpoint(self):
        """Start a new evaluation checkpoint row."""
        self._matrix.append([None] * self.n_tasks)
        self._current_checkpoint = len(self._matrix) - 1

    def record(self, task_id: int, accuracy: float, checkpoint: Optional[int] = None):
        """
        Record accuracy of model on task_id at a checkpoint.

        Parameters
        ----------
        task_id : int
        accuracy : float
            Accuracy value in [0, 1].
        checkpoint : int, optional
            If None, uses the most recently created checkpoint.
        """
        if checkpoint is None:
            checkpoint = self._current_checkpoint

        # Ensure the matrix has enough rows
        while len(self._matrix) <= checkpoint:
            self._matrix.append([None] * self.n_tasks)

        self._matrix[checkpoint][task_id] = accuracy

        # Track first training checkpoint
        if self._task_first_trained[task_id] == -1:
            self._task_first_trained[task_id] = checkpoint

    @property
    def matrix(self) -> List[List[Optional[float]]]:
        """Full accuracy matrix."""
        return self._matrix

    @property
    def forgetting(self) -> List[float]:
        """Per-task forgetting values."""
        task_order = list(range(self.n_tasks))
        return compute_forgetting(self._matrix, task_order)

    @property
    def avg_forgetting(self) -> float:
        """Average forgetting across all tasks (excluding last)."""
        return compute_average_forgetting(self.forgetting)

    @property
    def backward_transfer(self) -> float:
        """Backward transfer metric."""
        return compute_backward_transfer(self._matrix)

    def summary(self) -> Dict:
        """Return full summary dict for logging."""
        forgetting = self.forgetting
        return {
            "n_tasks": self.n_tasks,
            "n_checkpoints": len(self._matrix),
            "per_task_forgetting": forgetting,
            "avg_forgetting": self.avg_forgetting,
            "backward_transfer": self.backward_transfer,
            "final_accuracies": self._matrix[-1] if self._matrix else [],
        }

    def print_matrix(self):
        """Pretty-print the accuracy matrix."""
        header = "Checkpoint | " + " | ".join([f"Task {i}" for i in range(self.n_tasks)])
        print(header)
        print("-" * len(header))
        for ckpt, row in enumerate(self._matrix):
            vals = " | ".join(
                [f"{v:.3f}" if v is not None else "  --- " for v in row]
            )
            print(f"    {ckpt:5d}  | {vals}")
