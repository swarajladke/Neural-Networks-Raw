"""
Raw AGNIS — src/agnis/evaluation/metrics.py

ContinualLearningMetrics: Aggregated metrics container for a full experiment run.
"""

import json
from typing import List, Optional, Dict
from agnis.evaluation.forgetting import ForgettingTracker


class ContinualLearningMetrics:
    """
    Aggregated container for all continual learning metrics across an experiment.

    Tracks:
    - Per-task accuracy over time (via ForgettingTracker)
    - Prediction error curves
    - Sparsity levels
    - Memory utilization
    - Adaptation speed per task
    - Replay benefit (if applicable)
    """

    def __init__(self, n_tasks: int, model_name: str = "model"):
        self.n_tasks = n_tasks
        self.model_name = model_name
        self.forgetting_tracker = ForgettingTracker(n_tasks)

        # Time-series metrics
        self.error_curve: List[float] = []          # MSE per step
        self.sparsity_curve: List[float] = []       # sparsity % per step
        self.memory_size_curve: List[int] = []      # memory entries per step
        self.unit_count_curve: List[int] = []       # total units per step

        # Per-task metrics
        self.adaptation_speed: List[Optional[int]] = [None] * n_tasks  # steps to 80% acc

        # Step counter
        self.step: int = 0

    def log_step(
        self,
        error: float,
        sparsity: Optional[float] = None,
        memory_size: Optional[int] = None,
        unit_count: Optional[int] = None,
    ):
        """Log per-step metrics."""
        self.error_curve.append(error)
        if sparsity is not None:
            self.sparsity_curve.append(sparsity)
        if memory_size is not None:
            self.memory_size_curve.append(memory_size)
        if unit_count is not None:
            self.unit_count_curve.append(unit_count)
        self.step += 1

    def record_task_accuracy(self, task_id: int, accuracy: float, checkpoint: Optional[int] = None):
        """Record task accuracy at current checkpoint."""
        self.forgetting_tracker.record(task_id, accuracy, checkpoint)

    def new_checkpoint(self):
        """Create new evaluation checkpoint."""
        self.forgetting_tracker.new_checkpoint()

    def record_adaptation_speed(self, task_id: int, steps_to_threshold: int):
        """Record how many steps it took to reach accuracy threshold on task_id."""
        self.adaptation_speed[task_id] = steps_to_threshold

    def summary(self) -> Dict:
        """Return full metrics summary."""
        return {
            "model_name": self.model_name,
            "n_tasks": self.n_tasks,
            "total_steps": self.step,
            "forgetting": self.forgetting_tracker.summary(),
            "adaptation_speed": self.adaptation_speed,
            "final_error": self.error_curve[-1] if self.error_curve else None,
            "mean_sparsity": (
                sum(self.sparsity_curve) / len(self.sparsity_curve)
                if self.sparsity_curve else None
            ),
            "final_memory_size": (
                self.memory_size_curve[-1] if self.memory_size_curve else None
            ),
            "final_unit_count": (
                self.unit_count_curve[-1] if self.unit_count_curve else None
            ),
        }

    def to_json(self, filepath: str):
        """Save summary to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.summary(), f, indent=2)

    def print_summary(self):
        """Print a readable summary."""
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  {self.model_name} — Continual Learning Summary")
        print(f"{'='*60}")
        print(f"  Tasks: {self.n_tasks}  |  Steps: {self.step}")
        print(f"  Avg Forgetting: {s['forgetting']['avg_forgetting']:.4f}")
        print(f"  Backward Transfer: {s['forgetting']['backward_transfer']:.4f}")
        print(f"  Per-task Forgetting: {s['forgetting']['per_task_forgetting']}")
        print(f"  Final Error: {s['final_error']}")
        if s['mean_sparsity'] is not None:
            print(f"  Mean Sparsity: {s['mean_sparsity']:.2%}")
        print(f"{'='*60}\n")
