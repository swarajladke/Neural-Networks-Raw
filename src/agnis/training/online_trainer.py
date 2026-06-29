"""
Raw AGNIS — src/agnis/training/online_trainer.py

OnlineTrainer: Online sequential task training loop.

Manages:
- Sequential task presentation
- Per-step Hebbian updates on PredictiveCell/Hierarchy
- Memory writes on high-error/novelty events
- Metric logging
- Evaluation after each task

This is the primary training loop for Phase 1 and Phase 2.
"""

import torch
from typing import List, Callable, Optional, Dict, Tuple
from agnis.core.predictive_cell import PredictiveCell
from agnis.memory.fast_memory import FastMemory
from agnis.evaluation.metrics import ContinualLearningMetrics


class OnlineTrainer:
    """
    Online trainer for Raw AGNIS PredictiveCell on sequential tasks.

    Parameters
    ----------
    model : PredictiveCell
        The Raw AGNIS model to train.
    fast_memory : FastMemory, optional
        If provided, writes high-error experiences to fast memory.
    metrics : ContinualLearningMetrics, optional
        Metrics tracker.
    log_every : int
        Log metrics every N steps.
    """

    def __init__(
        self,
        model: PredictiveCell,
        fast_memory: Optional[FastMemory] = None,
        metrics: Optional[ContinualLearningMetrics] = None,
        log_every: int = 10,
    ):
        self.model = model
        self.fast_memory = fast_memory
        self.metrics = metrics
        self.log_every = log_every
        self._step = 0

    def train_task(
        self,
        task_id: int,
        data: List[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 1,
        target_key: str = "value",  # "value" = next token/target, "self" = reconstruction
    ) -> List[float]:
        """
        Train on a single task for n_epochs passes.

        Parameters
        ----------
        task_id : int
            Task identifier for logging and memory tagging.
        data : list of (input, target) pairs
            Input-target pairs for this task.
        n_epochs : int
            Number of passes through the data.

        Returns
        -------
        list of float
            Per-step prediction error during training.
        """
        error_log = []

        for epoch in range(n_epochs):
            for s, target in data:
                # Forward: settle latent state
                a = self.model.forward(s)

                # Prediction error
                error_val = self.model.prediction_error or 0.0
                error_log.append(error_val)

                # Weight update (local Hebbian)
                self.model.update_weights(s, a)

                # Memory write on high error/novelty
                if self.fast_memory is not None:
                    self.fast_memory.tick()
                    self.fast_memory.write(
                        key=a.detach().clone(),
                        value=s.detach().clone(),
                        error_val=error_val,
                        task_id=task_id,
                    )

                # Log metrics
                if self.metrics is not None and self._step % self.log_every == 0:
                    self.metrics.log_step(
                        error=error_val,
                        sparsity=self.model.sparsity_level,
                        memory_size=(
                            self.fast_memory.size if self.fast_memory else None
                        ),
                    )

                self._step += 1

        return error_log

    def evaluate_task(
        self,
        data: List[Tuple[torch.Tensor, torch.Tensor]],
        threshold: float = 0.5,
    ) -> float:
        """
        Evaluate accuracy on a task's dataset.

        For associative tasks: accuracy = fraction of inputs where
        prediction is closer to target than to any other target.

        Parameters
        ----------
        data : list of (input, target) pairs
        threshold : float
            Maximum prediction error to count as "correct" (MSE threshold).

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        correct = 0
        total = len(data)

        if total == 0:
            return 0.0

        for s, target in data:
            with torch.no_grad():
                a = self.model.forward(s)
                pred = self.model.D @ a
                error = ((pred - target) ** 2).mean().item()
                if error < threshold:
                    correct += 1

        return correct / total

    def evaluate_all_tasks(
        self,
        all_task_data: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        checkpoint: Optional[int] = None,
        threshold: float = 0.5,
    ) -> List[float]:
        """
        Evaluate all tasks and record in metrics tracker.

        Parameters
        ----------
        all_task_data : list of task datasets
        checkpoint : int, optional
            Evaluation checkpoint index.
        threshold : float
            Accuracy threshold for MSE-based accuracy.

        Returns
        -------
        list of float
            Accuracy per task.
        """
        if self.metrics is not None:
            self.metrics.new_checkpoint()

        accuracies = []
        for task_id, task_data in enumerate(all_task_data):
            acc = self.evaluate_task(task_data, threshold=threshold)
            accuracies.append(acc)

            if self.metrics is not None:
                self.metrics.record_task_accuracy(task_id, acc, checkpoint)

        return accuracies
