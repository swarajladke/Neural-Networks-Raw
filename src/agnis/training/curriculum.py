"""
Raw AGNIS — src/agnis/training/curriculum.py

Curriculum: Task sequencing and ordering utilities.

Provides:
- TaskDataset: one-hot symbolic task dataset builder for Phase 1
- ContinualCurriculum: manages sequential task presentation
"""

import torch
from typing import List, Tuple, Optional, Dict


def make_onehot(idx: int, n: int) -> torch.Tensor:
    """Create a one-hot vector of size n with a 1 at position idx."""
    v = torch.zeros(n)
    v[idx] = 1.0
    return v


class AssociativeTask:
    """
    A single associative mapping task.

    Each task consists of (input, target) pairs where inputs and targets
    are one-hot vectors from a shared vocabulary.

    Example:
        Task 1: A→B, C→D
        Using vocabulary size 12 (A=0, B=1, C=2, D=3, ...):
        pairs = [(onehot(0,12), onehot(1,12)), (onehot(2,12), onehot(3,12))]
    """

    def __init__(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        task_id: int,
        name: str = "",
    ):
        self.pairs = pairs
        self.task_id = task_id
        self.name = name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def build_phase1_tasks(vocab_size: int = 12) -> List[AssociativeTask]:
    """
    Build the Phase 1 associative continual learning tasks.

    Tasks:
        Task 0: A→B, C→D   (indices: 0→1, 2→3)
        Task 1: E→F, G→H   (indices: 4→5, 6→7)
        Task 2: I→J, K→L   (indices: 8→9, 10→11)

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size. Must be >= 12 for Phase 1.

    Returns
    -------
    list of AssociativeTask
    """
    assert vocab_size >= 12, "Phase 1 requires at least 12 symbols."

    tasks = []
    task_specs = [
        # (input_idx, target_idx) pairs per task
        [(0, 1), (2, 3)],    # Task 0: A->B, C->D
        [(4, 5), (6, 7)],    # Task 1: E->F, G->H
        [(8, 9), (10, 11)],  # Task 2: I->J, K->L
    ]
    task_names = ["Task0:A->B,C->D", "Task1:E->F,G->H", "Task2:I->J,K->L"]

    for task_id, (spec, name) in enumerate(zip(task_specs, task_names)):
        pairs = [
            (make_onehot(inp, vocab_size), make_onehot(tgt, vocab_size))
            for inp, tgt in spec
        ]
        tasks.append(AssociativeTask(pairs=pairs, task_id=task_id, name=name))

    return tasks


class ContinualCurriculum:
    """
    Manages sequential task presentation for continual learning experiments.

    Parameters
    ----------
    tasks : list of AssociativeTask
        All tasks in presentation order.
    n_repeats_per_task : int
        How many times to present each (input, target) pair per task epoch.
    """

    def __init__(
        self,
        tasks: List[AssociativeTask],
        n_repeats_per_task: int = 20,
    ):
        self.tasks = tasks
        self.n_repeats = n_repeats_per_task
        self._current_task_idx = 0

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def current_task(self) -> Optional[AssociativeTask]:
        if self._current_task_idx < self.n_tasks:
            return self.tasks[self._current_task_idx]
        return None

    def get_task_data(
        self, task_idx: int, n_repeats: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the repeated training data for a specific task.

        Parameters
        ----------
        task_idx : int
        n_repeats : int, optional
            Overrides self.n_repeats if provided.

        Returns
        -------
        list of (input, target) pairs
        """
        task = self.tasks[task_idx]
        reps = n_repeats if n_repeats is not None else self.n_repeats
        data = []
        for _ in range(reps):
            for s, t in task.pairs:
                data.append((s.clone(), t.clone()))
        return data

    def all_tasks_data(self, n_repeats: Optional[int] = None) -> List[List[Tuple]]:
        """Return training data for all tasks."""
        return [self.get_task_data(i, n_repeats) for i in range(self.n_tasks)]
