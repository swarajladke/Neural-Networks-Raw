"""
Raw AGNIS — experiments/phase1_associative/task_generators.py

Generates associative tasks for Phase 1 continual learning benchmark under different conditions:
- orthogonal: disjoint one-hot vectors
- overlapping: reused inputs mapping to task-specific targets (with optional context)
- clustered: inputs perturbed around cluster center prototypes, mapping to discrete targets
- capacity_stress: large number of orthogonal associations mapped under bottleneck constraints
"""

import torch
import random
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AssociationTask:
    task_id: int
    name: str
    inputs: torch.Tensor        # Shape: (pairs_per_task, d_in)
    targets: torch.Tensor       # Shape: (pairs_per_task, d_out)
    context: Optional[torch.Tensor] = None  # Shape: (d_context,) or None
    metadata: dict = field(default_factory=dict)


def make_onehot(idx: int, dim: int) -> torch.Tensor:
    """Helper to create a one-hot vector."""
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


def generate_phase1_tasks(
    condition: str,
    num_tasks: int,
    pairs_per_task: int,
    d_in: int,
    d_out: int,
    overlap_context: bool = True,
    clustered_noise: float = 0.15,
    seed: int = 42,
) -> List[AssociationTask]:
    """
    Generate tasks for Phase 1 benchmark.

    Parameters
    ----------
    condition : str
        orthogonal, overlapping, clustered, capacity_stress
    num_tasks : int
        Number of tasks.
    pairs_per_task : int
        Number of (input, target) associations per task.
    d_in : int
        Input dimension.
    d_out : int
        Target dimension.
    overlap_context : bool
        If True, overlapping tasks receive one-hot context vectors.
    clustered_noise : float
        Noise level (std dev) for similarity-clustered inputs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of AssociationTask
    """
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)

    tasks = []
    d_context = num_tasks if overlap_context else 0

    if condition == "orthogonal":
        # Check vocab constraints
        total_pairs = num_tasks * pairs_per_task
        assert d_in >= total_pairs, f"d_in must be >= {total_pairs} for orthogonal inputs."
        assert d_out >= total_pairs, f"d_out must be >= {total_pairs} for orthogonal targets."

        for t in range(num_tasks):
            inputs_list = []
            targets_list = []
            for p in range(pairs_per_task):
                idx = t * pairs_per_task + p
                inputs_list.append(make_onehot(idx, d_in))
                targets_list.append(make_onehot(idx, d_out))
            
            tasks.append(AssociationTask(
                task_id=t,
                name=f"Orthogonal_Task_{t}",
                inputs=torch.stack(inputs_list),
                targets=torch.stack(targets_list),
                context=None,
                metadata={"condition": "orthogonal"}
            ))

    elif condition == "overlapping":
        total_targets = num_tasks * pairs_per_task
        assert d_in >= pairs_per_task, f"d_in must be >= {pairs_per_task} for overlapping inputs."
        assert d_out >= total_targets, f"d_out must be >= {total_targets} for overlapping targets."

        for t in range(num_tasks):
            inputs_list = []
            targets_list = []
            for p in range(pairs_per_task):
                # Inputs are always in range [0, pairs_per_task)
                inputs_list.append(make_onehot(p, d_in))
                # Targets are task-specific
                idx = t * pairs_per_task + p
                targets_list.append(make_onehot(idx, d_out))
            
            # Context
            context_vec = None
            if overlap_context:
                context_vec = make_onehot(t, d_context)

            tasks.append(AssociationTask(
                task_id=t,
                name=f"Overlapping_Task_{t}",
                inputs=torch.stack(inputs_list),
                targets=torch.stack(targets_list),
                context=context_vec,
                metadata={"condition": "overlapping", "uses_context": overlap_context}
            ))

    elif condition == "clustered":
        # Clustered tasks: inputs are perturbed cluster centers
        total_targets = num_tasks * pairs_per_task
        assert d_out >= total_targets, f"d_out must be >= {total_targets} for clustered targets."

        # Create C base prototypes (C = pairs_per_task)
        prototypes = []
        for p in range(pairs_per_task):
            proto = torch.randn(d_in)
            # Normalize to unit norm
            proto = proto / (proto.norm() + 1e-8)
            prototypes.append(proto)

        for t in range(num_tasks):
            inputs_list = []
            targets_list = []
            for p in range(pairs_per_task):
                # Input is prototype + noise
                noise = torch.randn(d_in) * clustered_noise
                inp = prototypes[p] + noise
                inp = inp / (inp.norm() + 1e-8)  # keep unit norm
                inputs_list.append(inp)
                
                # Target is discrete/one-hot
                idx = t * pairs_per_task + p
                targets_list.append(make_onehot(idx, d_out))

            tasks.append(AssociationTask(
                task_id=t,
                name=f"Clustered_Task_{t}",
                inputs=torch.stack(inputs_list),
                targets=torch.stack(targets_list),
                context=None,
                metadata={"condition": "clustered", "noise_std": clustered_noise}
            ))

    elif condition == "capacity_stress":
        # Force orthogonal task generation but with large number of tasks/pairs
        total_pairs = num_tasks * pairs_per_task
        assert d_in >= total_pairs, f"d_in must be >= {total_pairs} for capacity_stress."
        assert d_out >= total_pairs, f"d_out must be >= {total_pairs} for capacity_stress."

        for t in range(num_tasks):
            inputs_list = []
            targets_list = []
            for p in range(pairs_per_task):
                idx = t * pairs_per_task + p
                inputs_list.append(make_onehot(idx, d_in))
                targets_list.append(make_onehot(idx, d_out))
            
            tasks.append(AssociationTask(
                task_id=t,
                name=f"Stress_Task_{t}",
                inputs=torch.stack(inputs_list),
                targets=torch.stack(targets_list),
                context=None,
                metadata={"condition": "capacity_stress"}
            ))

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return tasks
