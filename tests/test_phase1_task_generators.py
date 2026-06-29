"""
Raw AGNIS — tests/test_phase1_task_generators.py

Unit tests for Phase 1 task generators.
"""

import pytest
import torch
from experiments.phase1_associative.task_generators import generate_phase1_tasks


def test_orthogonal_generator():
    tasks = generate_phase1_tasks(
        condition="orthogonal",
        num_tasks=3,
        pairs_per_task=2,
        d_in=12,
        d_out=12,
    )
    assert len(tasks) == 3
    assert tasks[0].inputs.shape == (2, 12)
    assert tasks[0].targets.shape == (2, 12)
    assert tasks[0].context is None
    
    # Check orthogonality of inputs
    inputs = torch.cat([t.inputs for t in tasks]) # shape (6, 12)
    for i in range(6):
        for j in range(6):
            if i != j:
                assert (inputs[i] @ inputs[j]).item() == 0.0, "Orthogonal inputs overlap!"


def test_overlapping_generator_with_context():
    tasks = generate_phase1_tasks(
        condition="overlapping",
        num_tasks=3,
        pairs_per_task=2,
        d_in=8,
        d_out=12,
        overlap_context=True,
    )
    assert len(tasks) == 3
    assert tasks[0].inputs.shape == (2, 8)
    assert tasks[0].context.shape == (3,)
    assert tasks[0].context[0].item() == 1.0
    assert tasks[1].context[1].item() == 1.0

    # Inputs should be identical across tasks
    assert torch.allclose(tasks[0].inputs, tasks[1].inputs)
    # Targets should be different (orthogonal) across tasks
    t0_targets = tasks[0].targets
    t1_targets = tasks[1].targets
    for i in range(2):
        for j in range(2):
            assert (t0_targets[i] @ t1_targets[j]).item() == 0.0


def test_overlapping_generator_no_context():
    tasks = generate_phase1_tasks(
        condition="overlapping",
        num_tasks=3,
        pairs_per_task=2,
        d_in=8,
        d_out=12,
        overlap_context=False,
    )
    assert len(tasks) == 3
    assert tasks[0].context is None


def test_clustered_generator():
    tasks = generate_phase1_tasks(
        condition="clustered",
        num_tasks=3,
        pairs_per_task=2,
        d_in=10,
        d_out=12,
        clustered_noise=0.01, # low noise to verify prototype grouping
    )
    assert len(tasks) == 3
    assert tasks[0].inputs.shape == (2, 10)
    assert tasks[0].context is None

    # Verify similar inputs across tasks have high cosine similarity (prototype clustering)
    t0_pair0 = tasks[0].inputs[0]
    t1_pair0 = tasks[1].inputs[0]
    sim = (t0_pair0 @ t1_pair0).item() / (t0_pair0.norm().item() * t1_pair0.norm().item() + 1e-8)
    assert sim > 0.9, f"Clustered prototype inputs should be similar, got cosine {sim}"


def test_capacity_stress_generator():
    tasks = generate_phase1_tasks(
        condition="capacity_stress",
        num_tasks=5,
        pairs_per_task=4,
        d_in=20,
        d_out=20,
    )
    assert len(tasks) == 5
    assert tasks[0].inputs.shape == (4, 20)
