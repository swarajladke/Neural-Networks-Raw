"""
Raw AGNIS — tests/test_representation_overlap.py

Unit tests for task prototype representations, overlap, and context disambiguation.
"""

import pytest
import torch
from agnis.evaluation.representation import (
    compute_task_prototypes,
    compute_pairwise_overlap,
    compute_interference_score,
)
from agnis.evaluation.baselines import AgnisBaseline


def test_representation_overlap_calculation():
    # Prototypes
    p0 = torch.tensor([1.0, 0.0, 0.0])
    p1 = torch.tensor([0.0, 1.0, 0.0])
    p2 = torch.tensor([0.7071, 0.7071, 0.0]) # 45 degrees
    
    overlaps = compute_pairwise_overlap([p0, p1, p2])
    
    # cos(p0, p1) = 0
    assert abs(overlaps[0, 1].item()) < 1e-6
    # cos(p0, p2) = 0.7071
    assert abs(overlaps[0, 2].item() - 0.7071) < 1e-4


def test_interference_score_calculation():
    # Create a mock AgnisBaseline
    model = AgnisBaseline(d_in=4, d_out=4, d_z=8, use_memory=True)
    
    # Manually configure cell's importance_D to make unit 0 highly important
    model.cell.importance_D = torch.zeros(8, 8)
    model.cell.importance_D[0, 0] = 5.0 # unit 0 is extremely important
    
    # New task prototype activating unit 0
    proto_interfering = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # New task prototype activating unit 1 (safe)
    proto_safe = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    score_interfering = compute_interference_score(model, proto_interfering)
    score_safe = compute_interference_score(model, proto_safe)
    
    assert score_interfering > 0.9, f"Expected high interference score, got {score_interfering}"
    assert score_safe < 0.1, f"Expected low interference score, got {score_safe}"


def test_overlapping_tasks_with_context():
    # Verifies same inputs combined with different task context vectors yield different latent representations
    model = AgnisBaseline(d_in=6, d_out=4, d_z=8, use_memory=False)
    
    x = torch.tensor([1.0, 0.0, 0.0, 0.0]) # input
    ctx0 = torch.tensor([1.0, 0.0])         # context 0
    ctx1 = torch.tensor([0.0, 1.0])         # context 1
    
    # Set y to None to predict target completions
    a0 = model.get_latent(x, y=None, task_context=ctx0)
    a1 = model.get_latent(x, y=None, task_context=ctx1)
    
    # Latent representations must differ since context vectors differ
    assert not torch.allclose(a0, a1), "Context must disambiguate latent states!"
