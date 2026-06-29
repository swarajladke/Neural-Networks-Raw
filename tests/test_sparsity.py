"""
Raw AGNIS — tests/test_sparsity.py

Unit tests for kWTA sparsity mechanism.

Tests:
  1. test_kwta_keeps_exactly_k_active
  2. test_kwta_activates_largest_magnitude
  3. test_kwta_batch_shapes
  4. test_kwta_dense_mode
  5. test_sparsity_level_correct
  6. test_kwta_all_zero_input
"""

import pytest
import torch
from agnis.core.sparsity import kwta, kwta_batch, compute_sparsity_level, compute_active_fraction


def test_kwta_keeps_exactly_k_active():
    """kWTA should keep exactly k units active."""
    a = torch.randn(20)
    k = 5
    sparse = kwta(a, k)
    n_active = (sparse != 0).sum().item()
    assert n_active == k, f"Expected {k} active units, got {n_active}"


def test_kwta_activates_largest_magnitude():
    """kWTA should keep the units with the largest absolute values."""
    a = torch.tensor([0.1, 0.5, -0.9, 0.3, -0.2, 0.8])
    k = 2
    sparse = kwta(a, k)
    # Top-2 by magnitude: -0.9 and 0.8
    assert sparse[2].item() != 0, "Unit 2 (-0.9) should be active"
    assert sparse[5].item() != 0, "Unit 5 (0.8) should be active"
    assert sparse[0].item() == 0, "Unit 0 (0.1) should be inactive"


def test_kwta_values_unchanged_for_active():
    """kWTA should not modify the values of active units."""
    a = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.1])
    k = 2
    sparse = kwta(a, k)
    # Top-2: indices 3 (3.0) and 1 (2.0)
    assert sparse[3].item() == pytest.approx(3.0), "Active unit value should be preserved"
    assert sparse[1].item() == pytest.approx(2.0), "Active unit value should be preserved"


def test_kwta_dense_mode():
    """k >= d_z should return unchanged activations (dense mode)."""
    a = torch.randn(10)
    dense = kwta(a, k=10)
    assert torch.allclose(a, dense), "Dense mode should return unchanged activations"


def test_kwta_dense_mode_larger_k():
    """k > d_z should also return unchanged (safe)."""
    a = torch.randn(10)
    dense = kwta(a, k=100)
    assert torch.allclose(a, dense), "k > d_z should return unchanged activations"


def test_kwta_k_zero_all_suppressed():
    """k=1 should keep only 1 unit. k behavior at boundary."""
    a = torch.tensor([0.5, 1.0, 0.3, 0.8, 0.2])
    sparse = kwta(a, k=1)
    n_active = (sparse != 0).sum().item()
    assert n_active == 1, f"Expected 1 active unit, got {n_active}"
    assert sparse[1].item() == pytest.approx(1.0), "Highest magnitude should survive"


def test_kwta_batch_shape():
    """kwta_batch should return same shape as input."""
    a = torch.randn(8, 20)
    sparse = kwta_batch(a, k=5)
    assert sparse.shape == a.shape, f"Shape mismatch: {sparse.shape}"


def test_kwta_batch_per_row_k_active():
    """kwta_batch: each row should have exactly k active units."""
    a = torch.randn(8, 20)
    k = 5
    sparse = kwta_batch(a, k)
    for i in range(8):
        n_active = (sparse[i] != 0).sum().item()
        assert n_active == k, f"Row {i}: expected {k} active, got {n_active}"


def test_sparsity_level_full_zeros():
    """All-zero tensor → sparsity = 1.0."""
    a = torch.zeros(20)
    assert compute_sparsity_level(a) == pytest.approx(1.0)


def test_sparsity_level_all_active():
    """All-nonzero tensor → sparsity = 0.0."""
    a = torch.ones(20)
    assert compute_sparsity_level(a) == pytest.approx(0.0)


def test_sparsity_level_after_kwta():
    """After kWTA with k=5 on d=20, sparsity should be 75% (15/20 = 0.75)."""
    a = torch.randn(20)
    sparse = kwta(a, k=5)
    sparsity = compute_sparsity_level(sparse)
    assert sparsity == pytest.approx(0.75), f"Expected 0.75, got {sparsity}"


def test_active_fraction_complement_of_sparsity():
    """active_fraction + sparsity_level should equal 1.0."""
    a = torch.randn(20)
    sparse = kwta(a, k=7)
    sparsity = compute_sparsity_level(sparse)
    active = compute_active_fraction(sparse)
    assert abs(sparsity + active - 1.0) < 1e-6, "sparsity + active should = 1.0"


def test_kwta_no_nan_output():
    """kWTA on NaN-free input should not produce NaN."""
    a = torch.randn(16)
    sparse = kwta(a, k=4)
    assert not torch.isnan(sparse).any(), "kWTA output should not contain NaN"
