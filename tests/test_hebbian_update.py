"""
Raw AGNIS — tests/test_hebbian_update.py

Unit tests for local Hebbian update rules.

Tests:
  1. test_hebbian_generative_update_changes_weights
  2. test_hebbian_recognition_update_changes_weights
  3. test_hebbian_recurrent_update_changes_weights
  4. test_hebbian_update_zero_with_inactive_units  (sparsity benefit)
  5. test_importance_update_increases
  6. test_plasticity_shape_and_range
"""

import pytest
import torch
from agnis.core.hebbian_rules import (
    hebbian_generative_update,
    hebbian_recognition_update,
    hebbian_recurrent_update,
    plasticity_gated_update,
    compute_plasticity,
    update_importance,
    normalize_weights,
)


def test_hebbian_generative_update_shape():
    """ΔD should have shape (d_in, d_z)."""
    d_in, d_z = 8, 16
    e = torch.randn(d_in)
    a = torch.randn(d_z)
    delta_D = hebbian_generative_update(e, a, eta_D=0.01)
    assert delta_D.shape == (d_in, d_z), f"ΔD shape: {delta_D.shape}"


def test_hebbian_generative_update_changes_weights():
    """ΔD should be non-zero when e and a are non-zero."""
    e = torch.randn(8)
    a = torch.randn(16)
    delta_D = hebbian_generative_update(e, a, eta_D=0.01)
    assert delta_D.norm().item() > 0, "ΔD should be non-zero"


def test_hebbian_generative_zero_with_inactive_units():
    """When a unit is inactive (a_j = 0), its column in ΔD should be zero."""
    e = torch.randn(8)
    a = torch.zeros(16)
    a[3] = 1.5  # only unit 3 is active
    delta_D = hebbian_generative_update(e, a, eta_D=0.01)

    # All columns except column 3 should be zero
    for j in range(16):
        if j != 3:
            assert torch.allclose(delta_D[:, j], torch.zeros(8)), \
                f"Column {j} should be zero (unit inactive)"
    # Column 3 should be non-zero
    assert delta_D[:, 3].norm().item() > 0, "Active unit's column should be non-zero"


def test_hebbian_recognition_update_shape():
    """ΔE should have shape (d_z, d_in)."""
    d_in, d_z = 8, 16
    z = torch.randn(d_z)
    E = torch.randn(d_z, d_in)
    s = torch.randn(d_in)
    delta_E = hebbian_recognition_update(z, E, s, eta_E=0.01)
    assert delta_E.shape == (d_z, d_in), f"ΔE shape: {delta_E.shape}"


def test_hebbian_recognition_update_changes_weights():
    """ΔE should be non-zero."""
    z = torch.randn(16)
    E = torch.zeros(16, 8)   # zero E so error = z - 0 = z
    s = torch.randn(8)
    delta_E = hebbian_recognition_update(z, E, s, eta_E=0.01)
    assert delta_E.norm().item() > 0, "ΔE should be non-zero"


def test_hebbian_recognition_zero_when_no_error():
    """ΔE should be near zero when z ≈ E@s (no recognition error)."""
    d_in, d_z = 8, 16
    s = torch.randn(d_in)
    E = torch.randn(d_z, d_in)
    z = E @ s  # set z exactly equal to E@s → recognition error = 0
    delta_E = hebbian_recognition_update(z, E, s, eta_E=0.1)
    assert delta_E.norm().item() < 1e-5, f"ΔE should be ~0 when z = E@s, got norm {delta_E.norm()}"


def test_hebbian_recurrent_update_shape():
    """ΔR should have shape (d_z, d_z)."""
    z = torch.randn(16)
    z_prev = torch.randn(16)
    delta_R = hebbian_recurrent_update(z, z_prev, eta_R=0.005)
    assert delta_R.shape == (16, 16), f"ΔR shape: {delta_R.shape}"


def test_hebbian_recurrent_update_changes_weights():
    """ΔR should be non-zero when z and z_prev are non-zero."""
    z = torch.randn(16)
    z_prev = torch.randn(16)
    delta_R = hebbian_recurrent_update(z, z_prev, eta_R=0.005)
    assert delta_R.norm().item() > 0, "ΔR should be non-zero"


def test_importance_update_increases():
    """Importance should increase when weights change."""
    importance = torch.zeros(8, 16)
    delta_W = torch.randn(8, 16)
    updated = update_importance(importance, delta_W, alpha_I=0.1)
    assert (updated >= importance).all(), "Importance should not decrease after update"
    assert updated.sum().item() > 0, "Importance should increase from zero"


def test_importance_update_shape():
    """update_importance should preserve tensor shape."""
    importance = torch.zeros(8, 16)
    delta_W = torch.randn(8, 16)
    updated = update_importance(importance, delta_W)
    assert updated.shape == importance.shape


def test_plasticity_range():
    """Plasticity values should be in (0, 1) — sigmoid output."""
    importance = torch.rand(8, 16)
    age = torch.rand(8, 16)
    plasticity = compute_plasticity(
        novelty=0.5, importance=importance, age=age, uncertainty=0.1
    )
    assert (plasticity >= 0).all(), "Plasticity should be >= 0"
    assert (plasticity <= 1).all(), "Plasticity should be <= 1"
    assert plasticity.shape == importance.shape


def test_plasticity_decreases_with_importance():
    """Higher importance should yield lower plasticity (protection mechanism)."""
    low_imp = torch.zeros(1)
    high_imp = torch.ones(1) * 5.0
    age = torch.zeros(1)
    plast_low = compute_plasticity(novelty=0.5, importance=low_imp, age=age)
    plast_high = compute_plasticity(novelty=0.5, importance=high_imp, age=age)
    assert plast_low.item() > plast_high.item(), \
        "Higher importance should give lower plasticity"


def test_plasticity_gated_update_shape():
    """Plasticity-gated update should return correct shape."""
    d_pre, d_post = 8, 16
    pre = torch.randn(d_pre)
    post_error = torch.randn(d_post)
    plasticity = torch.rand(d_post, d_pre)
    delta = plasticity_gated_update(pre, post_error, plasticity, eta=0.01)
    assert delta.shape == (d_post, d_pre), f"Wrong shape: {delta.shape}"


def test_normalize_weights():
    """Normalized weight columns should have unit norm."""
    W = torch.randn(8, 16)
    W_norm = normalize_weights(W, dim=0)
    col_norms = W_norm.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones(16), atol=1e-5), \
        "Normalized columns should have unit norm"
