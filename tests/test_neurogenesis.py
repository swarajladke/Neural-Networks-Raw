"""
Raw AGNIS — tests/test_neurogenesis.py

Unit tests for neurogenesis components.

Tests:
  1. test_growth_controller_does_not_trigger_early
  2. test_growth_controller_triggers_after_n_consecutive
  3. test_maturity_starts_at_zero_for_new_unit
  4. test_maturity_increases_with_error_reduction
  5. test_maturity_gate_zeros_contribution
  6. test_pruning_candidates_found
  7. test_redundancy_computation
"""

import pytest
import torch
from agnis.neurogenesis.growth_controller import GrowthController
from agnis.neurogenesis.maturity import MaturityTracker
from agnis.neurogenesis.pruning import (
    compute_redundancy,
    find_prune_candidates,
)


# ── GrowthController Tests ────────────────────────────────────────────────────

def test_growth_controller_does_not_trigger_early():
    """Should not trigger before N consecutive above-threshold observations."""
    ctrl = GrowthController(threshold=0.5, consecutive_n=5)
    # Feed 4 high-score updates (below N=5)
    for _ in range(4):
        triggered = ctrl.update(
            error=1.0, novelty=1.0, uncertainty=0.5,
            interference=0.5, coverage=0.0, cost=0.0
        )
        assert not triggered, "Should not trigger before N consecutive observations"


def test_growth_controller_triggers_after_n_consecutive():
    """Should trigger after N consecutive above-threshold observations."""
    ctrl = GrowthController(threshold=0.5, consecutive_n=5, ema_alpha=1.0)
    # Feed 5 high-score updates: this uses alpha=1.0 so G = raw score
    triggered = False
    for i in range(10):
        t = ctrl.update(
            error=1.0, novelty=1.0, uncertainty=0.5,
            interference=0.5, coverage=0.0, cost=0.0
        )
        if t:
            triggered = True
            break
    assert triggered, "Should trigger after N consecutive high-score observations"


def test_growth_controller_resets_count_on_drop():
    """Consecutive count should reset when G drops below threshold."""
    ctrl = GrowthController(threshold=0.5, consecutive_n=5, ema_alpha=1.0)
    # 3 high, 1 low, then need N more high to trigger
    for _ in range(3):
        ctrl.update(error=1.0, novelty=1.0, uncertainty=0.5, interference=0.5, coverage=0.0, cost=0.0)
    # Drop below threshold
    ctrl.update(error=0.0, novelty=0.0, uncertainty=0.0, interference=0.0, coverage=1.0, cost=1.0)
    # Count should be reset; trigger should not fire immediately
    triggered = ctrl.update(error=1.0, novelty=1.0, uncertainty=0.5, interference=0.5, coverage=0.0, cost=0.0)
    assert not triggered, "After a drop, single high-score should not trigger"


# ── MaturityTracker Tests ─────────────────────────────────────────────────────

def test_maturity_initial_units_start_mature():
    """Original units (n_initial) start with maturity=1.0."""
    tracker = MaturityTracker(n_initial_units=10)
    assert (tracker.maturity == 1.0).all(), "Initial units should have maturity=1.0"


def test_maturity_new_unit_starts_at_zero():
    """Added units start with maturity=0.0."""
    tracker = MaturityTracker(n_initial_units=5)
    tracker.add_unit()
    assert tracker.maturity[-1].item() == pytest.approx(0.0), \
        "New unit should start with maturity=0"


def test_maturity_increases_with_error_reduction():
    """Maturity should increase when unit reduces prediction error."""
    tracker = MaturityTracker(n_initial_units=5, eta_maturity=0.5)
    tracker.add_unit()   # unit index 5
    unit_idx = 5

    initial_maturity = tracker.maturity[unit_idx].item()
    tracker.update(unit_idx, error_before=1.0, error_after=0.5)
    new_maturity = tracker.maturity[unit_idx].item()

    assert new_maturity > initial_maturity, "Maturity should increase with error reduction"


def test_maturity_does_not_change_without_improvement():
    """Maturity should not change when there is no error reduction."""
    tracker = MaturityTracker(n_initial_units=5, eta_maturity=0.5)
    tracker.add_unit()
    unit_idx = 5

    initial_maturity = tracker.maturity[unit_idx].item()
    tracker.update(unit_idx, error_before=0.5, error_after=0.8)  # error increased!
    new_maturity = tracker.maturity[unit_idx].item()

    assert new_maturity == pytest.approx(initial_maturity), \
        "Maturity should not change when error increases"


def test_maturity_capped_at_one():
    """Maturity should never exceed 1.0."""
    tracker = MaturityTracker(n_initial_units=1, eta_maturity=100.0)
    tracker.update(0, error_before=10.0, error_after=0.0)
    assert tracker.maturity[0].item() <= 1.0, "Maturity should be capped at 1.0"


def test_maturity_gate_zeros_new_unit():
    """New unit with maturity=0 should contribute nothing (effective activation = 0)."""
    tracker = MaturityTracker(n_initial_units=4)
    tracker.add_unit()  # unit 4: maturity=0

    a = torch.ones(5)
    gated = tracker.get_effective_activations(a)

    assert gated[4].item() == pytest.approx(0.0), "New unit contribution should be 0"
    assert gated[:4].sum().item() > 0, "Original units should still contribute"


def test_maturity_remove_unit():
    """Removing a unit should reduce the tracker size by 1."""
    tracker = MaturityTracker(n_initial_units=5)
    tracker.remove_unit(2)
    assert len(tracker.maturity) == 4, f"Expected 4 units, got {len(tracker.maturity)}"


# ── Pruning Tests ─────────────────────────────────────────────────────────────

def test_redundancy_zero_for_unique_unit():
    """A unique unit should have low redundancy."""
    D = torch.eye(8)   # Each column is a different basis vector
    redundancy = compute_redundancy(D, unit_idx=0)
    assert redundancy < 0.1, f"Unique unit should have low redundancy, got {redundancy}"


def test_redundancy_high_for_duplicate_unit():
    """A duplicate unit should have redundancy near 1."""
    D = torch.zeros(8, 4)
    D[:, 0] = torch.randn(8)
    D[:, 1] = D[:, 0].clone()  # exact duplicate
    D[:, 2] = torch.randn(8)
    D[:, 3] = torch.randn(8)

    redundancy = compute_redundancy(D, unit_idx=0)
    assert redundancy > 0.99, f"Duplicate unit should have redundancy ~1.0, got {redundancy}"


def test_pruning_candidates_found():
    """Units that are low-usage, low-importance, and high-redundancy should be found."""
    d_in, d_z = 8, 4
    D = torch.zeros(d_in, d_z)
    D[:, 0] = torch.randn(d_in)
    D[:, 1] = D[:, 0].clone()   # redundant with unit 0
    D[:, 2] = torch.randn(d_in)
    D[:, 3] = torch.randn(d_in)

    usage = torch.tensor([0.8, 0.01, 0.7, 0.6])      # unit 1: low usage
    importance = torch.zeros(d_in, d_z)               # all low importance
    importance[:, 0] = 1.0                            # unit 0 is important

    candidates = find_prune_candidates(
        usage=usage,
        importance=importance,
        D=D,
        usage_threshold=0.05,
        importance_threshold=0.01,
        redundancy_threshold=0.90,
    )

    assert 1 in candidates, "Unit 1 (low usage + low importance + high redundancy) should be a prune candidate"
    assert 0 not in candidates, "Unit 0 (high importance) should NOT be a prune candidate"
