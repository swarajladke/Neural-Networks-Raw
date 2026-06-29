"""
Raw AGNIS — tests/test_forgetting.py

Unit tests for the forgetting metric computation.

Tests:
  1. test_forgetting_formula_basic
  2. test_forgetting_zero_for_last_task
  3. test_forgetting_zero_when_no_forgetting
  4. test_forgetting_tracker_records_correctly
  5. test_average_forgetting_correct
  6. test_backward_transfer_negative_when_forgetting
"""

import pytest
from agnis.evaluation.forgetting import (
    compute_forgetting,
    compute_average_forgetting,
    compute_backward_transfer,
    ForgettingTracker,
)


def test_forgetting_formula_basic():
    """
    F_i = max_prev_acc_i - current_acc_i

    Example:
        Task 0 accuracies over checkpoints: [0.9, 0.7, 0.65]
        Peak = 0.9, Final = 0.65 → F_0 = 0.25

        Task 1 accuracies over checkpoints: [None, 0.85, 0.80]
        Peak = 0.85, Final = 0.80 → F_1 = 0.05
    """
    accuracy_matrix = [
        [0.9, None, None],   # checkpoint 0 (after task 0)
        [0.7, 0.85, None],   # checkpoint 1 (after task 1)
        [0.65, 0.80, 0.92],  # checkpoint 2 (after task 2)
    ]
    task_order = [0, 1, 2]
    forgetting = compute_forgetting(accuracy_matrix, task_order)

    assert len(forgetting) == 2, "Should return forgetting for tasks 0 and 1 (not last)"
    assert abs(forgetting[0] - 0.25) < 1e-5, f"F_0 should be 0.25, got {forgetting[0]}"
    assert abs(forgetting[1] - 0.05) < 1e-5, f"F_1 should be 0.05, got {forgetting[1]}"


def test_forgetting_zero_when_no_forgetting():
    """If accuracy never decreases, forgetting should be 0."""
    accuracy_matrix = [
        [0.8, None, None],
        [0.85, 0.9, None],
        [0.9, 0.92, 0.95],
    ]
    task_order = [0, 1, 2]
    forgetting = compute_forgetting(accuracy_matrix, task_order)
    for f in forgetting:
        assert f <= 0.0 + 1e-5, f"No-forgetting scenario should give F <= 0, got {f}"


def test_forgetting_zero_for_last_task():
    """Forgetting is not computed for the last task (no post-training evaluation)."""
    accuracy_matrix = [
        [0.9, None],
        [0.7, 0.8],
    ]
    task_order = [0, 1]
    forgetting = compute_forgetting(accuracy_matrix, task_order)
    # Only 1 task forgetting (task 0), not 2
    assert len(forgetting) == 1, f"Expected 1 forgetting value, got {len(forgetting)}"


def test_forgetting_empty_matrix():
    """Empty matrix should return empty forgetting list."""
    forgetting = compute_forgetting([], [])
    assert forgetting == [], "Empty matrix should give empty forgetting"


def test_average_forgetting_correct():
    """Average forgetting should be the mean of per-task forgetting values."""
    forgetting = [0.25, 0.05, 0.10]
    avg = compute_average_forgetting(forgetting)
    expected = (0.25 + 0.05 + 0.10) / 3
    assert abs(avg - expected) < 1e-6, f"Avg forgetting wrong: {avg} vs {expected}"


def test_average_forgetting_empty():
    """Average forgetting of empty list should be 0."""
    assert compute_average_forgetting([]) == 0.0


def test_backward_transfer_negative_on_forgetting():
    """BWT should be negative when there is forgetting."""
    accuracy_matrix = [
        [0.9, None, None],
        [0.7, 0.85, None],
        [0.65, 0.80, 0.92],
    ]
    bwt = compute_backward_transfer(accuracy_matrix)
    assert bwt < 0, f"BWT should be negative when forgetting occurs, got {bwt}"


def test_forgetting_tracker_records_and_computes():
    """ForgettingTracker should correctly record and compute forgetting."""
    tracker = ForgettingTracker(n_tasks=3)

    # Checkpoint 0: after training task 0
    tracker.new_checkpoint()
    tracker.record(task_id=0, accuracy=0.9)

    # Checkpoint 1: after training task 1
    tracker.new_checkpoint()
    tracker.record(task_id=0, accuracy=0.7)
    tracker.record(task_id=1, accuracy=0.85)

    # Checkpoint 2: after training task 2
    tracker.new_checkpoint()
    tracker.record(task_id=0, accuracy=0.65)
    tracker.record(task_id=1, accuracy=0.80)
    tracker.record(task_id=2, accuracy=0.92)

    forgetting = tracker.forgetting
    assert len(forgetting) == 2, f"Expected 2 forgetting values, got {len(forgetting)}"
    assert abs(forgetting[0] - 0.25) < 1e-5, f"F_0 should be 0.25, got {forgetting[0]}"
    assert abs(forgetting[1] - 0.05) < 1e-5, f"F_1 should be 0.05, got {forgetting[1]}"

    avg = tracker.avg_forgetting
    assert abs(avg - (0.25 + 0.05) / 2) < 1e-5, f"Avg forgetting wrong: {avg}"


def test_forgetting_tracker_summary():
    """Summary dict should contain all expected keys."""
    tracker = ForgettingTracker(n_tasks=2)
    tracker.new_checkpoint()
    tracker.record(task_id=0, accuracy=0.8)
    tracker.new_checkpoint()
    tracker.record(task_id=0, accuracy=0.7)
    tracker.record(task_id=1, accuracy=0.9)

    summary = tracker.summary()
    assert "n_tasks" in summary
    assert "per_task_forgetting" in summary
    assert "avg_forgetting" in summary
    assert "backward_transfer" in summary
    assert "final_accuracies" in summary
