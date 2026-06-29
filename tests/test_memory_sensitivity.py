"""
Raw AGNIS — tests/test_memory_sensitivity.py

Unit tests for memory and replay activation sensitivity under low thresholds.
"""

import pytest
import torch
from agnis.evaluation.baselines import AgnisBaseline
from agnis.evaluation.continual_metrics import evaluate_model_on_task


def test_memory_activation_under_low_threshold():
    # Setup model with low write_error_threshold to guarantee writes
    model = AgnisBaseline(
        d_in=8,
        d_out=8,
        d_z=16,
        use_sparsity=True,
        use_memory=True,
        use_replay=True,
        write_error_threshold=0.01, # extremely low, error will exceed this
        write_novelty_threshold=0.01,
        replay_buffer_size=10,
        n_sleep_steps=2,
        n_sleep_replay=5,
    )
    
    # Check start task resets tracking
    model.start_task(0)
    assert model.current_task_idx == 0
    assert len(model.memory_writes_per_task) == 1
    
    # Train some pairs that have high error initially
    # Since d_in=8, joint_dim=16
    x = torch.randn(8)
    y = torch.randn(8)
    
    metrics = model.train_pair(x, y)
    
    # Check that writes occurred
    stats = model.get_stats()
    assert stats["memory_writes_per_task"][0] > 0, "No memory writes occurred at low threshold!"
    assert stats["memory_size"] > 0, "Memory size did not increase!"
    
    # Execute sleep Consolidation
    sleep_stats = model.sleep()
    assert sleep_stats.get("sleep_steps", 0) > 0, "Sleep trainer did not execute updates!"
    
    stats_after_sleep = model.get_stats()
    assert stats_after_sleep["replay_steps_executed"] > 0, "Replay steps were not logged!"
