"""
Raw AGNIS — tests/test_phase2_metrics.py

Unit tests for next-symbol accuracy and condition-specific temporal consistency.
"""

import pytest
import torch
from agnis.utils.config import default_config
from agnis.sequence.sequence_tasks import SequenceTask
from agnis.sequence.sequence_wrapper import SimpleRNNBaseline
from agnis.sequence.temporal_metrics import (
    evaluate_model_on_sequence_task,
    compute_temporal_consistency,
)


def test_next_symbol_accuracy_and_consistency():
    d_in = 4
    d_out = 4
    model = SimpleRNNBaseline(d_in, d_out, d_hidden=8)
    
    # Mock task
    sequences = [[0, 1, 2, 3], [0, 1, 2, 3]]
    task = SequenceTask(name="mock_task", sequences=sequences, vocab_size=4, task_id=0)
    
    # Check evaluation
    acc = evaluate_model_on_sequence_task(model, task.sequences, vocab_size=4)
    assert 0.0 <= acc <= 1.0
    
    # Check temporal consistency under periodic
    cons = compute_temporal_consistency(model, "periodic", task, vocab_size=4)
    assert 0.0 <= cons <= 1.0
