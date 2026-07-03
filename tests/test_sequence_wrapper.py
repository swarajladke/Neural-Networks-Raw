"""
Raw AGNIS — tests/test_sequence_wrapper.py

Unit tests for sequence models and baselines wrappers.
"""

import pytest
import torch
from agnis.utils.config import default_config
from agnis.sequence.sequence_wrapper import (
    SeqAgnisModel,
    SimpleRNNBaseline,
    SimpleGRUBaseline,
    MLPWindowBaseline,
)


def test_sequence_wrappers_training_and_reset():
    config = default_config()
    d_in = 8
    d_out = 8
    
    models = [
        SeqAgnisModel(d_in, d_out, d_z=8, config=config),
        SimpleRNNBaseline(d_in, d_out, d_hidden=8),
        SimpleGRUBaseline(d_in, d_out, d_hidden=8),
        MLPWindowBaseline(d_in, d_out, context_window=2),
    ]
    
    for model in models:
        model.reset_sequence_state()
        
        x = torch.zeros(d_in)
        x[0] = 1.0
        y = torch.zeros(d_out)
        y[1] = 1.0
        
        # Train transition
        metrics = model.train_transition(x, y)
        assert "error" in metrics
        
        # Predict transition
        pred = model.predict_transition(x)
        assert pred.shape == (d_out,)
        assert abs(pred.sum().item() - 1.0) < 1e-4  # probability distribution
        
        # Reset state works
        model.reset_sequence_state()
