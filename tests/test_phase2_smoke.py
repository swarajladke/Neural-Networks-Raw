"""
Raw AGNIS — tests/test_phase2_smoke.py

Unit tests for Phase 2 sequence benchmark smoke runner and evaluation update guard.
"""

import os
import pytest
import torch
import subprocess
from agnis.utils.config import default_config
from agnis.sequence.sequence_wrapper import SeqAgnisModel


def test_sequence_evaluation_no_weight_update():
    # Verify that prediction does not update weights
    config = default_config()
    d_in = 8
    d_out = 8
    model = SeqAgnisModel(d_in, d_out, d_z=8, config=config)
    
    # Save initial weights
    D_init = model.base_model.cell.D.clone()
    E_init = model.base_model.cell.E.clone()
    R_init = model.base_model.cell.R.clone()
    
    x = torch.zeros(d_in)
    x[0] = 1.0
    
    # Predict transition (should be read-only)
    pred = model.predict_transition(x)
    
    # Assert weights did not change
    assert torch.equal(model.base_model.cell.D, D_init)
    assert torch.equal(model.base_model.cell.E, E_init)
    assert torch.equal(model.base_model.cell.R, R_init)


def test_phase2_smoke_benchmark(tmp_path):
    # Run the tiny sequence benchmark smoke test via python
    # We will pass a temporary results directory to prevent polluting the actual results
    benchmark_script = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "phase2_sequences", "run_sequence_benchmark.py"
    )
    config_yaml = os.path.join(
        os.path.dirname(__file__), "..", "configs", "phase2_smoke.yaml"
    )
    
    cmd = [
        "python",
        benchmark_script,
        "--condition", "periodic",
        "--model", "seq_agnis_recurrent",
        "--seed", "0",
        "--config", config_yaml,
        "--smoke",
    ]
    
    # Run process
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Benchmark smoke run failed: {res.stderr}"
    assert "Run finished successfully" in res.stdout
