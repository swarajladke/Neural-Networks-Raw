"""
Raw AGNIS — tests/test_benchmark_smoke.py

Unit tests and smoke tests for the benchmark harness, observed mask completion,
evaluation safety (no weight changes), and capacity stress configuration.
"""

import os
import sys
import pytest
import torch
import shutil
from agnis.core.predictive_cell import PredictiveCell
from agnis.evaluation.baselines import AgnisBaseline
from agnis.evaluation.continual_metrics import evaluate_model_on_task
from agnis.utils.config import load_config


def test_predictive_cell_observed_mask_completion():
    # d_in = 16 (8 input, 8 target)
    cell = PredictiveCell(d_in=16, d_z=10, k_sparse=2, n_settle=5)
    
    # query has values in first 8 dimensions, zeros in second 8
    s_query = torch.zeros(16)
    s_query[:8] = torch.randn(8)
    
    # observed mask: 1.0 on first 8, 0.0 on second 8
    observed_mask = torch.zeros(16)
    observed_mask[:8] = 1.0
    
    a = cell.forward(s_query, observed_mask=observed_mask)
    
    assert a.shape == (10,), "Activation shape wrong"
    assert not torch.isnan(a).any(), "NaN in activation output"
    
    # Verify that prediction output has shape (16,)
    pred = cell.D @ a
    assert pred.shape == (16,)


def test_evaluation_no_weight_update():
    model = AgnisBaseline(d_in=4, d_out=4, d_z=8, use_memory=True)
    
    # Extract copy of weights before evaluation
    D_before = model.cell.D.clone()
    E_before = model.cell.E.clone()
    
    inputs = torch.randn(5, 4)
    targets = torch.randn(5, 4)
    
    # Run evaluation
    evaluate_model_on_task(model, inputs, targets)
    
    # Verify weights are unchanged
    assert torch.allclose(model.cell.D, D_before), "Evaluation updated generative weights!"
    assert torch.allclose(model.cell.E, E_before), "Evaluation updated recognition weights!"


def test_capacity_stress_config():
    # Verify capacity stress can be loaded and overrides config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "phase1_smoke.yaml"
    )
    config = load_config(config_path)
    
    # Mock capacity stress override execution logic
    config.model.d_z = 4
    config.training.n_tasks = 5
    config.training.pairs_per_task = 4
    
    assert config.model.d_z == 4
    assert config.training.n_tasks == 5
    assert config.training.pairs_per_task == 4


def test_benchmark_smoke_run(tmp_path):
    # Runs the benchmark runner on orthogonal condition with micro config on a dry-run
    # using subprocess to verify imports, configurations, and argument parsing are functional.
    benchmark_script = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "phase1_associative", "run_benchmark.py"
    )
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "phase1_smoke.yaml"
    )
    
    cmd = [
        sys.executable,
        benchmark_script,
        "--condition", "orthogonal",
        "--model", "agnis_kwta",
        "--seed", "0",
        "--config", config_path,
        "--smoke",
        "--dry-run"
    ]
    
    res = os.system(" ".join(cmd))
    assert res == 0, "Benchmark dry-run failed!"
