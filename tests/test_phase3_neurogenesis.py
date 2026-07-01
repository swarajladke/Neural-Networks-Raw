"""
Raw AGNIS — tests/test_phase3_neurogenesis.py

Comprehensive unit tests verifying the Phase 3 neurogenesis growth mechanisms,
maturity gating, in-place weight expansions, and pruning in PredictiveCell.
"""

import pytest
import torch
import os
import subprocess
from agnis.core.predictive_cell import PredictiveCell
from agnis.neurogenesis.growth_controller import GrowthController


def test_growth_controller_consecutive_trigger():
    """Verify that GrowthController triggers growth only after exceeding threshold consecutively."""
    gc = GrowthController(threshold=0.35, consecutive_n=3, lambda_cost=0.01, ema_alpha=1.0)

    # Exceed threshold once (consecutive_n=3)
    triggered = gc.update(
        error=0.5, novelty=0.5, uncertainty=0.0,
        interference=0.0, coverage=0.0, cost=32.0
    )
    assert not triggered, "Should not trigger after 1 step"

    triggered = gc.update(
        error=0.5, novelty=0.5, uncertainty=0.0,
        interference=0.0, coverage=0.0, cost=32.0
    )
    assert not triggered, "Should not trigger after 2 steps"

    triggered = gc.update(
        error=0.5, novelty=0.5, uncertainty=0.0,
        interference=0.0, coverage=0.0, cost=32.0
    )
    assert triggered, "Should trigger after 3 consecutive steps"


def test_predictive_cell_dynamic_growth():
    """Verify grow_units dynamically expands all matrices in-place without corrupting existing weights."""
    cell = PredictiveCell(d_in=8, d_z=6, max_latent_dim=12)
    
    # Save original weights
    old_D = cell.D.clone()
    old_E = cell.E.clone()
    old_R = cell.R.clone()
    
    current_input = torch.randn(8)
    residual_error = torch.randn(8)
    
    # Spawn 2 new units
    cell.grow_units(k=2, current_input=current_input, residual_error=residual_error)
    
    # Assert dimension updates
    assert cell.d_z == 8
    assert cell.D.shape == (8, 8)
    assert cell.E.shape == (8, 8)
    assert cell.R.shape == (8, 8)
    assert cell.L.shape == (8, 8)
    assert cell.maturity.shape == (8,)
    assert cell.usage.shape == (8,)
    
    # Assert old weights are preserved in-place
    assert torch.allclose(cell.D[:, :6], old_D)
    assert torch.allclose(cell.E[:6, :], old_E)
    assert torch.allclose(cell.R[:6, :6], old_R)
    
    # Assert new unit maturity is initialized to 0.0
    assert (cell.maturity[6:] == 0.0).all()


def test_maturity_gating_suppression():
    """Verify that immature units have their activation gated to zero during settling."""
    cell = PredictiveCell(d_in=4, d_z=3, max_latent_dim=5)
    
    # Set one unit to immature (0.0)
    cell.maturity[2] = 0.0
    
    s = torch.randn(4)
    # Forward run
    a = cell.forward(s)
    
    # Immature unit should be forced to zero activation
    assert a[2].item() == 0.0, "Immature unit activation should be gated to zero"


def test_predictive_cell_pruning():
    """Verify prune_units correctly removes eligible low-performance units."""
    cell = PredictiveCell(d_in=4, d_z=3, max_latent_dim=6)
    
    # Setup age, usage, and importance to trigger pruning on unit 2
    cell.age[2] = 150.0
    cell.usage[2] = 0.001
    cell.maturity[2] = 0.1
    cell.importance_D[:, 2] = 0.001  # extremely low importance
    
    # Units 0 and 1 are young/active/important
    cell.age[:2] = 10.0
    cell.usage[:2] = 0.8
    cell.importance_D[:, :2] = 1.0
    
    cell.prune_units(min_age=100, usage_threshold=0.01, importance_threshold=0.01, maturity_threshold=0.5)
    
    # Assert unit 2 was deleted, reducing dimension to 2
    assert cell.d_z == 2
    assert cell.D.shape == (4, 2)
    assert cell.E.shape == (2, 4)
    assert cell.maturity.shape == (2,)


def test_phase3_smoke_run():
    """Verify running the Phase 3 benchmark with smoke config runs successfully without error."""
    script_path = os.path.join("experiments", "phase3_neurogenesis", "run_neurogenesis_benchmark.py")
    config_path = os.path.join("configs", "phase3_smoke.yaml")
    
    cmd = [
        "python",
        script_path,
        "--condition", "doublet",
        "--model", "seq_agnis_neurogenesis",
        "--seed", "0",
        "--config", config_path,
        "--smoke"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Smoke run failed with stderr: {res.stderr}\nStdout: {res.stdout}"
