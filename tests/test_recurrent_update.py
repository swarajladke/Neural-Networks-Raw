"""
Raw AGNIS — tests/test_recurrent_update.py

Unit tests for temporal recurrent matrix R updates and stability bounds.
"""

import pytest
import torch
from agnis.core.predictive_cell import PredictiveCell


def test_recurrent_update_changes_weights():
    # Setup cell with recurrence enabled
    cell = PredictiveCell(
        d_in=8,
        d_z=8,
        use_recurrent=True,
        R_update_enabled=True,
        R_drive_enabled=True,
        spectral_radius_max=1.0,
    )
    
    # Save initial R
    R_init = cell.R.clone()
    
    s = torch.randn(8)
    a = cell.forward(s)
    # Run second forward step so z_prev is non-zero and Hebbian recurrence updates are non-zero
    s2 = torch.randn(8)
    a2 = cell.forward(s2)
    
    # Trigger update
    cell.update_weights(s2, a2)
    
    # R should have changed
    assert not torch.equal(cell.R, R_init), "Recurrent matrix R did not change after update!"
    assert len(cell.R_update_norms) == 1


def test_recurrent_stability_norm():
    cell = PredictiveCell(
        d_in=8,
        d_z=8,
        use_recurrent=True,
        R_update_enabled=True,
        R_drive_enabled=True,
        spectral_radius_max=0.5, # very tight constraint
    )
    
    # Artificially expand R to exceed the limit
    cell.R = cell.R * 10.0
    
    # Normalize R
    cell.normalize_R()
    
    # Check spectral norm is bounded by spectral_radius_max
    R_norm = torch.linalg.matrix_norm(cell.R, ord=2).item()
    assert R_norm <= 0.51, f"Recurrent matrix norm exceeds spectral_radius_max: {R_norm}"
