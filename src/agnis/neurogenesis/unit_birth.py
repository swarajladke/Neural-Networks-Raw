"""
Raw AGNIS — src/agnis/neurogenesis/unit_birth.py

UnitBirth: Initializes new units from residual error.

Phase 3 implementation. Creates new units whose generative weights are
initialized from the current residual prediction error, so they are
born to explain exactly what existing units cannot predict.

See NOTES/neurogenesis_design.md for full design.
"""

import torch
from typing import Tuple


def birth_new_unit(
    D: torch.Tensor,
    E: torch.Tensor,
    R: torch.Tensor,
    L: torch.Tensor,
    residual_error: torch.Tensor,
    current_input: torch.Tensor,
    r_init_std: float = 0.01,
    l_density: float = 0.2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add a new unit to the network by expanding D, E, R, L.

    New unit initialization:
        D[:, new] = normalize(residual_error)
        E[new, :] = normalize(current_input)
        R[new, :] = small random stable values
        L[new, :] = L[:, new] = sparse inhibitory connections

    Parameters
    ----------
    D : torch.Tensor of shape (d_in, d_z)
    E : torch.Tensor of shape (d_z, d_in)
    R : torch.Tensor of shape (d_z, d_z)
    L : torch.Tensor of shape (d_z, d_z)
    residual_error : torch.Tensor of shape (d_in,)
        Prediction error at birth time.
    current_input : torch.Tensor of shape (d_in,)
        Current input stimulus.
    r_init_std : float
        Standard deviation of recurrent weight initialization.
    l_density : float
        Density of lateral inhibitory connections for new unit.
    eps : float
        Small constant for safe normalization.

    Returns
    -------
    D_new, E_new, R_new, L_new : expanded weight matrices
    """
    d_in, d_z = D.shape
    new_d_z = d_z + 1

    # --- Generative column (d_in,) ---
    e_norm = residual_error.norm().item()
    new_d_col = residual_error / max(e_norm, eps)

    # --- Recognition row (d_in,) ---
    s_norm = current_input.norm().item()
    new_e_row = current_input / max(s_norm, eps)

    # --- Recurrent row (d_z,) for new unit's connections to existing units ---
    new_r_row = torch.randn(d_z) * r_init_std

    # --- Expand D: (d_in, d_z) → (d_in, d_z+1) ---
    D_new = torch.cat([D, new_d_col.unsqueeze(1)], dim=1)

    # --- Expand E: (d_z, d_in) → (d_z+1, d_in) ---
    E_new = torch.cat([E, new_e_row.unsqueeze(0)], dim=0)

    # --- Expand R: (d_z, d_z) → (d_z+1, d_z+1) ---
    # New unit's row: connections from old units to new unit
    new_r_col = torch.randn(d_z) * r_init_std  # old units → new unit
    # Build new R
    R_new = torch.zeros(new_d_z, new_d_z)
    R_new[:d_z, :d_z] = R
    R_new[d_z, :d_z] = new_r_row    # new unit's connections to old units
    R_new[:d_z, d_z] = new_r_col    # old units' connections to new unit
    # R_new[d_z, d_z] = 0 (no self-recurrence)

    # --- Expand L: (d_z, d_z) → (d_z+1, d_z+1) ---
    L_new = torch.zeros(new_d_z, new_d_z)
    L_new[:d_z, :d_z] = L
    # Sparse inhibitory connections for new unit
    mask_row = torch.rand(d_z) < l_density
    L_new[d_z, :d_z][mask_row] = torch.empty(mask_row.sum()).uniform_(-0.2, 0.0)
    mask_col = torch.rand(d_z) < l_density
    L_new[:d_z, d_z][mask_col] = torch.empty(mask_col.sum()).uniform_(-0.2, 0.0)

    return D_new, E_new, R_new, L_new
