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


def birth_unit_in_hierarchy(
    hierarchy,
    l: int,
    residual_error: torch.Tensor,
    current_input: torch.Tensor,
) -> Tuple[int, int]:
    """
    Birth a new unit in layer l of a PredictiveHierarchy.

    Parameters
    ----------
    hierarchy : PredictiveHierarchy
    l : int
        Layer index where the new unit is born.
    residual_error : torch.Tensor of shape (dim_below_l,)
        Current bottom-up prediction error.
    current_input : torch.Tensor of shape (dim_below_l,)
        Current input to the encoder (raw input or error accum).

    Returns
    -------
    tuple of (old_dim, new_dim)
        Latent dimensions before and after growth.
    """
    E_l = hierarchy._E(l)
    R_l = hierarchy._R(l)
    L_l = hierarchy._L(l)
    D_l = hierarchy._D_inter(l)
    z_l = hierarchy._z(l)
    z_prev_l = hierarchy._z_prev(l)

    # Call single-layer birth logic to expand l's weight matrices
    D_l_new, E_l_new, R_l_new, L_l_new = birth_new_unit(
        D=D_l,
        E=E_l,
        R=R_l,
        L=L_l,
        residual_error=residual_error,
        current_input=current_input,
    )

    # Update layer l's weight parameters
    hierarchy._set_E(l, E_l_new)
    hierarchy._set_D_inter(l, D_l_new)
    hierarchy._set_R(l, R_l_new)
    hierarchy._set_L(l, L_l_new)

    # Expand state vectors
    device = z_l.device
    hierarchy._set_z(l, torch.cat([z_l, torch.zeros(1, device=device)]))
    hierarchy._set_z_prev(l, torch.cat([z_prev_l, torch.zeros(1, device=device)]))

    # Expand the layer above if it exists (layer l+1)
    if l < hierarchy.n_layers - 1:
        # D_inter_{l+1} predicts z_l from z_{l+1}
        # It needs a new row of zeros at the end to match the new dimension of z_l
        D_above = hierarchy._D_inter(l + 1)
        new_row = torch.zeros(1, D_above.shape[1], device=D_above.device)
        hierarchy._set_D_inter(l + 1, torch.cat([D_above, new_row], dim=0))

        # E_{l+1} encodes z_l into z_{l+1}
        # It needs a new column of zeros at the end
        E_above = hierarchy._E(l + 1)
        new_col = torch.zeros(E_above.shape[0], 1, device=E_above.device)
        hierarchy._set_E(l + 1, torch.cat([E_above, new_col], dim=1))

        # The error accumulator of layer l+1 needs a new element
        accum = hierarchy._error_accum(l + 1)
        hierarchy._set_error_accum(l + 1, torch.cat([accum, torch.zeros(1, device=accum.device)]))

    old_dim = hierarchy.layer_dims[l]
    hierarchy.layer_dims[l] += 1
    new_dim = hierarchy.layer_dims[l]

    # Keep _dim_below in sync so reset_state() allocates correct sizes
    if l + 1 < hierarchy.n_layers:
        hierarchy._dim_below[l + 1] = new_dim

    return old_dim, new_dim

