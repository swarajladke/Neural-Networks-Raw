"""
Raw AGNIS — src/agnis/core/sparsity.py

Sparse activation mechanisms:
  - kwta: k-Winners-Take-All hard mask
  - compute_sparsity_level: fraction of inactive units
  - lateral_inhibition_mask: sparse inhibitory connection builder

Design notes:
  - kWTA is the primary sparsity mechanism for Raw AGNIS.
  - All operations are differentiable in the forward direction
    (though Raw AGNIS core does not rely on backprop for Hebbian updates).
  - Sparsity fraction should be configurable; typical range is 5–25%.
"""

import torch
from typing import Optional


def kwta(a: torch.Tensor, k: int, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    k-Winners-Take-All: Keep the top-k activations, zero the rest.

    Parameters
    ----------
    a : torch.Tensor of shape (d_z,)
        Activation vector before sparsification.
    k : int
        Number of units to keep active. If k >= d_z, returns a unchanged.
    bias : torch.Tensor of shape (d_z,), optional
        Per-unit competition bias subtracted from the selection score
        (|a| - bias). Used for latent fatigue/adaptation: recently active
        units accumulate bias and are handicapped in the competition,
        which destroys short deterministic limit cycles. Retained values
        are the ORIGINAL activations; only winner selection is biased.

    Returns
    -------
    torch.Tensor of shape (d_z,)
        Sparse activation with only top-k values retained.

    Notes
    -----
    Selection is based on absolute magnitude (not raw value).
    This ensures that strongly negative activations also compete.
    """
    d_z = a.shape[0]
    if k >= d_z:
        return a  # dense mode — no sparsity applied

    # Selection score: absolute value, optionally handicapped by fatigue bias
    abs_a = a.abs()
    score = abs_a - bias if bias is not None else abs_a

    # Find threshold: the k-th largest score
    threshold_val = torch.topk(score, k, largest=True, sorted=True).values[-1]

    # Create mask: keep units where score >= threshold
    mask = score >= threshold_val

    # If more than k units pass (ties), keep exactly k by additional selection
    if mask.sum() > k:
        # Zero out ties by selecting only the first k in index order
        indices = torch.where(mask)[0][:k]
        mask = torch.zeros_like(a, dtype=torch.bool)
        mask[indices] = True

    return a * mask.float()


def kwta_batch(a: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-Winners-Take-All applied to a batch of activation vectors.

    Parameters
    ----------
    a : torch.Tensor of shape (batch, d_z)
        Batch of activation vectors.
    k : int
        Number of units to keep active per sample.

    Returns
    -------
    torch.Tensor of shape (batch, d_z)
        Sparsified batch activations.
    """
    if k >= a.shape[1]:
        return a

    abs_a = a.abs()
    threshold_vals = torch.topk(abs_a, k, dim=1, largest=True, sorted=True).values[:, -1:]
    mask = abs_a >= threshold_vals
    return a * mask.float()


def compute_sparsity_level(a: torch.Tensor) -> float:
    """
    Compute the sparsity level: fraction of units that are zero (inactive).

    Parameters
    ----------
    a : torch.Tensor of shape (d_z,) or (batch, d_z)
        Activation tensor.

    Returns
    -------
    float
        Fraction of units equal to zero. 1.0 = fully sparse, 0.0 = fully dense.
    """
    return (a == 0).float().mean().item()


def compute_active_fraction(a: torch.Tensor) -> float:
    """
    Compute the active fraction: fraction of units that are nonzero.

    Parameters
    ----------
    a : torch.Tensor

    Returns
    -------
    float
        Fraction of nonzero units. Complement of sparsity_level.
    """
    return (a != 0).float().mean().item()


def build_lateral_inhibition_matrix(
    d_z: int,
    connection_density: float = 0.2,
    strength_range: tuple = (-0.2, 0.0),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Build a sparse lateral inhibition matrix L.

    Parameters
    ----------
    d_z : int
        Number of units.
    connection_density : float
        Fraction of off-diagonal connections that are non-zero. Default 0.2.
    strength_range : tuple (min, max)
        Range of inhibitory connection strengths. Should be negative.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor of shape (d_z, d_z)
        Sparse lateral inhibition matrix with zero diagonal.
    """
    if seed is not None:
        torch.manual_seed(seed)

    L = torch.zeros(d_z, d_z)
    # Mask for off-diagonal connections to include
    mask = torch.rand(d_z, d_z) < connection_density
    mask.fill_diagonal_(False)  # no self-inhibition

    n_connections = mask.sum().item()
    L[mask] = torch.empty(int(n_connections)).uniform_(
        strength_range[0], strength_range[1]
    )
    return L
