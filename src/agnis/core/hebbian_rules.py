"""
Raw AGNIS — src/agnis/core/hebbian_rules.py

Local Hebbian weight update rules.

All updates are purely local:
  - Generative update uses prediction error and post-synaptic activation
  - Recognition update uses latent error and pre-synaptic input
  - Recurrent update uses temporal state association
  - Plasticity-gated update scales by per-weight plasticity

No global backpropagation is used for these updates.
PyTorch is used only for tensor operations.

Reference equations: NOTES/predictive_coding_equations.md
"""

import torch
from typing import Optional


def hebbian_generative_update(
    e: torch.Tensor,
    a: torch.Tensor,
    eta_D: float,
) -> torch.Tensor:
    """
    Compute the generative (decoding) weight update.

    ΔD = η_D * outer(e, a)

    Parameters
    ----------
    e : torch.Tensor of shape (d_in,)
        Prediction error: e = s - D @ a.
    a : torch.Tensor of shape (d_z,)
        Post-settling sparse activation.
    eta_D : float
        Learning rate for generative weights.

    Returns
    -------
    torch.Tensor of shape (d_in, d_z)
        Weight update matrix. Add to D.

    Notes
    -----
    This update pushes D[:, j] toward better reconstructing s when unit j is active.
    Active units with high prediction error get the largest updates.
    Inactive units (a_j = 0) receive no update — a key benefit of sparsity.
    """
    return eta_D * torch.outer(e, a)


def hebbian_recognition_update(
    z: torch.Tensor,
    E: torch.Tensor,
    s: torch.Tensor,
    eta_E: float,
) -> torch.Tensor:
    """
    Compute the recognition (encoding) weight update.

    ΔE = η_E * outer(z - E @ s, s)

    Parameters
    ----------
    z : torch.Tensor of shape (d_z,)
        Post-settling latent state.
    E : torch.Tensor of shape (d_z, d_in)
        Current recognition weight matrix.
    s : torch.Tensor of shape (d_in,)
        Current input stimulus.
    eta_E : float
        Learning rate for recognition weights.

    Returns
    -------
    torch.Tensor of shape (d_z, d_in)
        Weight update matrix. Add to E.

    Notes
    -----
    The error (z - E@s) is the difference between the settled latent state
    and the direct linear encoding. E learns to predict z from s directly
    (fast recognition pathway).
    """
    recognition_error = z - E @ s
    return eta_E * torch.outer(recognition_error, s)


def hebbian_recurrent_update(
    z: torch.Tensor,
    z_prev: torch.Tensor,
    eta_R: float,
) -> torch.Tensor:
    """
    Compute the recurrent weight update.

    ΔR = η_R * outer(z, z_prev)

    Parameters
    ----------
    z : torch.Tensor of shape (d_z,)
        Current latent state.
    z_prev : torch.Tensor of shape (d_z,)
        Previous latent state.
    eta_R : float
        Learning rate for recurrent weights.

    Returns
    -------
    torch.Tensor of shape (d_z, d_z)
        Weight update matrix. Add to R.

    Notes
    -----
    This is a temporal Hebbian rule: units that co-occur across consecutive
    timesteps strengthen their recurrent connections.
    """
    return eta_R * torch.outer(z, z_prev)


def plasticity_gated_update(
    pre: torch.Tensor,
    post_error: torch.Tensor,
    plasticity: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """
    Compute a plasticity-gated Hebbian weight update.

    ΔW_ij = η * plasticity_ij * pre_i * post_error_j

    Parameters
    ----------
    pre : torch.Tensor of shape (d_pre,)
        Pre-synaptic activity vector.
    post_error : torch.Tensor of shape (d_post,)
        Post-synaptic error vector.
    plasticity : torch.Tensor of shape (d_post, d_pre)
        Per-weight plasticity values in [0, 1].
    eta : float
        Base learning rate.

    Returns
    -------
    torch.Tensor of shape (d_post, d_pre)
        Plasticity-gated weight update. Add to corresponding weight matrix.

    Notes
    -----
    High plasticity (≈1) → weight updates freely.
    Low plasticity (≈0) → weight barely changes (protected).
    """
    # outer product: (d_post, d_pre)
    raw_update = torch.outer(post_error, pre)
    return eta * plasticity * raw_update


def compute_plasticity(
    novelty: float,
    importance: torch.Tensor,
    age: torch.Tensor,
    uncertainty: float = 0.0,
    a_p: float = 2.0,
    b_p: float = 1.0,
    c_p: float = 3.0,
    d_p: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-weight plasticity values.

    plasticity_ij = sigmoid(
        a_p * novelty
      + b_p * uncertainty
      - c_p * importance_ij
      - d_p * age_ij
    )

    Parameters
    ----------
    novelty : float
        Current input novelty (EMA of |e|).
    importance : torch.Tensor
        Per-weight importance values (EMA of |ΔW|). Same shape as weight matrix.
    age : torch.Tensor
        Per-weight age (steps since last meaningful update). Same shape.
    uncertainty : float
        Variance of z across settling steps.
    a_p, b_p, c_p, d_p : float
        Scaling coefficients for each term.

    Returns
    -------
    torch.Tensor
        Plasticity values in (0, 1), same shape as importance.
    """
    logit = (
        a_p * novelty
        + b_p * uncertainty
        - c_p * importance
        - d_p * age
    )
    return torch.sigmoid(logit)


def update_importance(
    importance: torch.Tensor,
    delta_W: torch.Tensor,
    alpha_I: float = 0.01,
) -> torch.Tensor:
    """
    Update per-weight importance using EMA of absolute weight changes.

    importance_ij ← (1 - α_I) * importance_ij + α_I * |ΔW_ij|

    Parameters
    ----------
    importance : torch.Tensor
        Current importance tensor (same shape as delta_W).
    delta_W : torch.Tensor
        Weight change tensor (same shape as importance).
    alpha_I : float
        EMA decay rate. Small = slow-changing importance.

    Returns
    -------
    torch.Tensor
        Updated importance tensor (same shape).
    """
    return (1.0 - alpha_I) * importance + alpha_I * delta_W.abs()


def normalize_weights(W: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize weight matrix columns (or rows) to unit norm.
    Used to prevent weight explosion during sustained Hebbian updates.

    Parameters
    ----------
    W : torch.Tensor of shape (m, n)
        Weight matrix to normalize.
    dim : int
        Dimension to normalize over. 0 = normalize columns (typical for D).
        1 = normalize rows (typical for E).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor of shape (m, n)
        Normalized weight matrix.
    """
    norms = W.norm(dim=dim, keepdim=True).clamp(min=eps)
    return W / norms
