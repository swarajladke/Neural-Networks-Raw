"""
Raw AGNIS — src/agnis/core/settling.py

Utility functions for iterative state settling.

Provides:
  - settle_z: standalone settling function (for use outside PredictiveCell)
  - settling_convergence: check whether z has converged
  - SettlingMonitor: tracks stability metrics across settling steps
  - joint_settle: synchronous double-buffered settling for PredictiveHierarchy
"""

import torch
from typing import List, Tuple, Optional


def settle_z(
    z_init: torch.Tensor,
    s: torch.Tensor,
    D: torch.Tensor,
    E: torch.Tensor,
    R: Optional[torch.Tensor] = None,
    L: Optional[torch.Tensor] = None,
    z_prev: Optional[torch.Tensor] = None,
    activation_fn=torch.tanh,
    eta_z: float = 0.05,
    rho: float = 0.3,
    lambda_lat: float = 0.1,
    lambda_sparse: float = 0.01,
    n_settle: int = 10,
    clip_val: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Iterative settling of latent state z toward reduced prediction error.

    Parameters
    ----------
    z_init : torch.Tensor of shape (d_z,)
        Initial latent state.
    s : torch.Tensor of shape (d_in,)
        Current input stimulus.
    D : torch.Tensor of shape (d_in, d_z)
        Generative weight matrix.
    E : torch.Tensor of shape (d_z, d_in)
        Recognition weight matrix.
    R : torch.Tensor of shape (d_z, d_z), optional
        Recurrent weight matrix. If None, no recurrent drive.
    L : torch.Tensor of shape (d_z, d_z), optional
        Lateral inhibition matrix. If None, no lateral drive.
    z_prev : torch.Tensor of shape (d_z,), optional
        Previous latent state (for recurrent drive).
    activation_fn : callable
        Nonlinear activation function.
    eta_z : float
        Settling step size.
    rho : float
        Recurrent drive strength.
    lambda_lat : float
        Lateral inhibition strength.
    lambda_sparse : float
        L1 sparsity penalty.
    n_settle : int
        Number of settling iterations.
    clip_val : float
        Clipping bound for z to prevent explosion.

    Returns
    -------
    z : torch.Tensor of shape (d_z,)
        Final settled latent state.
    e : torch.Tensor of shape (d_in,)
        Final prediction error.
    error_history : list of float
        MSE at each settling step (for convergence analysis).
    """
    z = z_init.clone()
    error_history = []

    for _ in range(n_settle):
        a = activation_fn(z)
        s_hat = D @ a
        e = s - s_hat

        error_history.append((e ** 2).mean().item())

        # Activation derivative (for tanh: 1 - a^2)
        a_prime = 1.0 - a ** 2

        # Drive terms
        d_rec = E @ s - z
        d_fb = (D.T @ e) * a_prime

        d_time = torch.zeros_like(z)
        if R is not None and z_prev is not None:
            d_time = R @ z_prev

        d_lat = torch.zeros_like(z)
        if L is not None:
            d_lat = L @ a

        # Settling update
        delta_z = (
            d_rec
            + d_fb
            + rho * d_time
            + lambda_lat * d_lat
            - lambda_sparse * torch.sign(z)
        )
        z = z + eta_z * delta_z
        z = torch.clamp(z, -clip_val, clip_val)

    # Final prediction and error
    a_final = activation_fn(z)
    s_hat_final = D @ a_final
    e_final = s - s_hat_final

    return z, e_final, error_history


def settling_convergence(
    error_history: list,
    tolerance: float = 1e-4,
) -> int:
    """
    Find the step at which settling converged (error change < tolerance).

    Parameters
    ----------
    error_history : list of float
        MSE at each settling step.
    tolerance : float
        Convergence threshold for error change.

    Returns
    -------
    int
        Step index at convergence. Returns len(error_history) if not converged.
    """
    for i in range(1, len(error_history)):
        if abs(error_history[i] - error_history[i - 1]) < tolerance:
            return i
    return len(error_history)


class SettlingMonitor:
    """
    Tracks settling dynamics across multiple forward passes.

    Useful for detecting instability, measuring average convergence speed,
    and computing settling-based uncertainty.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._histories = []
        self._convergence_steps = []

    def record(self, error_history: list):
        """Record a settling trajectory."""
        self._histories.append(error_history)
        self._convergence_steps.append(settling_convergence(error_history))
        # Keep only last `window` records
        if len(self._histories) > self.window:
            self._histories.pop(0)
            self._convergence_steps.pop(0)

    @property
    def mean_convergence_steps(self) -> float:
        """Average number of settling steps to convergence over recent history."""
        if not self._convergence_steps:
            return 0.0
        return sum(self._convergence_steps) / len(self._convergence_steps)

    @property
    def uncertainty(self) -> float:
        """
        Variance of final errors across recent settling trajectories.
        High variance → settling is unstable → high uncertainty.
        """
        if len(self._histories) < 2:
            return 0.0
        final_errors = [h[-1] for h in self._histories if h]
        t = torch.tensor(final_errors)
        return t.var().item()



def joint_settle(
    z_list: List[torch.Tensor],
    z_prev_list: List[torch.Tensor],
    s: torch.Tensor,
    E_list: List[torch.Tensor],
    R_list: List[torch.Tensor],
    L_list: List[torch.Tensor],
    D_inter_list: List[torch.Tensor],
    k_sparse_list: List[int],
    committing: List[bool],
    lambda_td: float = 0.3,
    eta_z: float = 0.05,
    n_settle: int = 10,
    clip_val: float = 10.0,
    bias_list: Optional[List[Optional[torch.Tensor]]] = None,
    observed_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """
    Synchronous double-buffered joint settling across all committing layers.

    All committing layers update their z simultaneously in each micro-iteration.
    Non-committing layers hold their current state (frozen during settling).
    Uses double buffering: new z values are computed into a separate buffer,
    then swapped after all layers are updated.

    Parameters
    ----------
    z_list : list of torch.Tensor
        Current latent states [z_0, ..., z_{N-1}].
    z_prev_list : list of torch.Tensor
        Previous latent states (for recurrent drive).
    s : torch.Tensor of shape (d_input,)
        Raw input stimulus.
    E_list : list of torch.Tensor
        Recognition matrices [E_0, ..., E_{N-1}].
    R_list : list of torch.Tensor
        Recurrent matrices [R_0, ..., R_{N-1}].
    L_list : list of torch.Tensor
        Lateral inhibition matrices [L_0, ..., L_{N-1}].
    D_inter_list : list of torch.Tensor
        Inter-layer decoder matrices [D_inter_0, ..., D_inter_{N-1}].
        D_inter[l] has shape (dim_below_l, dim_l) and maps z^l to prediction
        of the level below.
    k_sparse_list : list of int
        kWTA k values per layer.
    committing : list of bool
        Which layers are committing (updating) at this timestep.
    lambda_td : float
        Top-down conformity pressure strength.
    eta_z : float
        Settling step size.
    n_settle : int
        Number of settling micro-iterations.
    clip_val : float
        Clipping bound for z to prevent explosion.
    bias_list : list of torch.Tensor or None
        Optional fatigue bias tensors per layer.

    Returns
    -------
    list of torch.Tensor
        Settled latent states [z_0, ..., z_{N-1}].
    """
    from agnis.core.sparsity import kwta

    n_layers = len(z_list)

    # Clone the z values into a working buffer
    z = [z_l.clone() for z_l in z_list]

    for _ in range(n_settle):
        # Double buffer: compute all new z values before swapping
        z_new = [z_l.clone() for z_l in z]

        for l in range(n_layers):
            if not committing[l]:
                continue  # non-committing layers hold state

            # ── Bottom-up error (what this layer must explain from below) ──
            if l == 0:
                e_below = s - D_inter_list[0] @ z[0]
                if observed_mask is not None:
                    e_below = e_below * observed_mask
            else:
                e_below = z[l - 1] - D_inter_list[l] @ z[l]

            # ── Top-down error (pressure to conform to prediction from above)
            if l < n_layers - 1:
                e_above = z[l] - D_inter_list[l + 1] @ z[l + 1]
            else:
                e_above = torch.zeros_like(z[l])

            # ── Full drive ─────────────────────────────────────────────────
            drive = (
                E_list[l] @ e_below
                - lambda_td * e_above
                + R_list[l] @ z_prev_list[l]
                - L_list[l] @ z[l]
            )

            z_candidate = z[l] + eta_z * drive
            z_candidate = torch.clamp(z_candidate, -clip_val, clip_val)
            bias = bias_list[l] if bias_list is not None else None
            z_new[l] = kwta(z_candidate, k_sparse_list[l], bias=bias)

        # Double-buffer swap: copy new values into z
        z = z_new

    return z

