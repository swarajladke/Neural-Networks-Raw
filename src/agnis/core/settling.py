"""
Raw AGNIS — src/agnis/core/settling.py

Utility functions for iterative state settling.

Provides:
  - settle_z: standalone settling function (for use outside PredictiveCell)
  - settling_convergence: check whether z has converged
  - SettlingMonitor: tracks stability metrics across settling steps
"""

import torch
from typing import Tuple, Optional


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
