"""
Raw AGNIS — src/agnis/core/predictive_cell.py

PredictiveCell: A single-layer predictive coding unit.

Implements:
  - Recognition pathway: E (d_z x d_in) — encodes input s into latent z
  - Generative pathway: D (d_in x d_z) — reconstructs input s from activation a
  - Recurrent pathway: R (d_z x d_z) — injects previous latent state
  - Lateral inhibition: L (d_z x d_z) — sparse inhibitory connections
  - State settling: iterative update of z toward reduced prediction error
  - Local Hebbian updates for D, E, R
  - kWTA sparsity on activation a
  - Per-weight importance and plasticity tracking

Design notes:
  - No global backprop for core mechanism.
  - PyTorch is used for tensor operations and parameter storage only.
  - All updates are explicitly computed using local Hebbian rules.
  - For ablation, set use_sparsity=False, use_recurrent=False, etc.

Reference equations: NOTES/predictive_coding_equations.md
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from agnis.core.sparsity import kwta, compute_sparsity_level
from agnis.core.hebbian_rules import (
    hebbian_generative_update,
    hebbian_recognition_update,
    hebbian_recurrent_update,
    update_importance,
)


class PredictiveCell(nn.Module):
    """
    A single predictive coding layer with local Hebbian learning.

    Parameters
    ----------
    d_in : int
        Dimensionality of the input (stimulus) vector.
    d_z : int
        Dimensionality of the latent state.
    k_sparse : int
        Number of units to keep active after kWTA (0 = all, i.e., dense).
    n_settle : int
        Number of settling iterations per input.
    eta_z : float
        Settling step size.
    eta_D : float
        Generative weight learning rate.
    eta_E : float
        Recognition weight learning rate.
    eta_R : float
        Recurrent weight learning rate.
    rho : float
        Recurrent drive strength.
    lambda_lat : float
        Lateral inhibition strength.
    lambda_sparse : float
        L1 sparsity penalty on z during settling.
    use_sparsity : bool
        Whether to apply kWTA. Set False for ablation.
    use_recurrent : bool
        Whether to use recurrent drive. Set False for ablation.
    use_lateral : bool
        Whether to use lateral inhibition. Set False for ablation.
    importance_decay : float
        EMA decay for importance tracking (alpha_I).
    """

    def __init__(
        self,
        d_in: int,
        d_z: int,
        k_sparse: int = 0,
        n_settle: int = 10,
        eta_z: float = 0.05,
        eta_D: float = 0.01,
        eta_E: float = 0.01,
        eta_R: float = 0.005,
        rho: float = 0.3,
        lambda_lat: float = 0.1,
        lambda_sparse: float = 0.01,
        use_sparsity: bool = True,
        use_recurrent: bool = False,
        use_lateral: bool = False,
        importance_decay: float = 0.01,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_z = d_z
        self.k_sparse = k_sparse if k_sparse > 0 else d_z  # default: dense
        self.n_settle = n_settle
        self.eta_z = eta_z
        self.eta_D = eta_D
        self.eta_E = eta_E
        self.eta_R = eta_R
        self.rho = rho
        self.lambda_lat = lambda_lat
        self.lambda_sparse = lambda_sparse
        self.use_sparsity = use_sparsity
        self.use_recurrent = use_recurrent
        self.use_lateral = use_lateral
        self.importance_decay = importance_decay

        # ── Weight matrices (not nn.Parameters — we update manually) ──────────
        # Generative: D (d_in x d_z) — decodes latent to input space
        self.D = torch.nn.init.xavier_uniform_(
            torch.empty(d_in, d_z)
        )
        # Recognition: E (d_z x d_in) — encodes input to latent space
        self.E = torch.nn.init.xavier_uniform_(
            torch.empty(d_z, d_in)
        )
        # Recurrent: R (d_z x d_z) — temporal state injection
        self.R = torch.zeros(d_z, d_z)
        torch.nn.init.orthogonal_(self.R)
        self.R = self.R * 0.1  # start small and stable

        # Lateral: L (d_z x d_z) — sparse inhibitory connections
        # Off-diagonal negative, diagonal zero
        self.L = torch.zeros(d_z, d_z)
        if use_lateral:
            self._init_lateral_connections()

        # ── Per-weight importance (d_in x d_z for D, d_z x d_in for E) ───────
        self.importance_D = torch.zeros(d_in, d_z)
        self.importance_E = torch.zeros(d_z, d_in)
        self.importance_R = torch.zeros(d_z, d_z)

        # ── Latent state ──────────────────────────────────────────────────────
        self.z = torch.zeros(d_z)
        self.z_prev = torch.zeros(d_z)

        # ── Metrics ───────────────────────────────────────────────────────────
        self._last_error: Optional[torch.Tensor] = None
        self._last_sparsity: float = 0.0
        self._last_pred: Optional[torch.Tensor] = None

    def _init_lateral_connections(self):
        """Initialize sparse inhibitory lateral connections."""
        sparsity = 0.8  # 80% of lateral connections are zero
        mask = torch.rand(self.d_z, self.d_z) > sparsity
        # Off-diagonal only; zero diagonal
        mask.fill_diagonal_(False)
        self.L[mask] = torch.empty(mask.sum()).uniform_(-0.2, 0.0)

    def activation(self, z: torch.Tensor) -> torch.Tensor:
        """Bounded nonlinearity: tanh."""
        return torch.tanh(z)

    def activation_deriv(self, z: torch.Tensor) -> torch.Tensor:
        """Derivative of tanh: 1 - tanh^2."""
        a = torch.tanh(z)
        return 1.0 - a ** 2

    def forward(
        self,
        s: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform state settling given input s.

        Parameters
        ----------
        s : torch.Tensor of shape (d_in,)
            Current input (stimulus) vector.
        z_prev : torch.Tensor of shape (d_z,), optional
            Previous latent state (for recurrent drive). If None, uses self.z_prev.
        observed_mask : torch.Tensor of shape (d_in,), optional
            Mask specifying which dimensions are observed (1.0) or target/unobserved (0.0).

        Returns
        -------
        a : torch.Tensor of shape (d_z,)
            Sparse latent activation after settling and kWTA.
        """
        if z_prev is None:
            z_prev = self.z_prev.clone()

        z = self.z.clone()

        for _ in range(self.n_settle):
            a = self.activation(z)

            # Apply kWTA during settling if sparsity is enabled
            if self.use_sparsity:
                a = kwta(a, self.k_sparse)

            # Predicted reconstruction
            s_hat = self.D @ a  # (d_in,)

            # Prediction error
            e = s - s_hat  # (d_in,)
            if observed_mask is not None:
                e = e * observed_mask

            # Drive terms
            d_rec = self.E @ s - z                          # recognition drive
            d_fb = (self.D.T @ e) * self.activation_deriv(z)  # feedback drive

            d_time = torch.zeros(self.d_z)
            if self.use_recurrent:
                d_time = self.R @ z_prev

            d_lat = torch.zeros(self.d_z)
            if self.use_lateral:
                d_lat = self.L @ a

            # Settling update
            delta_z = (
                d_rec
                + d_fb
                + self.rho * d_time
                + self.lambda_lat * d_lat
                - self.lambda_sparse * torch.sign(z)
            )
            z = z + self.eta_z * delta_z

            # Stability: clip to avoid runaway
            z = torch.clamp(z, -10.0, 10.0)

        # Final activation after settling
        a = self.activation(z)
        if self.use_sparsity:
            a = kwta(a, self.k_sparse)

        # Compute final prediction and error for metrics
        s_hat = self.D @ a
        e = s - s_hat
        if observed_mask is not None:
            e = e * observed_mask

        # Store state
        self.z_prev = self.z.clone()
        self.z = z.clone()
        self._last_error = e.clone()
        self._last_pred = s_hat.clone()
        self._last_sparsity = compute_sparsity_level(a)

        return a

    def update_weights(self, s: torch.Tensor, a: torch.Tensor) -> dict:
        """
        Apply local Hebbian weight updates after a forward pass.

        Parameters
        ----------
        s : torch.Tensor of shape (d_in,)
            Current input stimulus.
        a : torch.Tensor of shape (d_z,)
            Post-settling sparse activation.

        Returns
        -------
        dict
            Dictionary of weight delta norms for logging.
        """
        e = self._last_error
        z = self.z
        z_prev = self.z_prev

        # Generative update: ΔD = η_D * outer(e, a)
        delta_D = hebbian_generative_update(e, a, self.eta_D)
        self.D = self.D + delta_D

        # Recognition update: ΔE = η_E * outer(z - E@s, s)
        delta_E = hebbian_recognition_update(z, self.E, s, self.eta_E)
        self.E = self.E + delta_E

        # Recurrent update: ΔR = η_R * outer(z, z_prev)
        delta_R = torch.zeros_like(self.R)
        if self.use_recurrent:
            delta_R = hebbian_recurrent_update(z, z_prev, self.eta_R)
            self.R = self.R + delta_R

        # Update importance
        self.importance_D = update_importance(
            self.importance_D, delta_D, self.importance_decay
        )
        self.importance_E = update_importance(
            self.importance_E, delta_E, self.importance_decay
        )

        return {
            "delta_D_norm": delta_D.norm().item(),
            "delta_E_norm": delta_E.norm().item(),
            "delta_R_norm": delta_R.norm().item(),
        }

    @property
    def prediction_error(self) -> Optional[float]:
        """Last prediction error MSE."""
        if self._last_error is None:
            return None
        return (self._last_error ** 2).mean().item()

    @property
    def sparsity_level(self) -> float:
        """Last observed sparsity level (fraction of inactive units)."""
        return self._last_sparsity

    def reset_state(self):
        """Reset recurrent/temporal state (call between unrelated sequences)."""
        self.z = torch.zeros(self.d_z)
        self.z_prev = torch.zeros(self.d_z)

    def get_config(self) -> dict:
        """Return configuration dictionary for logging/reproducibility."""
        return {
            "d_in": self.d_in,
            "d_z": self.d_z,
            "k_sparse": self.k_sparse,
            "n_settle": self.n_settle,
            "eta_z": self.eta_z,
            "eta_D": self.eta_D,
            "eta_E": self.eta_E,
            "eta_R": self.eta_R,
            "rho": self.rho,
            "lambda_lat": self.lambda_lat,
            "lambda_sparse": self.lambda_sparse,
            "use_sparsity": self.use_sparsity,
            "use_recurrent": self.use_recurrent,
            "use_lateral": self.use_lateral,
        }

    def extra_repr(self) -> str:
        return (
            f"d_in={self.d_in}, d_z={self.d_z}, k_sparse={self.k_sparse}, "
            f"n_settle={self.n_settle}, sparsity={self.use_sparsity}, "
            f"recurrent={self.use_recurrent}"
        )
