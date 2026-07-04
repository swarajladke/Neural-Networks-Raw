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
        R_update_enabled: bool = True,
        R_drive_enabled: bool = True,
        spectral_radius_max: float = 1.0,
        eta_m: float = 0.1,
        max_latent_dim: int = 128,
        maturity_enabled: bool = True,
        output_slice_start: Optional[int] = None,
        use_softmax_output: bool = False,
        use_fatigue: bool = False,
        fatigue_decay: float = 0.9,
        gamma_fatigue: float = 0.5,
        use_precision_gating: bool = False,
        gate_alpha_min: float = 0.2,
        gate_alpha_max: float = 0.8,
        gate_beta: float = 1.0,
        gate_ema: float = 0.05,
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
        self.R_update_enabled = R_update_enabled
        self.R_drive_enabled = R_drive_enabled
        self.spectral_radius_max = spectral_radius_max
        self.eta_m = eta_m
        self.max_latent_dim = max_latent_dim
        self.maturity_enabled = maturity_enabled

        # ── Softmax output error geometry (cross-entropy gradient, local) ─────
        # With use_softmax_output, the error on the target slice becomes
        # e_y = y - softmax(D_y @ a): the exact CE gradient at the output.
        # Prevents the Hebbian decoder from collapsing to the marginal
        # symbol distribution E[y] under MSE on one-hot targets.
        self.output_slice_start = output_slice_start
        self.use_softmax_output = use_softmax_output

        # ── Latent fatigue (adaptation trace; kWTA limit-cycle breaker) ──────
        # f <- lambda_a * f + (1 - lambda_a) * 1[active]
        # kWTA selection score = |a| - gamma_f * f
        self.use_fatigue = use_fatigue
        self.fatigue_decay = fatigue_decay
        self.gamma_fatigue = gamma_fatigue
        self.fatigue = torch.zeros(d_z)

        # ── Precision-weighted E/R gating (Kalman-gain-like) ─────────────────
        # alpha_t = a_min + (a_max - a_min) * sigmoid(beta * zscore(surprise))
        # delta_z = 2*alpha*d_rec + d_fb + 2*(1-alpha)*rho*d_time + ...
        # alpha = 0.5 exactly recovers the ungated dynamics.
        self.use_precision_gating = use_precision_gating
        self.gate_alpha_min = gate_alpha_min
        self.gate_alpha_max = gate_alpha_max
        self.gate_beta = gate_beta
        self.gate_ema = gate_ema
        self.surprise_mean = 0.0
        self.surprise_var = 1.0
        self.last_gate = 0.5

        # Neurogenesis state trackers (registered as PyTorch buffers for device/serialization safety)
        self.register_buffer("maturity", torch.ones(d_z))
        self.register_buffer("plasticity", torch.ones(d_z))
        self.register_buffer("usage", torch.zeros(d_z))
        self.register_buffer("age", torch.zeros(d_z))

        # ── Weight matrices (registered as buffers — updated manually via Hebbian rules)
        # Generative: D (d_in x d_z) — decodes latent to input space
        self.register_buffer("D", torch.nn.init.xavier_uniform_(
            torch.empty(d_in, d_z)
        ))
        # Recognition: E (d_z x d_in) — encodes input to latent space
        self.register_buffer("E", torch.nn.init.xavier_uniform_(
            torch.empty(d_z, d_in)
        ))
        # Recurrent: R (d_z x d_z) — temporal state injection
        R_temp = torch.zeros(d_z, d_z)
        torch.nn.init.orthogonal_(R_temp)
        self.register_buffer("R", R_temp * 0.1)  # start small and stable

        # Lateral: L (d_z x d_z) — sparse inhibitory connections
        # Off-diagonal negative, diagonal zero
        self.register_buffer("L", torch.zeros(d_z, d_z))
        if use_lateral:
            self._init_lateral_connections()

        # ── Per-weight importance (d_in x d_z for D, d_z x d_in for E) ───────
        self.register_buffer("importance_D", torch.zeros(d_in, d_z))
        self.register_buffer("importance_E", torch.zeros(d_z, d_in))
        self.register_buffer("importance_R", torch.zeros(d_z, d_z))

        # ── Latent state ──────────────────────────────────────────────────────
        self.register_buffer("z", torch.zeros(d_z))
        self.register_buffer("z_prev", torch.zeros(d_z))

        # ── Metrics ───────────────────────────────────────────────────────────
        self.recurrent_drive_norms = []
        self.R_update_norms = []
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

        # Precision gate: balance recognition (E) vs recurrent (R) drives.
        # alpha = 0.5 recovers the ungated dynamics exactly (gains = 1).
        alpha = self._compute_gate()
        self.last_gate = alpha
        if self.use_precision_gating:
            gain_rec = 2.0 * alpha
            gain_time = 2.0 * (1.0 - alpha)
        else:
            gain_rec = 1.0
            gain_time = 1.0

        for _ in range(self.n_settle):
            a = self.maturity * self.activation(z)

            # Apply kWTA during settling if sparsity is enabled
            # (fatigue-biased competition when use_fatigue is set)
            a = self._sparsify(a)

            # Predicted reconstruction (softmax on target slice if enabled)
            s_hat = self._predict_s(a)  # (d_in,)

            # Prediction error
            e = s - s_hat  # (d_in,)
            if observed_mask is not None:
                e = e * observed_mask

            # Drive terms
            d_rec = self.E @ s - z                          # recognition drive
            d_fb = (self.D.T @ e) * self.activation_deriv(z)  # feedback drive

            d_time = torch.zeros(self.d_z)
            if self.use_recurrent and self.R_drive_enabled:
                a_prev = self.maturity * self.activation(z_prev)
                if self.use_sparsity:
                    a_prev = kwta(a_prev, self.k_sparse)
                d_time = self.R @ a_prev
                self.recurrent_drive_norms.append(d_time.norm().item())

            d_lat = torch.zeros(self.d_z)
            if self.use_lateral:
                d_lat = self.L @ a

            # Settling update (precision-gated recognition vs recurrent)
            delta_z = (
                gain_rec * d_rec
                + d_fb
                + gain_time * self.rho * d_time
                + self.lambda_lat * d_lat
                - self.lambda_sparse * torch.sign(z)
            )
            z = z + self.eta_z * delta_z

            # Stability: clip to avoid runaway
            z = torch.clamp(z, -10.0, 10.0)

        # Final activation after settling
        a = self.maturity * self.activation(z)
        a = self._sparsify(a)

        # Compute final prediction and error for metrics
        s_hat = self._predict_s(a)
        e = s - s_hat
        if observed_mask is not None:
            e = e * observed_mask

        # Fatigue trace: only track the actual trajectory (training /
        # state-advance steps, observed_mask is None), not masked query
        # lookups, so evaluation predictions have no side effects here.
        if self.use_fatigue and observed_mask is None:
            active = (a != 0).float()
            self.fatigue = self.fatigue_decay * self._fatigue_vec() \
                + (1.0 - self.fatigue_decay) * active

        # Store state
        self.z_prev = self.z.clone()
        self.z = z.clone()
        self._last_error = e.clone()
        self._last_mask = observed_mask.clone() if observed_mask is not None else None
        self._last_pred = s_hat.clone()
        self._last_sparsity = compute_sparsity_level(a)

        return a

    def _fatigue_vec(self) -> torch.Tensor:
        """Fatigue trace, resized on demand after neurogenesis growth/pruning."""
        if self.fatigue.shape[0] != self.d_z:
            f = torch.zeros(self.d_z)
            n = min(self.fatigue.shape[0], self.d_z)
            f[:n] = self.fatigue[:n]
            self.fatigue = f
        return self.fatigue

    def _sparsify(self, a: torch.Tensor) -> torch.Tensor:
        """Apply kWTA, optionally handicapping recently active units (fatigue)."""
        if not self.use_sparsity:
            return a
        bias = self.gamma_fatigue * self._fatigue_vec() if self.use_fatigue else None
        return kwta(a, self.k_sparse, bias=bias)

    def _predict_s(self, a: torch.Tensor) -> torch.Tensor:
        """Generative prediction of s from activation a.

        With softmax output enabled, the target slice is normalized so the
        resulting error e_y = y - softmax(logits_y) is the cross-entropy
        gradient at the output (fully local, no autograd).
        """
        s_hat = self.D @ a
        if self.use_softmax_output and self.output_slice_start is not None:
            sl = self.output_slice_start
            s_hat = torch.cat([s_hat[:sl], torch.softmax(s_hat[sl:], dim=-1)])
        return s_hat

    def _compute_gate(self) -> float:
        """Precision-weighted gate alpha_t from standardized surprise.

        Uses the previous timestep's prediction error as surprise u_t,
        standardized by online EMA mean/variance:
        alpha_t = a_min + (a_max - a_min) * sigmoid(beta * zscore(u_t)).
        Returns 0.5 (neutral) when gating is disabled or no history exists.
        """
        if not self.use_precision_gating or self._last_error is None:
            return 0.5
        u = (self._last_error ** 2).mean().item()
        std = max(self.surprise_var, 1e-8) ** 0.5
        zscore = (u - self.surprise_mean) / std
        x = max(min(self.gate_beta * zscore, 30.0), -30.0)
        gate = float(torch.sigmoid(torch.tensor(x)).item())
        alpha = self.gate_alpha_min + (self.gate_alpha_max - self.gate_alpha_min) * gate
        # Online EMA update of surprise statistics
        lam = self.gate_ema
        self.surprise_mean = (1.0 - lam) * self.surprise_mean + lam * u
        self.surprise_var = (1.0 - lam) * self.surprise_var + lam * (u - self.surprise_mean) ** 2
        return alpha

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

        # Recurrent update
        delta_R = torch.zeros_like(self.R)
        if self.use_recurrent and self.R_update_enabled:
            # error-driven local recurrent learning: ΔR = η_R * outer(z - R @ a_prev, a_prev)
            a_prev = self.activation(z_prev.detach())
            if self.use_sparsity:
                a_prev = kwta(a_prev, self.k_sparse)
            z_target = z.detach()
            z_pred = self.R @ a_prev
            r_error = z_target - z_pred
            delta_R = self.eta_R * torch.outer(r_error, a_prev)
            self.R = self.R + delta_R
            self.normalize_R()
            self.R_update_norms.append(delta_R.norm().item())

        # Compute error before and after update to calculate maturity update
        err_before = e.norm().item()
        s_hat_after = self._predict_s(a)
        e_after = s - s_hat_after
        if self._last_mask is not None:
            e_after = e_after * self._last_mask
        err_after = e_after.norm().item()

        delta_err = max(0.0, err_before - err_after)

        # Update maturity vector
        if self.maturity_enabled:
            self.maturity = torch.clamp(self.maturity + self.eta_m * delta_err * a, 0.0, 1.0)

        # Track usage and age
        self.age = self.age + 1.0
        self.usage = 0.99 * self.usage + 0.01 * a

        # Update importance
        from agnis.core.hebbian_rules import update_importance
        self.importance_D = update_importance(
            self.importance_D, delta_D, self.importance_decay
        )
        self.importance_E = update_importance(
            self.importance_E, delta_E, self.importance_decay
        )
        if self.use_recurrent and self.R_update_enabled:
            self.importance_R = update_importance(
                self.importance_R, delta_R, self.importance_decay
            )

        return {
            "delta_D_norm": delta_D.norm().item(),
            "delta_E_norm": delta_E.norm().item(),
            "delta_R_norm": delta_R.norm().item(),
        }

    def normalize_R(self):
        """Clamp/normalize R to maintain spectral stability."""
        R_norm = torch.linalg.matrix_norm(self.R, ord=2)
        if R_norm > self.spectral_radius_max:
            self.R = self.R * (self.spectral_radius_max / (R_norm + 1e-8))

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

    def grow_units(self, k: int, current_input: torch.Tensor, residual_error: torch.Tensor):
        """Autonomously add k new units to the predictive cell."""
        if self.d_z + k > self.max_latent_dim:
            k = self.max_latent_dim - self.d_z
        if k <= 0:
            return

        print(f"[Neurogenesis] Spawning {k} new units. Capacity: {self.d_z} -> {self.d_z + k}")

        # 1. Expand Generative D: concat columns (d_in, k)
        D_born_list = []
        for _ in range(k):
            noise = torch.randn(self.d_in) * 0.05
            vec = residual_error + noise
            v_norm = vec.norm().item()
            if v_norm > 1e-8:
                D_born_list.append(vec / v_norm)
            else:
                D_born_list.append(torch.randn(self.d_in) * 0.1)
        D_born = torch.stack(D_born_list, dim=1)
        self.D = torch.cat([self.D, D_born], dim=1)

        # 2. Expand Recognition E: concat rows (k, d_in)
        E_born_list = []
        for _ in range(k):
            noise = torch.randn(self.d_in) * 0.05
            vec = current_input + noise
            v_norm = vec.norm().item()
            if v_norm > 1e-8:
                E_born_list.append(vec / v_norm)
            else:
                E_born_list.append(torch.randn(self.d_in) * 0.1)
        E_born = torch.stack(E_born_list, dim=0)
        self.E = torch.cat([self.E, E_born], dim=0)

        # 3. Expand Recurrent R: both dimensions (d_z + k, d_z + k)
        R_new = torch.zeros(self.d_z + k, self.d_z + k)
        R_new[:self.d_z, :self.d_z] = self.R
        R_new[self.d_z:, :] = torch.randn(k, self.d_z + k) * 0.05
        R_new[:, self.d_z:] = torch.randn(self.d_z + k, k) * 0.05
        self.R = R_new

        # 4. Expand Lateral L: both dimensions (d_z + k, d_z + k)
        L_new = torch.zeros(self.d_z + k, self.d_z + k)
        L_new[:self.d_z, :self.d_z] = self.L
        sparsity = 0.8
        new_mask = torch.rand(self.d_z + k, self.d_z + k) > sparsity
        new_mask.fill_diagonal_(False)
        update_mask = torch.zeros(self.d_z + k, self.d_z + k, dtype=torch.bool)
        update_mask[self.d_z:, :] = True
        update_mask[:, self.d_z:] = True
        combined_mask = new_mask & update_mask
        L_new[combined_mask] = torch.empty(combined_mask.sum()).uniform_(-0.2, 0.0)
        self.L = L_new

        # 5. Expand Importance trackers
        self.importance_D = torch.cat([self.importance_D, torch.zeros(self.d_in, k)], dim=1)
        self.importance_E = torch.cat([self.importance_E, torch.zeros(k, self.d_in)], dim=0)
        importance_R_new = torch.zeros(self.d_z + k, self.d_z + k)
        importance_R_new[:self.d_z, :self.d_z] = self.importance_R
        self.importance_R = importance_R_new

        # 6. Expand neurogenesis trackers
        maturity_val = 0.0 if self.maturity_enabled else 1.0
        self.maturity = torch.cat([self.maturity, torch.full((k,), maturity_val)])
        self.plasticity = torch.cat([self.plasticity, torch.ones(k)])
        self.usage = torch.cat([self.usage, torch.zeros(k)])
        self.age = torch.cat([self.age, torch.zeros(k)])

        # 7. Expand current latent state vectors
        self.z = torch.cat([self.z, torch.zeros(k)])
        self.z_prev = torch.cat([self.z_prev, torch.zeros(k)])

        # 8. Update latent dim size
        self.d_z = self.d_z + k

        # 9. Re-register buffers to update PyTorch's internal state
        self.register_buffer("D", self.D)
        self.register_buffer("E", self.E)
        self.register_buffer("R", self.R)
        self.register_buffer("L", self.L)
        self.register_buffer("importance_D", self.importance_D)
        self.register_buffer("importance_E", self.importance_E)
        self.register_buffer("importance_R", self.importance_R)
        self.register_buffer("maturity", self.maturity)
        self.register_buffer("plasticity", self.plasticity)
        self.register_buffer("usage", self.usage)
        self.register_buffer("age", self.age)
        self.register_buffer("z", self.z)
        self.register_buffer("z_prev", self.z_prev)

    def prune_units(self, min_age: int = 100, usage_threshold: float = 0.01, importance_threshold: float = 0.01, maturity_threshold: float = 0.5):
        """Autonomously prune low-usage, low-importance, immature units."""
        keep_mask = torch.ones(self.d_z, dtype=torch.bool)
        for j in range(self.d_z):
            imp_j = self.importance_D[:, j].norm().item()
            if (self.age[j].item() > min_age and 
                self.usage[j].item() < usage_threshold and 
                imp_j < importance_threshold and 
                self.maturity[j].item() < maturity_threshold):
                keep_mask[j] = False

        if keep_mask.all():
            return

        keep_indices = torch.where(keep_mask)[0]
        n_pruned = self.d_z - len(keep_indices)
        print(f"[Neurogenesis] Pruned {n_pruned} redundant units. Capacity: {self.d_z} -> {len(keep_indices)}")

        self.D = self.D[:, keep_indices]
        self.E = self.E[keep_indices, :]
        self.R = self.R[keep_indices, :][:, keep_indices]
        self.L = self.L[keep_indices, :][:, keep_indices]
        
        self.importance_D = self.importance_D[:, keep_indices]
        self.importance_E = self.importance_E[keep_indices, :]
        self.importance_R = self.importance_R[keep_indices, :][:, keep_indices]

        self.maturity = self.maturity[keep_indices]
        self.plasticity = self.plasticity[keep_indices]
        self.usage = self.usage[keep_indices]
        self.age = self.age[keep_indices]

        self.z = self.z[keep_indices]
        self.z_prev = self.z_prev[keep_indices]

        self.d_z = len(keep_indices)
        self.k_sparse = min(self.k_sparse, self.d_z)

        # Re-register buffers to update PyTorch's internal state
        self.register_buffer("D", self.D)
        self.register_buffer("E", self.E)
        self.register_buffer("R", self.R)
        self.register_buffer("L", self.L)
        self.register_buffer("importance_D", self.importance_D)
        self.register_buffer("importance_E", self.importance_E)
        self.register_buffer("importance_R", self.importance_R)
        self.register_buffer("maturity", self.maturity)
        self.register_buffer("plasticity", self.plasticity)
        self.register_buffer("usage", self.usage)
        self.register_buffer("age", self.age)
        self.register_buffer("z", self.z)
        self.register_buffer("z_prev", self.z_prev)

    def reset_state(self):
        """Reset recurrent/temporal state (call between unrelated sequences)."""
        self.z = torch.zeros(self.d_z)
        self.z_prev = torch.zeros(self.d_z)
        self.recurrent_drive_norms.clear()
        self.R_update_norms.clear()

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
