"""
Raw AGNIS — src/agnis/core/predictive_hierarchy.py

PredictiveHierarchy: Phase 6 multi-level predictive coding hierarchy
with joint settling, temporal commitment strides, vertical precision gating,
and hierarchical neurogenesis/importance tracking.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
import math

from agnis.core.sparsity import kwta
from agnis.core.settling import joint_settle


class PredictiveHierarchy(nn.Module):
    """
    Phase 6 multi-level predictive coding hierarchy with joint settling.

    Parameters
    ----------
    d_input : int
        Dimensionality of the raw input stimulus.
    layer_dims : list of int
        Latent dimensions for each layer, e.g. [64, 32, 16] for 3 layers.
    k_sparse_per_layer : list of int
        kWTA k values per layer.
    commit_strides : list of int
        Temporal commitment stride per layer. Layer l commits every
        commit_strides[l] timesteps. E.g. [1, 4, 16].
    lambda_td : float
        Top-down conformity pressure strength.
    n_settle : int
        Number of synchronous joint settling iterations (Stage B).
    eta_z : float
        Settling step size.
    eta_d : float
        Base learning rate for decoder weight updates.
    eta_e : float
        Base learning rate for recognition weight updates.
    eta_r : float
        Base learning rate for recurrent weight updates.
    lr_decay : float
        Per-layer learning rate decay factor. Layer l uses eta * (lr_decay^l).
    use_precision_gating : bool
        Whether to apply vertical precision gating during weight updates.
    use_fatigue : bool
        Whether to apply latent fatigue (adaptation trace) during settling.
    fatigue_decay : float
        EMA decay rate for latent fatigue traces.
    gamma_fatigue : float
        Handicap multiplier for latent fatigue trace.
    importance_decay : float
        EMA decay rate for importance tracking.
    """

    def __init__(
        self,
        d_input: int,
        layer_dims: List[int],
        k_sparse_per_layer: List[int],
        commit_strides: List[int],
        lambda_td: float = 0.3,
        n_settle: int = 10,
        eta_z: float = 0.05,
        eta_d: float = 0.01,
        eta_e: float = 0.01,
        eta_r: float = 0.005,
        lr_decay: float = 0.1,
        use_precision_gating: bool = False,
        use_fatigue: bool = False,
        fatigue_decay: float = 0.9,
        gamma_fatigue: float = 0.5,
        importance_decay: float = 0.01,
    ):
        super().__init__()

        self.d_input = d_input
        self.layer_dims = list(layer_dims)
        self.n_layers = len(layer_dims)
        self.k_sparse_per_layer = list(k_sparse_per_layer)
        self.commit_strides = list(commit_strides)
        self.lambda_td = lambda_td
        self.n_settle = n_settle
        self.eta_z = eta_z
        self.eta_d = eta_d
        self.eta_e = eta_e
        self.eta_r = eta_r
        self.lr_decay = lr_decay
        self.use_precision_gating = use_precision_gating
        self.use_fatigue = use_fatigue
        self.fatigue_decay = fatigue_decay
        self.gamma_fatigue = gamma_fatigue
        self.importance_decay = importance_decay

        assert len(k_sparse_per_layer) == self.n_layers
        assert len(commit_strides) == self.n_layers

        # ── Per-layer dimensions: dim_below[l] is the input dimension to layer l
        self._dim_below = []
        for l in range(self.n_layers):
            if l == 0:
                self._dim_below.append(d_input)
            else:
                self._dim_below.append(layer_dims[l - 1])

        # ── Weight matrices (registered as buffers — updated manually via Hebbian rules)
        for l in range(self.n_layers):
            dim_l = layer_dims[l]
            dim_below_l = self._dim_below[l]

            # Recognition: E^l (dim_l x dim_below_l) — encodes error from below
            scale_e = 1.0 / math.sqrt(dim_below_l)
            E_l = torch.randn(dim_l, dim_below_l) * scale_e
            self.register_buffer(f"E_{l}", E_l)

            # Recurrent: R^l (dim_l x dim_l) — temporal state injection
            R_l = torch.randn(dim_l, dim_l) * 0.01
            self.register_buffer(f"R_{l}", R_l)

            # Lateral inhibition: L^l (dim_l x dim_l)
            L_l = torch.zeros(dim_l, dim_l)
            self.register_buffer(f"L_{l}", L_l)

            # Inter-layer decoder: D_inter^l (dim_below_l x dim_l)
            # D_inter[l] maps z^l -> prediction of level below l
            #   D_inter[0]: (d_input, dim_0) — layer 0 predicts raw input
            #   D_inter[l]: (dim_{l-1}, dim_l) — layer l predicts layer l-1
            scale_d = 1.0 / math.sqrt(dim_l)
            D_l = torch.randn(dim_below_l, dim_l) * scale_d
            self.register_buffer(f"D_inter_{l}", D_l)

        # ── Latent state vectors ──────────────────────────────────────────────
        for l in range(self.n_layers):
            dim_l = layer_dims[l]
            self.register_buffer(f"z_{l}", torch.zeros(dim_l))
            self.register_buffer(f"z_prev_{l}", torch.zeros(dim_l))

        # ── Error accumulators (running sum + count for mean) ─────────────────
        for l in range(self.n_layers):
            dim_below_l = self._dim_below[l]
            self.register_buffer(f"error_accum_{l}", torch.zeros(dim_below_l))
            self.register_buffer(f"error_count_{l}", torch.tensor(0, dtype=torch.long))

        # ── Neurogenesis states per layer ─────────────────────────────────────
        for l in range(self.n_layers):
            dim_l = layer_dims[l]
            dim_below_l = self._dim_below[l]
            self.register_buffer(f"maturity_{l}", torch.ones(dim_l))
            self.register_buffer(f"plasticity_{l}", torch.ones(dim_l))
            self.register_buffer(f"usage_{l}", torch.zeros(dim_l))
            self.register_buffer(f"age_{l}", torch.zeros(dim_l))
            self.register_buffer(f"importance_D_{l}", torch.zeros(dim_below_l, dim_l))
            self.register_buffer(f"importance_E_{l}", torch.zeros(dim_l, dim_below_l))
            self.register_buffer(f"importance_R_{l}", torch.zeros(dim_l, dim_l))
            self.register_buffer(f"fatigue_{l}", torch.zeros(dim_l))
            self.register_buffer(f"precision_ema_{l}", torch.tensor(1.0))

        # ── Step counter ──────────────────────────────────────────────────────
        self._step: int = 0

        # ── Error tracking ────────────────────────────────────────────────────
        self.per_layer_errors: List[float] = []

    # ── Accessors for named buffers ───────────────────────────────────────

    def _E(self, l: int) -> torch.Tensor:
        return getattr(self, f"E_{l}")

    def _R(self, l: int) -> torch.Tensor:
        return getattr(self, f"R_{l}")

    def _L(self, l: int) -> torch.Tensor:
        return getattr(self, f"L_{l}")

    def _D_inter(self, l: int) -> torch.Tensor:
        return getattr(self, f"D_inter_{l}")

    def _z(self, l: int) -> torch.Tensor:
        return getattr(self, f"z_{l}")

    def _z_prev(self, l: int) -> torch.Tensor:
        return getattr(self, f"z_prev_{l}")

    def _error_accum(self, l: int) -> torch.Tensor:
        return getattr(self, f"error_accum_{l}")

    def _error_count(self, l: int) -> torch.Tensor:
        return getattr(self, f"error_count_{l}")

    def _maturity(self, l: int) -> torch.Tensor:
        return getattr(self, f"maturity_{l}")

    def _plasticity(self, l: int) -> torch.Tensor:
        return getattr(self, f"plasticity_{l}")

    def _usage(self, l: int) -> torch.Tensor:
        return getattr(self, f"usage_{l}")

    def _age(self, l: int) -> torch.Tensor:
        return getattr(self, f"age_{l}")

    def _fatigue(self, l: int) -> torch.Tensor:
        return getattr(self, f"fatigue_{l}")

    def _precision_ema(self, l: int) -> torch.Tensor:
        return getattr(self, f"precision_ema_{l}")

    def _importance_D(self, l: int) -> torch.Tensor:
        return getattr(self, f"importance_D_{l}")

    def _importance_E(self, l: int) -> torch.Tensor:
        return getattr(self, f"importance_E_{l}")

    def _importance_R(self, l: int) -> torch.Tensor:
        return getattr(self, f"importance_R_{l}")

    def _set_z(self, l: int, val: torch.Tensor):
        self.register_buffer(f"z_{l}", val)

    def _set_z_prev(self, l: int, val: torch.Tensor):
        self.register_buffer(f"z_prev_{l}", val)

    def _set_error_accum(self, l: int, val: torch.Tensor):
        self.register_buffer(f"error_accum_{l}", val)

    def _set_error_count(self, l: int, val: torch.Tensor):
        self.register_buffer(f"error_count_{l}", val)

    def _set_E(self, l: int, val: torch.Tensor):
        self.register_buffer(f"E_{l}", val)

    def _set_R(self, l: int, val: torch.Tensor):
        self.register_buffer(f"R_{l}", val)

    def _set_L(self, l: int, val: torch.Tensor):
        self.register_buffer(f"L_{l}", val)

    def _set_D_inter(self, l: int, val: torch.Tensor):
        self.register_buffer(f"D_inter_{l}", val)

    def _set_maturity(self, l: int, val: torch.Tensor):
        self.register_buffer(f"maturity_{l}", val)

    def _set_plasticity(self, l: int, val: torch.Tensor):
        self.register_buffer(f"plasticity_{l}", val)

    def _set_usage(self, l: int, val: torch.Tensor):
        self.register_buffer(f"usage_{l}", val)

    def _set_age(self, l: int, val: torch.Tensor):
        self.register_buffer(f"age_{l}", val)

    def _set_fatigue(self, l: int, val: torch.Tensor):
        self.register_buffer(f"fatigue_{l}", val)

    def _set_precision_ema(self, l: int, val: torch.Tensor):
        self.register_buffer(f"precision_ema_{l}", val)

    def _set_importance_D(self, l: int, val: torch.Tensor):
        self.register_buffer(f"importance_D_{l}", val)

    def _set_importance_E(self, l: int, val: torch.Tensor):
        self.register_buffer(f"importance_E_{l}", val)

    def _set_importance_R(self, l: int, val: torch.Tensor):
        self.register_buffer(f"importance_R_{l}", val)

    # ── Bounded activations ────────────────────────────────────────────────

    def activation(self, z: torch.Tensor) -> torch.Tensor:
        """Bounded activation nonlinearity: tanh."""
        return torch.tanh(z)

    def activation_deriv(self, z: torch.Tensor) -> torch.Tensor:
        """Derivative of tanh: 1 - tanh^2."""
        a = torch.tanh(z)
        return 1.0 - a ** 2

    # ── Core forward pass ─────────────────────────────────────────────────

    def forward(self, s: torch.Tensor, t: int, observed_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Two-stage forward pass through the hierarchy.

        Parameters
        ----------
        s : torch.Tensor of shape (d_input,)
            Raw input stimulus.
        t : int
            Current global timestep.
        observed_mask : torch.Tensor, optional
            Observed dimensions (1.0 for observed, 0.0 for unobserved).

        Returns
        -------
        activations : list of torch.Tensor
            Latent activations [z_0, z_1, ..., z_{N-1}] after settling.
        """
        # Determine which layers commit at this timestep
        committing = [t % self.commit_strides[l] == 0 for l in range(self.n_layers)]

        # ── Accumulate bottom-up errors for non-committing layers ─────────
        self._accumulate_errors(s, observed_mask=observed_mask)

        # ── Stage A: Amortized feedforward initialization ─────────────────
        # For each committing layer, compute initial z from accumulated error
        for l in range(self.n_layers):
            if not committing[l]:
                continue  # keep held state

            # Get the error signal to drive this layer
            error_count = self._error_count(l).item()
            if error_count > 0:
                error_input = self._error_accum(l) / float(error_count)
            else:
                error_input = self._compute_bottom_up_error(l, s, observed_mask=observed_mask)

            # Save current z as z_prev before updating
            self._set_z_prev(l, self._z(l).clone())

            # Amortized init: E^l @ error_input + R^l @ z_prev^l
            z_init = self._E(l) @ error_input + self._R(l) @ self._z_prev(l)
            z_init = kwta(z_init, self.k_sparse_per_layer[l])
            self._set_z(l, z_init)

        # ── Stage B: Synchronous joint settling ───────────────────────────
        # Collect current z states into a list
        z_list = [self._z(l).clone() for l in range(self.n_layers)]
        z_prev_list = [self._z_prev(l).clone() for l in range(self.n_layers)]

        # Collect weight matrices into lists for the settling function
        E_list = [self._E(l) for l in range(self.n_layers)]
        R_list = [self._R(l) for l in range(self.n_layers)]
        L_list = [self._L(l) for l in range(self.n_layers)]
        D_inter_list = [self._D_inter(l) for l in range(self.n_layers)]

        # Setup fatigue biases if enabled
        bias_list = []
        for l in range(self.n_layers):
            if self.use_fatigue:
                bias_list.append(self.gamma_fatigue * self._fatigue(l))
            else:
                bias_list.append(None)

        z_settled = joint_settle(
            z_list=z_list,
            z_prev_list=z_prev_list,
            s=s,
            E_list=E_list,
            R_list=R_list,
            L_list=L_list,
            D_inter_list=D_inter_list,
            k_sparse_list=self.k_sparse_per_layer,
            committing=committing,
            lambda_td=self.lambda_td,
            eta_z=self.eta_z,
            n_settle=self.n_settle,
            bias_list=bias_list,
            observed_mask=observed_mask,
        )

        # ── Store settled states ──────────────────────────────────────────
        for l in range(self.n_layers):
            self._set_z(l, z_settled[l])

        # ── Reset error accumulators for layers that just committed ───────
        for l in range(self.n_layers):
            if committing[l]:
                self._set_error_accum(l, torch.zeros_like(self._error_accum(l)))
                self._set_error_count(l, torch.tensor(0, dtype=torch.long))

        # ── Track per-layer errors and update neurogenesis stats ──────────
        self.per_layer_errors = []
        for l in range(self.n_layers):
            e = self._compute_bottom_up_error(l, s)
            self.per_layer_errors.append((e ** 2).mean().item())

            # Update age, usage, maturity, fatigue
            z_l = z_settled[l]
            active = (z_l != 0.0).float()
            # Age
            new_age = torch.where(z_l != 0.0, torch.zeros_like(self._age(l)), self._age(l) + 1.0)
            self._set_age(l, new_age)
            # Usage
            self._set_usage(l, 0.99 * self._usage(l) + 0.01 * active)
            # Maturity (active units mature)
            self._set_maturity(l, self._maturity(l) + 0.1 * (1.0 - self._maturity(l)) * active)
            # Fatigue
            self._set_fatigue(l, self.fatigue_decay * self._fatigue(l) + (1.0 - self.fatigue_decay) * active)

        self._step = t

        return z_settled

    # ── Error computation helpers ─────────────────────────────────────────

    def _compute_bottom_up_error(
        self,
        l: int,
        s: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute bottom-up prediction error at layer l.

        e_below[0] = s - D_inter[0] @ z[0]
        e_below[l] = z[l-1] - D_inter[l] @ z[l]   for l > 0
        """
        # Make sure target activation uses tanh nonlinearity and maturity
        if l == 0:
            pred = self._D_inter(0) @ (self._maturity(0) * self.activation(self._z(0)))
            err = s - pred
            if observed_mask is not None:
                err = err * observed_mask
            return err
        else:
            pred = self._D_inter(l) @ (self._maturity(l) * self.activation(self._z(l)))
            return self._z(l - 1) - pred

    def _accumulate_errors(self, s: torch.Tensor, observed_mask: Optional[torch.Tensor] = None):
        """Accumulate bottom-up errors as running sum for all layers."""
        for l in range(self.n_layers):
            e = self._compute_bottom_up_error(l, s, observed_mask=observed_mask)
            self._set_error_accum(l, self._error_accum(l) + e)
            self._set_error_count(l, self._error_count(l) + 1)

    # ── Weight updates ────────────────────────────────────────────────────

    def update_all_weights(
        self,
        s: torch.Tensor,
        t: int,
    ) -> Dict[str, float]:
        """
        Apply local Hebbian weight updates to layers that committed at timestep t.

        Parameters
        ----------
        s : torch.Tensor of shape (d_input,)
            Raw input stimulus.
        t : int
            Current global timestep.

        Returns
        -------
        dict
            Weight delta norms per layer for logging.
        """
        from agnis.core.hebbian_rules import (
            compute_vertical_gate,
            decoder_update,
            encoder_update,
            recurrent_update,
            update_importance,
        )

        metrics: Dict[str, float] = {}

        for l in range(self.n_layers):
            # Only update layers that committed at this timestep
            if t % self.commit_strides[l] != 0:
                continue

            # Layer-specific learning rate with decay
            decay_factor = self.lr_decay ** l
            eta_d_l = self.eta_d * decay_factor
            eta_e_l = self.eta_e * decay_factor
            eta_r_l = self.eta_r * decay_factor

            z_l = self._z(l)
            z_prev_l = self._z_prev(l)
            a_l = self._maturity(l) * self.activation(z_l)

            # Bottom-up error at this layer
            e_below = self._compute_bottom_up_error(l, s)

            # Precision gating
            error_norm_sq = (e_below ** 2).mean().item()
            gate = 1.0
            if self.use_precision_gating:
                gate = compute_vertical_gate(
                    error_norm_sq=error_norm_sq,
                    precision_ema=self._precision_ema(l).item(),
                    beta=5.0,
                    theta=1.5,
                )

            # Update precision EMA
            new_ema = 0.95 * self._precision_ema(l) + 0.05 * error_norm_sq
            self._set_precision_ema(l, torch.as_tensor(new_ema, device=e_below.device))

            # ── Decoder update: ΔD_inter[l] = gate * eta_d * outer(e_below, a_l)
            delta_D = decoder_update(e_below, a_l, eta_d_l, gate)
            new_D = self._D_inter(l) + delta_D
            self._set_D_inter(l, new_D)

            # ── Recognition update: ΔE[l] = gate * eta_e * outer(z[l] - E[l]@e_below, e_below)
            # Input to recognition encoder is stimulus or e_below from below
            inp = s if l == 0 else e_below
            delta_E = encoder_update(z_l, self._E(l) @ inp, inp, eta_e_l, gate)
            new_E = self._E(l) + delta_E
            self._set_E(l, new_E)

            # ── Recurrent update: ΔR[l] = gate * eta_r * outer(z_l - R[l]@z_prev_l, z_prev_l)
            delta_R = recurrent_update(z_l, z_prev_l, self._R(l), eta_r_l, gate)
            new_R = self._R(l) + delta_R
            self._set_R(l, new_R)

            # ── Update importance tracking
            self._set_importance_D(l, update_importance(self._importance_D(l), delta_D, self.importance_decay))
            self._set_importance_E(l, update_importance(self._importance_E(l), delta_E, self.importance_decay))
            self._set_importance_R(l, update_importance(self._importance_R(l), delta_R, self.importance_decay))

            metrics[f"layer{l}/delta_D_norm"] = delta_D.norm().item()
            metrics[f"layer{l}/delta_E_norm"] = delta_E.norm().item()
            metrics[f"layer{l}/delta_R_norm"] = delta_R.norm().item()
            metrics[f"layer{l}/gate"] = gate
            metrics[f"layer{l}/precision_ema"] = self._precision_ema(l).item()

        return metrics

    # ── Output prediction ─────────────────────────────────────────────────

    def get_output_prediction(
        self,
        z: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Reconstruct the predicted input from layer 0's latent state.

        Parameters
        ----------
        z : list of torch.Tensor, optional
            Latent activations. If None, uses internal state.

        Returns
        -------
        torch.Tensor of shape (d_input,)
            Predicted input reconstruction: D_inter[0] @ (maturity * activation(z[0])).
        """
        if z is not None:
            z0 = z[0]
        else:
            z0 = self._z(0)
        return self._D_inter(0) @ (self._maturity(0) * self.activation(z0))

    # ── Neurogenesis: Grow & Prune ────────────────────────────────────────

    def grow_units(
        self,
        l: int,
        current_input: torch.Tensor,
        residual_error: torch.Tensor,
    ):
        """
        Birth a new unit in layer l.
        """
        from agnis.neurogenesis.unit_birth import birth_unit_in_hierarchy

        old_dim, new_dim = birth_unit_in_hierarchy(self, l, residual_error, current_input)

        device = self._z(l).device

        # Expand neurogenesis and importance state vectors
        self._set_maturity(l, torch.cat([self._maturity(l), torch.zeros(1, device=device)]))
        self._set_plasticity(l, torch.cat([self._plasticity(l), torch.ones(1, device=device)]))
        self._set_usage(l, torch.cat([self._usage(l), torch.zeros(1, device=device)]))
        self._set_age(l, torch.cat([self._age(l), torch.zeros(1, device=device)]))
        self._set_fatigue(l, torch.cat([self._fatigue(l), torch.zeros(1, device=device)]))

        self._set_importance_D(l, torch.cat([self._importance_D(l), torch.zeros(self._importance_D(l).shape[0], 1, device=device)], dim=1))
        self._set_importance_E(l, torch.cat([self._importance_E(l), torch.zeros(1, self._importance_E(l).shape[1], device=device)], dim=0))

        # R is expanded as 2D square matrix
        new_imp_R = torch.zeros(new_dim, new_dim, device=device)
        new_imp_R[:old_dim, :old_dim] = self._importance_R(l)
        self._set_importance_R(l, new_imp_R)

        # Expand the layer above's importance buffers if it exists
        if l < self.n_layers - 1:
            imp_D_above = self._importance_D(l + 1)
            self._set_importance_D(l + 1, torch.cat([imp_D_above, torch.zeros(1, imp_D_above.shape[1], device=device)], dim=0))

            imp_E_above = self._importance_E(l + 1)
            self._set_importance_E(l + 1, torch.cat([imp_E_above, torch.zeros(imp_E_above.shape[0], 1, device=device)], dim=1))

    def prune_units(
        self,
        min_age: int = 100,
        usage_threshold: float = 0.01,
        importance_threshold: float = 0.01,
        maturity_threshold: float = 0.5,
        parent_weight_threshold: float = 0.05,
    ):
        """
        Prune redundant/low-usage units across all layers.
        """
        for l in range(self.n_layers):
            dim_l = self.layer_dims[l]
            if dim_l <= 4:
                continue  # keep a minimum layer capacity

            keep_mask = torch.ones(dim_l, dtype=torch.bool, device=self._z(l).device)

            for j in range(dim_l):
                imp_j = self._importance_D(l)[:, j].norm().item()

                # Top-down dependency protection check
                parent_dependent = False
                if l < self.n_layers - 1:
                    # Norm of parent's incoming decoder weights for this unit
                    parent_weight_norm = self._D_inter(l + 1)[j, :].norm().item()
                    if parent_weight_norm > parent_weight_threshold:
                        parent_dependent = True

                if (self._age(l)[j].item() > min_age and
                    self._usage(l)[j].item() < usage_threshold and
                    imp_j < importance_threshold and
                    self._maturity(l)[j].item() < maturity_threshold and
                    not parent_dependent):
                    keep_mask[j] = False

            if keep_mask.all():
                continue

            keep_indices = torch.where(keep_mask)[0]
            n_pruned = dim_l - len(keep_indices)
            print(f"[Neurogenesis] Pruning layer {l}: {n_pruned} units. Capacity: {dim_l} -> {len(keep_indices)}")

            # Shrink layer l
            self._set_E(l, self._E(l)[keep_indices, :])
            self._set_D_inter(l, self._D_inter(l)[:, keep_indices])
            self._set_R(l, self._R(l)[keep_indices, :][:, keep_indices])
            self._set_L(l, self._L(l)[keep_indices, :][:, keep_indices])

            self._set_z(l, self._z(l)[keep_indices])
            self._set_z_prev(l, self._z_prev(l)[keep_indices])

            self._set_maturity(l, self._maturity(l)[keep_indices])
            self._set_plasticity(l, self._plasticity(l)[keep_indices])
            self._set_usage(l, self._usage(l)[keep_indices])
            self._set_age(l, self._age(l)[keep_indices])
            self._set_fatigue(l, self._fatigue(l)[keep_indices])

            self._set_importance_D(l, self._importance_D(l)[:, keep_indices])
            self._set_importance_E(l, self._importance_E(l)[keep_indices, :])
            self._set_importance_R(l, self._importance_R(l)[keep_indices, :][:, keep_indices])

            # Shrink layer above (l+1) dependencies if they exist
            if l < self.n_layers - 1:
                self._set_D_inter(l + 1, self._D_inter(l + 1)[keep_indices, :])
                self._set_E(l + 1, self._E(l + 1)[:, keep_indices])
                self._set_error_accum(l + 1, self._error_accum(l + 1)[keep_indices])
                self._set_importance_D(l + 1, self._importance_D(l + 1)[keep_indices, :])
                self._set_importance_E(l + 1, self._importance_E(l + 1)[:, keep_indices])

            self.layer_dims[l] = len(keep_indices)
            self.k_sparse_per_layer[l] = min(self.k_sparse_per_layer[l], len(keep_indices))

            # Keep _dim_below in sync so reset_state() allocates correct sizes
            if l + 1 < self.n_layers:
                self._dim_below[l + 1] = len(keep_indices)

    # ── State management ──────────────────────────────────────────────────

    def reset_state(self):
        """Reset all latent states, error accumulators, and step counter."""
        for l in range(self.n_layers):
            dim_l = self.layer_dims[l]
            dim_below_l = self._dim_below[l]
            self._set_z(l, torch.zeros(dim_l))
            self._set_z_prev(l, torch.zeros(dim_l))
            self._set_error_accum(l, torch.zeros(dim_below_l))
            self._set_error_count(l, torch.tensor(0, dtype=torch.long))
            self._set_fatigue(l, torch.zeros(dim_l))
        self._step = 0
        self.per_layer_errors = []

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def total_prediction_error(self) -> float:
        """Sum of prediction errors across all layers."""
        if not self.per_layer_errors:
            return 0.0
        return sum(self.per_layer_errors)

    @property
    def per_layer_prediction_error(self) -> List[float]:
        """Prediction error (MSE) per layer."""
        return list(self.per_layer_errors)

    def get_config(self) -> dict:
        """Return full hierarchy configuration."""
        return {
            "d_input": self.d_input,
            "layer_dims": self.layer_dims,
            "n_layers": self.n_layers,
            "k_sparse_per_layer": self.k_sparse_per_layer,
            "commit_strides": self.commit_strides,
            "lambda_td": self.lambda_td,
            "n_settle": self.n_settle,
            "eta_z": self.eta_z,
            "eta_d": self.eta_d,
            "eta_e": self.eta_e,
            "eta_r": self.eta_r,
            "lr_decay": self.lr_decay,
        }

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, layer_dims={self.layer_dims}, "
            f"k_sparse={self.k_sparse_per_layer}, strides={self.commit_strides}, "
            f"n_settle={self.n_settle}, lambda_td={self.lambda_td}"
        )

