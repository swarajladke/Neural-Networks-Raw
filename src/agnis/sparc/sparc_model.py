"""
Raw AGNIS — src/agnis/sparc/sparc_model.py

SPARC v0.2 Sequence Model Wrapper.
Ties columns and routers together under a sequential step interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
from agnis.sparc.column import PredictiveColumn
from agnis.sparc.router import TaskIDOracleRouter, NearestPrototypeRouter


class SPARCSequenceModel(nn.Module):
    """
    SPARC Sequence Model.
    Coordinates multiple PredictiveColumns, routes inputs via task routers,
    tracks latent recurrent history, and guarantees parameter isolation.
    """

    def __init__(
        self,
        num_columns: int,
        d_input: int,
        d_latent: int,
        d_output: int,
        alpha: float = 0.01,
        beta: float = 0.5,
        eta_D: float = 0.01,
        eta_R: float = 0.01,
        eta_Q: float = 0.01,
        step_c: float = 0.5,
        n_settle: int = 15,
        routing_mode: str = "task_id_oracle",
        decay_factor: float = 0.9,
    ):
        super().__init__()
        self.num_columns = num_columns
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_output = d_output
        self.routing_mode = routing_mode
        self.decay_factor = decay_factor

        # 1. Instantiate the column bank
        self.columns = nn.ModuleList(
            [
                PredictiveColumn(
                    d_input=d_input,
                    d_latent=d_latent,
                    d_output=d_output,
                    alpha=alpha,
                    beta=beta,
                    eta_D=eta_D,
                    eta_R=eta_R,
                    eta_Q=eta_Q,
                    step_c=step_c,
                    n_settle=n_settle,
                )
                for _ in range(num_columns)
            ]
        )

        # 2. Instantiate task routers
        if routing_mode == "task_id_oracle":
            self.router = TaskIDOracleRouter()
        elif routing_mode == "nearest_prototype":
            self.router = NearestPrototypeRouter(d_latent=d_input, num_columns=num_columns)
        elif routing_mode == "minimum_energy":
            from agnis.sparc.minimum_energy_router import MinimumEnergyRouter
            self.router = MinimumEnergyRouter(self.columns, num_columns)
        elif routing_mode in [
            "supervised_router",
            "energy_distilled_router",
            "learned_router_no_distill",
            "learned_router_distill",
            "learned_router_mixture",
        ]:
            from agnis.sparc.learned_router import DifferentiableTopKRouter
            self.router = DifferentiableTopKRouter(
                d_input=d_input,
                num_columns=num_columns,
                temperature=1.0,
                route_inertia=0.9,
                context_decay=0.9,
            )
        else:
            raise ValueError(f"Unknown routing mode: {routing_mode}")

        # Instantiate persistent MinimumEnergyRouter for energy teaching & diagnostics
        from agnis.sparc.minimum_energy_router import MinimumEnergyRouter
        self.energy_teacher = MinimumEnergyRouter(self.columns, num_columns)

        # 3. Persistent Recurrent Latent buffers (one per column)
        self.register_buffer("h_prev", torch.zeros(num_columns, d_latent))

        # 4. External Router Context and Logit Inertia states (one per sequence)
        self.register_buffer("context_state", torch.zeros(d_input))
        self.register_buffer("smoothed_logits", torch.zeros(num_columns))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper to store persistent state as buffers."""
        setattr(self, name, tensor)

    def reset_states(self):
        """Reset latent recurrent history and router states to zero (at sequence boundaries)."""
        self.h_prev.zero_()
        self.context_state.zero_()
        self.smoothed_logits.zero_()

    def freeze_experts(self):
        """Freezes all columns, heads, calibration, and prototype parameters."""
        for column in self.columns:
            for param in column.parameters():
                param.requires_grad = False
            column.eval()

        if isinstance(self.router, NearestPrototypeRouter):
            self.router.requires_grad_(False)

    def forward_step(
        self, z: torch.Tensor, target: torch.Tensor, task_id: int = None, is_training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Processes a single token z_t in the sequence:
        1. Select column via router.
        2. Settle active column state h_t.
        3. If training, run local error updates on D and R, and train Q.
        4. Decay inactive column states.
        5. Return readout logits and settling diagnostics.
        """
        # Step 1: Routing
        col_idx = 0
        routing_weights = None
        soft_probs = None

        # Snapshot external states
        context_state = self.context_state.clone()
        smoothed_logits = self.smoothed_logits.clone()

        if self.routing_mode == "task_id_oracle":
            if task_id is None:
                raise ValueError("task_id is required for task_id_oracle routing mode.")
            col_idx = self.router.route(task_id, self.num_columns)
            routing_weights = torch.zeros(self.num_columns, device=z.device)
            routing_weights[col_idx] = 1.0
            soft_probs = routing_weights.clone()
        elif self.routing_mode == "nearest_prototype":
            col_idx = self.router.route(z)
            if is_training:
                self.router.update_prototype(col_idx, z)
            routing_weights = torch.zeros(self.num_columns, device=z.device)
            routing_weights[col_idx] = 1.0
            soft_probs = routing_weights.clone()
        elif self.routing_mode == "minimum_energy":
            col_idx, settled_candidates, calibrated_energies, candidate_priors = self.energy_teacher.route_step(
                z, list(self.h_prev), self.decay_factor
            )
            routing_weights = torch.zeros(self.num_columns, device=z.device)
            routing_weights[col_idx] = 1.0
            soft_probs = routing_weights.clone()
        elif self.routing_mode in [
            "supervised_router",
            "energy_distilled_router",
            "learned_router_no_distill",
            "learned_router_distill",
            "learned_router_mixture",
        ]:
            routing_weights, soft_probs, new_context, new_smoothed_logits = self.router.route(
                z, context_state, smoothed_logits
            )
            # Update external states safely (detached)
            self.context_state = new_context.detach()
            self.smoothed_logits = new_smoothed_logits.detach()
            col_idx = int(routing_weights.argmax(dim=-1).item())

        # Compute candidate priors for all columns (decay losers exactly once)
        candidate_priors = [self.decay_factor * self.h_prev[j].clone() for j in range(self.num_columns)]

        # Step 2: Settling & Predictions
        readout_loss = 0.0

        if self.routing_mode == "learned_router_mixture" and is_training:
            # Router C mixture: Settle all columns and mix predicted probabilities
            settled_states = []
            raw_logits_list = []
            diagnostics = {"steps_taken": 0}

            for j in range(self.num_columns):
                h_settled, diag = self.columns[j].settle(z, candidate_priors[j])
                settled_states.append(h_settled)
                raw_logits_list.append(self.columns[j].readout(h_settled))
                diagnostics["steps_taken"] += diag["steps_taken"]

            diagnostics["steps_taken"] /= self.num_columns

            # Mix probabilities: p(y_t) = sum_j r_{t,j} p_j(y_t)
            column_log_probs = torch.stack([F.log_softmax(lg, dim=-1) for lg in raw_logits_list], dim=0)
            log_routes = torch.log(soft_probs.clamp_min(1e-8))
            mixture_log_probs = torch.logsumexp(log_routes.unsqueeze(-1) + column_log_probs, dim=0).unsqueeze(0)

            # Advance all column states
            for j in range(self.num_columns):
                self.h_prev[j] = settled_states[j].detach()

            logits = mixture_log_probs
        else:
            # Single-column execution (default top-1)
            h_prev_active = candidate_priors[col_idx]
            h_settled, diagnostics = self.columns[col_idx].settle(z, h_prev_active)

            # Run local weight updates if in training mode
            if is_training:
                self.columns[col_idx].update_predictive_parameters(z, h_settled, h_prev_active)
                readout_loss = self.columns[col_idx].update_readout(h_settled, target)

            # Commit winner state, decay all losers
            for j in range(self.num_columns):
                if j == col_idx:
                    self.h_prev[j] = h_settled.detach()
                else:
                    self.h_prev[j] = candidate_priors[j]

            logits = self.columns[col_idx].readout(h_settled.detach()).unsqueeze(0)

        # Counterfactual diagnostics (no state mutations)
        with torch.no_grad():
            oracle_col = task_id if task_id is not None else 0
            
            # Settle selected counterfactually from snapshot
            h_sel, _ = self.columns[col_idx].settle(z, candidate_priors[col_idx])
            loss_sel = F.cross_entropy(self.columns[col_idx].readout(h_sel).unsqueeze(0), target.view(-1)).item()
            
            # Settle oracle counterfactually from snapshot
            h_orac, _ = self.columns[oracle_col].settle(z, candidate_priors[oracle_col])
            loss_orac = F.cross_entropy(self.columns[oracle_col].readout(h_orac).unsqueeze(0), target.view(-1)).item()
            
            task_loss_routing_regret = loss_sel - loss_orac

        diagnostics.update(
            {
                "active_column": col_idx,
                "readout_loss": readout_loss,
                "task_loss_routing_regret": task_loss_routing_regret,
                "routing_weights": routing_weights if routing_weights is not None else torch.zeros(self.num_columns),
            }
        )

        return logits, diagnostics
