"""
Raw AGNIS — src/agnis/sparc/sparc_model.py

SPARC v0.1 Sequence Model Wrapper.
Ties columns and routers together under a sequential step interface.
"""

import torch
import torch.nn as nn
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
        else:
            raise ValueError(f"Unknown routing mode: {routing_mode}")

        # 3. Persistent Recurrent Latent buffers (one per column)
        self.register_buffer("h_prev", torch.zeros(num_columns, d_latent))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper to store persistent state as buffers."""
        setattr(self, name, tensor)

    def reset_states(self):
        """Reset latent recurrent history to zero (at sequence boundaries)."""
        self.h_prev.zero_()

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
        if self.routing_mode == "task_id_oracle":
            if task_id is None:
                raise ValueError("task_id is required for task_id_oracle routing mode.")
            col_idx = self.router.route(task_id, self.num_columns)
        elif self.routing_mode == "nearest_prototype":
            col_idx = self.router.route(z)
            # Update prototype centroid during training
            if is_training:
                self.router.update_prototype(col_idx, z)
        else:
            col_idx = 0

        active_col = self.columns[col_idx]

        # Step 2: Settle active column state h_t
        h_prev_active = self.h_prev[col_idx].clone()
        h_settled, diagnostics = active_col.settle(z, h_prev_active)

        # Step 3: Run local weight updates if in training mode
        readout_loss = 0.0
        if is_training:
            # Update prediction decoder D and recurrence R via local predictive errors
            active_col.update_predictive_parameters(z, h_settled, h_prev_active)
            # Train column-local readout head Q via supervised backprop (detached latent)
            readout_loss = active_col.update_readout(h_settled, target)

        # Step 4: Update latent state history & apply state decays
        self.h_prev.data *= self.decay_factor  # Decay all inactive column states
        self.h_prev[col_idx] = h_settled.detach()  # Save active column's settled state

        # Step 5: Readout prediction
        logits = active_col.readout(h_settled.detach())

        diagnostics.update(
            {"active_column": col_idx, "readout_loss": readout_loss, "h_norm": torch.norm(h_settled).item()}
        )

        return logits, diagnostics
