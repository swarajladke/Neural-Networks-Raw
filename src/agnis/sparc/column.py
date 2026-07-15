"""
Raw AGNIS — src/agnis/sparc/column.py

SPARC v0.1 Predictive Column module.
Implements proximal gradient settling, line search, and local updates.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple


class PredictiveColumn(nn.Module):
    """
    SPARC v0.1 Predictive Column.
    Wraps generative decoder D, transition matrix R, and column-local task readout head Q.
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_output = d_output
        self.alpha = alpha
        self.beta = beta
        self.eta_D = eta_D
        self.eta_R = eta_R
        self.n_settle = n_settle
        self.step_c = step_c

        # 1. Generative Decoder (D): maps h -> z prediction
        scale_d = 1.0 / math.sqrt(d_latent)
        self.D = nn.Parameter(torch.randn(d_input, d_latent) * scale_d)

        # 2. Recurrent Transition (R): maps h_prev -> h prior prediction
        self.R = nn.Parameter(torch.randn(d_latent, d_latent) * 0.01)

        # 3. Column-Local Readout Head (Q)
        self.Q = nn.Linear(d_latent, d_output)

        # Isolated optimizer for readout parameter Q to guarantee zero drift elsewhere
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=eta_Q)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_recurrent_prior(self, h_previous: torch.Tensor) -> torch.Tensor:
        """Compute the temporal prior prior_h = R * stopgrad(h_previous)."""
        with torch.no_grad():
            h_prev_local = h_previous.detach()
            return torch.mv(self.R, h_prev_local)

    def energy(self, z: torch.Tensor, h: torch.Tensor, h_prior: torch.Tensor) -> torch.Tensor:
        """Compute total column energy E_j = E_smooth + alpha * ||h||_1."""
        z_hat = torch.mv(self.D, h)
        recon_err = 0.5 * torch.sum((z - z_hat) ** 2)
        sparsity = self.alpha * torch.sum(torch.abs(h))
        temp_err = 0.5 * self.beta * torch.sum((h - h_prior) ** 2)
        return recon_err + sparsity + temp_err

    def soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Elementwise soft thresholding operator."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    def compute_smooth_gradient(self, z: torch.Tensor, h: torch.Tensor, h_prior: torch.Tensor) -> torch.Tensor:
        """Compute gradient of smooth energy terms with respect to h."""
        z_hat = torch.mv(self.D, h)
        diff = z_hat - z
        grad_recon = torch.mv(self.D.T, diff)
        grad_temp = self.beta * (h - h_prior)
        return grad_recon + grad_temp

    def settle(
        self,
        z: torch.Tensor,
        h_previous: torch.Tensor,
        early_stop: bool = False,
        min_steps: int = 5,
        energy_tol: float = 1e-4,
        state_tol: float = 1e-4,
        consecutive_steps: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run proximal gradient settling with backtracking line search to infer latent h.
        Returns settled latent state and diagnostic statistics.
        """
        h_prior = self.get_recurrent_prior(h_previous)
        h = h_prior.detach().clone()

        # Lipschitz spectral-norm approximation (L2 matrix norm)
        d_norm_sq = (torch.linalg.matrix_norm(self.D, ord=2).item()) ** 2
        initial_step = self.step_c / (d_norm_sq + self.beta + 1e-8)
        minimum_step = initial_step * 1e-4
        backtrack_factor = 0.5

        rejected_steps = 0
        failures = 0
        backtrack_count = 0
        consecutive_converged = 0
        epsilon = 1e-8
        steps_taken = 0

        for s in range(self.n_settle):
            steps_taken = s + 1
            old_h = h.clone()
            old_energy = self.energy(z, old_h, h_prior).item()

            smooth_grad = self.compute_smooth_gradient(z, old_h, h_prior)

            step_size = initial_step
            step_success = False

            while step_size >= minimum_step:
                u = old_h - step_size * smooth_grad
                candidate = self.soft_threshold(u, step_size * self.alpha)
                candidate_energy = self.energy(z, candidate, h_prior).item()

                if math.isfinite(candidate_energy) and (candidate_energy <= old_energy + 1e-7):
                    h = candidate
                    step_success = True
                    break
                else:
                    rejected_steps += 1
                    step_size *= backtrack_factor

            if not step_success:
                failures += 1
                h = old_h  # Rollback
            else:
                if step_size < initial_step:
                    backtrack_count += 1

            if early_stop:
                new_energy = self.energy(z, h, h_prior).item()
                relative_energy_change = abs(new_energy - old_energy) / (abs(old_energy) + epsilon)
                relative_state_change = (torch.linalg.vector_norm(h - old_h) / (torch.linalg.vector_norm(old_h) + epsilon)).item()

                converged = (
                    s + 1 >= min_steps
                    and relative_energy_change < energy_tol
                    and relative_state_change < state_tol
                )
                consecutive_converged = consecutive_converged + 1 if converged else 0
                if consecutive_converged >= consecutive_steps:
                    break

        diagnostics = {
            "rejected_steps": rejected_steps,
            "line_search_failures": failures,
            "backtrack_occurrences": backtrack_count,
            "final_energy": self.energy(z, h, h_prior).item(),
            "active_fraction": torch.mean((torch.abs(h) > 1e-5).float()).item(),
            "steps_taken": steps_taken,
        }

        return h, diagnostics

    def update_predictive_parameters(self, z: torch.Tensor, h: torch.Tensor, h_previous: torch.Tensor):
        """Update D and R via local error rules, treating latents as fixed."""
        h_det = h.detach()
        h_prev_det = h_previous.detach()
        h_prior = self.get_recurrent_prior(h_prev_det)

        # 1. Decoder delta rule
        z_hat = torch.mv(self.D, h_det)
        diff = z_hat - z
        grad_D = torch.outer(diff, h_det)
        self.D.data -= self.eta_D * grad_D

        # 2. Recurrence delta rule
        temp_diff = h_prior - h_det
        grad_R = self.beta * torch.outer(temp_diff, h_prev_det)
        self.R.data -= self.eta_R * grad_R

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """Compute output logits from latent state h."""
        return self.Q(h)

    def update_readout(self, h_settled: torch.Tensor, target: torch.Tensor) -> float:
        """Train readout head Q via supervised backprop on detached latent state."""
        h_det = h_settled.detach()
        logits = self.Q(h_det).unsqueeze(0)
        target_idx = target.view(-1)
        loss = self.loss_fn(logits, target_idx)

        self.Q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.Q_optimizer.step()

        return loss.item()
