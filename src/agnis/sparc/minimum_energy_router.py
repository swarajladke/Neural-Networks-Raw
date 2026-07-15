"""
Raw AGNIS — src/agnis/sparc/minimum_energy_router.py

Minimum Energy Router and Calibration for SPARC v0.2.
Routes context inputs by evaluating proximal settling on all candidate columns.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple


class MinimumEnergyRouter:
    """
    SPARC v0.2 Minimum Energy Router.
    Routes context inputs to columns by comparing robustly calibrated proximal settling energies.
    Does not mutate column recurrent states, usage statistics, or growth states during evaluation.
    """

    def __init__(self, columns: nn.ModuleList, num_columns: int, minimum_energy_scale: float = 1e-4):
        self.columns = columns
        self.num_columns = num_columns
        self.minimum_energy_scale = minimum_energy_scale

        # Calibration buffers (frozen during router training)
        self.calibration_metadata = {}
        for j in range(num_columns):
            self.calibration_metadata[j] = {
                "median": 0.0,
                "mad": 0.0,
                "scale": 1.0,  # max(1.4826 * mad, s_min)
                "n_samples": 0,
            }

    def set_calibration(self, column_idx: int, median: float, mad: float, n_samples: int):
        """Sets robust calibration parameters for a specific column."""
        scale = max(1.4826 * mad, self.minimum_energy_scale)
        self.calibration_metadata[column_idx] = {
            "median": median,
            "mad": mad,
            "scale": scale,
            "n_samples": n_samples,
        }

    def calibrate_energy(self, column_idx: int, raw_energy: float) -> float:
        """Calibrates raw settling energy using robust median/MAD metadata."""
        meta = self.calibration_metadata[column_idx]
        return (raw_energy - meta["median"]) / meta["scale"]

    def route_step(
        self, z: torch.Tensor, permanent_states: List[torch.Tensor], decay_factor: float
    ) -> Tuple[int, List[torch.Tensor], List[float], List[torch.Tensor]]:
        """
        Evaluate settling on all columns and select the winner:
        1. Computes candidate priors under temporal decay for all columns.
        2. Settles every column without committing states or updating stats.
        3. Normalizes and calibrates energies.
        Returns selected_column_index, settled_candidates, calibrated_energies, candidate_priors.
        """
        candidate_priors = []
        for j in range(self.num_columns):
            # Compute effective decayed prior for all candidate columns
            candidate_priors.append(decay_factor * permanent_states[j].clone())

        settled_candidates = []
        raw_energies = []

        with torch.no_grad():
            for j, column in enumerate(self.columns):
                # Settle without committing state, mutating parameters, or updating usage statistics
                settled_h, diagnostics = column.settle(
                    z=z,
                    h_previous=candidate_priors[j],
                    # Additional flags inside PredictiveColumn should prevent internal stat mutation
                )
                settled_candidates.append(settled_h)
                raw_energies.append(diagnostics["final_energy"])

        # Calibrate energies using robust scale
        calibrated_energies = [
            self.calibrate_energy(j, raw_energies[j]) for j in range(self.num_columns)
        ]

        # Select the calibrated minimum energy column
        winner = int(torch.tensor(calibrated_energies).argmin().item())

        return winner, settled_candidates, calibrated_energies, candidate_priors
