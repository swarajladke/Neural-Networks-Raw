"""
Raw AGNIS — src/agnis/neurogenesis/maturity.py

Maturity tracking for newly born units.

New units start with maturity=0 (no influence) and earn influence
only by demonstrably reducing prediction error.

See NOTES/neurogenesis_design.md for full design.
"""

import torch
from typing import Dict, List


class MaturityTracker:
    """
    Tracks maturity for all units (both original and newly born).

    Original units are initialized with maturity=1.0 (fully mature).
    New units are initialized with maturity=0.0.

    Parameters
    ----------
    n_initial_units : int
        Number of units at model initialization (all start mature).
    eta_maturity : float
        Learning rate for maturity updates.
    """

    def __init__(self, n_initial_units: int, eta_maturity: float = 0.01):
        self.eta_maturity = eta_maturity
        # maturity[j] ∈ [0, 1]
        self.maturity = torch.ones(n_initial_units)

    def add_unit(self):
        """Add a new unit with maturity=0."""
        new_mat = torch.zeros(1)
        self.maturity = torch.cat([self.maturity, new_mat])

    def remove_unit(self, unit_idx: int):
        """Remove a unit from maturity tracking."""
        keep = [i for i in range(len(self.maturity)) if i != unit_idx]
        self.maturity = self.maturity[keep]

    def update(self, unit_idx: int, error_before: float, error_after: float):
        """
        Update maturity of unit unit_idx based on error reduction.

        Parameters
        ----------
        unit_idx : int
            Index of the unit to update.
        error_before : float
            Prediction error before this unit contributed.
        error_after : float
            Prediction error after this unit contributed.
        """
        improvement = max(0.0, error_before - error_after)
        self.maturity[unit_idx] = min(
            1.0,
            self.maturity[unit_idx].item() + self.eta_maturity * improvement
        )

    def get_effective_activations(self, a: torch.Tensor) -> torch.Tensor:
        """
        Apply maturity gate to activations.

        z_eff_j = maturity_j * a_j

        Parameters
        ----------
        a : torch.Tensor of shape (n_units,)

        Returns
        -------
        torch.Tensor of shape (n_units,)
        """
        n = min(len(self.maturity), a.shape[0])
        gated = a.clone()
        gated[:n] = self.maturity[:n] * a[:n]
        return gated

    def low_maturity_units(self, threshold: float = 0.1) -> List[int]:
        """Return indices of units with maturity below threshold."""
        return [i for i, m in enumerate(self.maturity.tolist()) if m < threshold]

    def stats(self) -> Dict:
        return {
            "n_units": len(self.maturity),
            "mean_maturity": self.maturity.mean().item(),
            "min_maturity": self.maturity.min().item(),
            "max_maturity": self.maturity.max().item(),
            "n_low_maturity": len(self.low_maturity_units()),
        }
