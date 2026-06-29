"""
Raw AGNIS — src/agnis/neurogenesis/growth_controller.py

GrowthController: Computes the growth score G_l(t) and triggers neurogenesis.

PHASE 3 IMPLEMENTATION — Stub for Phase 0 bootstrap.

Growth score equation:
G_l(t) = alpha * EMA(error_l) + beta * novelty_l + gamma * uncertainty_l
        + delta * interference_l - kappa * coverage_l - lambda * cost_l

See: NOTES/neurogenesis_design.md for full specification.
"""

import torch
from typing import Optional


class GrowthController:
    """
    Computes layer-level growth scores and determines when to trigger neurogenesis.

    Parameters
    ----------
    alpha, beta, gamma, delta : float
        Positive weights for error, novelty, uncertainty, interference.
    kappa, lambda_cost : float
        Negative weights for coverage and capacity cost.
    threshold : float
        Growth score threshold to trigger neurogenesis.
    consecutive_n : int
        Number of consecutive above-threshold observations before triggering.
    ema_window : float
        EMA smoothing factor for growth score.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.4,
        kappa: float = 0.3,
        lambda_cost: float = 0.2,
        threshold: float = 0.6,
        consecutive_n: int = 10,
        ema_alpha: float = 0.05,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.lambda_cost = lambda_cost
        self.threshold = threshold
        self.consecutive_n = consecutive_n
        self.ema_alpha = ema_alpha

        # State
        self._ema_error: float = 0.0
        self._ema_G: float = 0.0
        self._consecutive_count: int = 0
        self._total_births: int = 0

    def update(
        self,
        error: float,
        novelty: float,
        uncertainty: float,
        interference: float,
        coverage: float,
        cost: float,
    ) -> bool:
        """
        Update growth score and check if neurogenesis should trigger.

        Returns
        -------
        bool
            True if neurogenesis should trigger now.
        """
        # Compute instantaneous growth score
        G = (
            self.alpha * error
            + self.beta * novelty
            + self.gamma * uncertainty
            + self.delta * interference
            - self.kappa * coverage
            - self.lambda_cost * cost
        )

        # EMA smoothing
        self._ema_G = (1 - self.ema_alpha) * self._ema_G + self.ema_alpha * G

        if self._ema_G > self.threshold:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        if self._consecutive_count >= self.consecutive_n:
            self._consecutive_count = 0
            self._total_births += 1
            return True

        return False

    @property
    def growth_score(self) -> float:
        return self._ema_G

    @property
    def total_births(self) -> int:
        return self._total_births
