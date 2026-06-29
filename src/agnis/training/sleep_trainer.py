"""
Raw AGNIS — src/agnis/training/sleep_trainer.py

SleepTrainer: Offline replay-based memory consolidation.

The sleep phase:
- Runs AFTER a task training session is complete (offline, not interleaved)
- Samples important memories from FastMemory / ReplayBuffer
- Reconstructs inputs from memory prototypes
- Applies slow weight updates with reduced plasticity
- Protects high-importance weights

This separation of online learning (OnlineTrainer) from offline consolidation
(SleepTrainer) is critical: mixing replay into online updates causes interference.
"""

import torch
from typing import Optional, List
from agnis.core.predictive_cell import PredictiveCell
from agnis.memory.fast_memory import FastMemory
from agnis.memory.replay_buffer import ReplayBuffer
from agnis.core.hebbian_rules import hebbian_generative_update, hebbian_recognition_update


class SleepTrainer:
    """
    Offline replay-based consolidation trainer.

    Parameters
    ----------
    model : PredictiveCell
        The model to consolidate.
    fast_memory : FastMemory, optional
        Source of episodic memories for replay.
    replay_buffer : ReplayBuffer, optional
        Curated replay buffer (pre-populated from FastMemory).
    sleep_lr_scale : float
        Scale factor for learning rate during sleep (< 1.0 = slower).
    importance_protect_threshold : float
        Weights with importance above this are not updated during sleep.
    """

    def __init__(
        self,
        model: PredictiveCell,
        fast_memory: Optional[FastMemory] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        sleep_lr_scale: float = 0.3,
        importance_protect_threshold: float = 0.5,
    ):
        self.model = model
        self.fast_memory = fast_memory
        self.replay_buffer = replay_buffer
        self.sleep_lr_scale = sleep_lr_scale
        self.importance_protect_threshold = importance_protect_threshold

    def sleep(
        self,
        n_replay: int = 32,
        n_steps: int = 1,
    ) -> List[float]:
        """
        Run one sleep consolidation phase.

        Parameters
        ----------
        n_replay : int
            Number of memories to replay per sleep step.
        n_steps : int
            Number of sleep consolidation rounds.

        Returns
        -------
        list of float
            Per-step replay error.
        """
        errors = []

        for _ in range(n_steps):
            # Sample replay batch
            if self.replay_buffer is not None and self.replay_buffer.size > 0:
                batch = self.replay_buffer.sample(n_replay)
                replay_pairs = [(v, v) for _, v, _ in batch]  # reconstruct input from value
            elif self.fast_memory is not None and self.fast_memory.size > 0:
                sampled_entries = self.fast_memory.sample_by_importance(n_replay)
                replay_pairs = [(e.value, e.value) for e in sampled_entries]
            else:
                break  # nothing to replay

            for s, _ in replay_pairs:
                # Forward settling with replay input
                a = self.model.forward(s)
                e = self.model._last_error
                if e is None:
                    continue

                error_val = (e ** 2).mean().item()
                errors.append(error_val)

                # Apply scaled Hebbian update during sleep
                eta_sleep_D = self.model.eta_D * self.sleep_lr_scale
                eta_sleep_E = self.model.eta_E * self.sleep_lr_scale

                delta_D = hebbian_generative_update(e, a, eta_sleep_D)
                delta_E = hebbian_recognition_update(
                    self.model.z, self.model.E, s, eta_sleep_E
                )

                # Protect high-importance weights: zero out their updates
                protect_mask_D = self.model.importance_D > self.importance_protect_threshold
                protect_mask_E = self.model.importance_E > self.importance_protect_threshold

                delta_D[protect_mask_D] = 0.0
                delta_E[protect_mask_E] = 0.0

                self.model.D = self.model.D + delta_D
                self.model.E = self.model.E + delta_E

        return errors
