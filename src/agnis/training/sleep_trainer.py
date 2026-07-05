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


from agnis.memory.replay_buffer import ReplayBuffer, LatentReplayBuffer


class SleepTrainer:
    """
    Offline replay-based consolidation trainer. Supports both flat cells
    and stacked hierarchies.

    Parameters
    ----------
    model : PredictiveCell or PredictiveHierarchy
        The model to consolidate.
    fast_memory : FastMemory, optional
        Source of episodic memories for replay.
    replay_buffer : ReplayBuffer, optional
        Curated replay buffer (pre-populated from FastMemory).
    sleep_lr_scale : float
        Scale factor for learning rate during sleep (< 1.0 = slower).
    importance_protect_threshold : float
        Weights with importance above this are not updated during sleep.
    latent_replay_buffer : LatentReplayBuffer, optional
        Buffer storing latent snapshots for hierarchical models.
    """

    def __init__(
        self,
        model,
        fast_memory: Optional[FastMemory] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        sleep_lr_scale: float = 0.3,
        importance_protect_threshold: float = 0.5,
        latent_replay_buffer: Optional[LatentReplayBuffer] = None,
    ):
        self.model = model
        self.fast_memory = fast_memory
        self.replay_buffer = replay_buffer
        self.sleep_lr_scale = sleep_lr_scale
        self.importance_protect_threshold = importance_protect_threshold
        self.latent_replay_buffer = latent_replay_buffer

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

        is_hierarchy = hasattr(self.model, "n_layers")

        for _ in range(n_steps):
            # 1. Input-level Replay
            if self.replay_buffer is not None and self.replay_buffer.size > 0:
                batch = self.replay_buffer.sample(n_replay)
                replay_pairs = [(v, v) for _, v, _ in batch]
            elif self.fast_memory is not None and self.fast_memory.size > 0:
                sampled_entries = self.fast_memory.sample_by_importance(n_replay)
                replay_pairs = [(e.value, e.value) for e in sampled_entries]
            else:
                replay_pairs = []

            if is_hierarchy:
                # Temporarily scale learning rates
                orig_eta_d = self.model.eta_d
                orig_eta_e = self.model.eta_e
                orig_eta_r = self.model.eta_r
                self.model.eta_d *= self.sleep_lr_scale
                self.model.eta_e *= self.sleep_lr_scale
                self.model.eta_r *= self.sleep_lr_scale

                for s, _ in replay_pairs:
                    z_settled = self.model.forward(s, t=self.model._step)
                    errors.append(self.model.total_prediction_error)
                    self.model.update_all_weights(s, t=self.model._step)

                # Restore original learning rates
                self.model.eta_d = orig_eta_d
                self.model.eta_e = orig_eta_e
                self.model.eta_r = orig_eta_r
            else:
                for s, _ in replay_pairs:
                    a = self.model.forward(s)
                    e = self.model._last_error
                    if e is None:
                        continue

                    error_val = (e ** 2).mean().item()
                    errors.append(error_val)

                    eta_sleep_D = self.model.eta_D * self.sleep_lr_scale
                    eta_sleep_E = self.model.eta_E * self.sleep_lr_scale

                    delta_D = hebbian_generative_update(e, a, eta_sleep_D)
                    delta_E = hebbian_recognition_update(
                        self.model.z, self.model.E, s, eta_sleep_E
                    )

                    protect_mask_D = self.model.importance_D > self.importance_protect_threshold
                    protect_mask_E = self.model.importance_E > self.importance_protect_threshold

                    delta_D[protect_mask_D] = 0.0
                    delta_E[protect_mask_E] = 0.0

                    self.model.D = self.model.D + delta_D
                    self.model.E = self.model.E + delta_E

            # 2. Inter-layer Latent Replay (hierarchy models only)
            if is_hierarchy and self.latent_replay_buffer is not None and self.latent_replay_buffer.size > 0:
                latent_batch = self.latent_replay_buffer.sample(n_replay)
                for latent_states in latent_batch:
                    for l in range(self.model.n_layers - 1):
                        z_l = latent_states[l]
                        z_above = latent_states[l + 1]

                        a_above = self.model._maturity(l + 1) * self.model.activation(z_above)
                        e_latent = z_l - self.model._D_inter(l + 1) @ a_above

                        decay_factor = self.model.lr_decay ** (l + 1)
                        eta_sleep = self.model.eta_d * decay_factor * self.sleep_lr_scale

                        delta_D_latent = eta_sleep * torch.outer(e_latent, a_above)

                        # Importance protection guard
                        protect_mask = self.model._importance_D(l + 1) > self.importance_protect_threshold
                        delta_D_latent[protect_mask] = 0.0

                        new_D = self.model._D_inter(l + 1) + delta_D_latent
                        self.model._set_D_inter(l + 1, new_D)

        return errors

