"""
Raw AGNIS — src/agnis/sparc/learned_router.py

Differentiable contextual learned router for SPARC v0.2.
Implements Option A deterministic ST top-1 and Gumbel-Softmax training objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class DifferentiableTopKRouter(nn.Module):
    """
    SPARC v0.2 Differentiable Top-1 Learned Router.
    Routes context history states to column indices without task-IDs at inference.
    """

    def __init__(
        self,
        d_input: int,
        num_columns: int,
        temperature: float = 1.0,
        route_inertia: float = 0.9,
        context_decay: float = 0.9,
        gumbel: bool = False,
    ):
        super().__init__()
        self.d_input = d_input
        self.num_columns = num_columns
        self.temperature = temperature
        self.route_inertia = route_inertia
        self.context_decay = context_decay
        self.gumbel = gumbel

        # Learnable projection mapping causal context c_t to column logits
        self.proj = nn.Linear(d_input, num_columns)

    def contextual_logits(self, context: torch.Tensor) -> torch.Tensor:
        """Projects context state c_t onto logits."""
        return self.proj(context)

    def apply_inertia(self, raw_logits: torch.Tensor, previous_smoothed_logits: torch.Tensor) -> torch.Tensor:
        """Applies routing inertia smoothing over logits."""
        return self.route_inertia * previous_smoothed_logits + (1.0 - self.route_inertia) * raw_logits

    def route(
        self, z: torch.Tensor, context_state: torch.Tensor, smoothed_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infers routing decision for input token z:
        1. Updates exponentially decayed context history.
        2. Computes raw and smoothed logits.
        3. Generates straight-through top-1 routing weights.
        Returns routing_weights, soft_probs, new_context, new_smoothed_logits.
        """
        # 1. Update causal context summary
        new_context = self.context_decay * context_state + (1.0 - self.context_decay) * z

        # 2. Project context to raw logits and apply inertia
        raw_logits = self.contextual_logits(new_context)
        new_smoothed_logits = self.apply_inertia(raw_logits, smoothed_logits)

        # 3. Compute routing selection
        if self.training and self.gumbel:
            # Gumbel-Softmax stochastic sampling path (ablation)
            soft = F.gumbel_softmax(new_smoothed_logits, tau=self.temperature, hard=True, dim=-1)
            routing_weights = soft
        else:
            # Deterministic ST Top-1 selection path (default)
            soft = torch.softmax(new_smoothed_logits / self.temperature, dim=-1)
            selected = soft.argmax(dim=-1)
            hard = F.one_hot(selected, num_classes=self.num_columns).to(soft.dtype)
            routing_weights = hard.detach() + soft - soft.detach()

        return routing_weights, soft, new_context.detach(), new_smoothed_logits.detach()
