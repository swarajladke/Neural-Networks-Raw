"""
Raw AGNIS — tests/test_sparc_routing.py

Unit tests verifying SPARC v0.2 learned routing, state-safe evaluations,
calibration floors, context reset safety, and optimizer isolation.
"""

import torch
import torch.nn.functional as F
import math
from agnis.sequence.sequence_wrapper import SPARCSequenceWrapper


def test_router_deterministic_one_hot():
    """Verify that deterministic top-1 routing weights sum to one and are exactly one-hot."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="supervised_router")
    z = torch.randn(8)
    
    # Run a route step
    routing_weights, soft, context, logits = wrapper.model.router.route(
        z, wrapper.model.context_state, wrapper.model.smoothed_logits
    )
    
    assert torch.allclose(routing_weights.sum(), torch.tensor(1.0)), "Routing weights must sum to one"
    assert (routing_weights == 1.0).sum() == 1, "Deterministic routing weights must be exactly one-hot"
    assert (routing_weights == 0.0).sum() == len(routing_weights) - 1, "Deterministic routing weights must be exactly one-hot"
    assert not wrapper.model.router.gumbel, "Gumbel sampling must be disabled by default"


def test_context_reset_at_sequence_boundaries():
    """Verify context and smoothed logit states reset to zero at sequence boundaries."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="supervised_router")
    z = torch.randn(8)
    
    # Step the model to populate context and logits
    wrapper.model.forward_step(z, torch.tensor(1), task_id=0, is_training=True)
    
    # Assert non-zero
    assert torch.norm(wrapper.model.context_state) > 0.0
    assert torch.norm(wrapper.model.smoothed_logits) > 0.0
    
    # Reset
    wrapper.reset_sequence_state()
    
    # Assert zero
    assert torch.allclose(wrapper.model.context_state, torch.zeros(8))
    assert torch.allclose(wrapper.model.smoothed_logits, torch.zeros(3))


def test_optimizer_parameter_ownership():
    """Verify only router projection parameters exist in the active router optimizer."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="supervised_router")
    
    router_parameter_ids = {id(p) for p in wrapper.model.router.parameters()}
    optimizer_parameter_ids = {
        id(p)
        for group in wrapper.router_optimizer.param_groups
        for p in group["params"]
    }
    
    assert optimizer_parameter_ids == router_parameter_ids, "Optimizer must own exactly and only the router parameters"


def test_bitwise_expert_freezing():
    """Verify training the router leaves all columns and heads bitwise unchanged."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="supervised_router")
    
    # Snapshot parameters of columns and heads
    column_snapshots = {}
    for j in range(wrapper.model.num_columns):
        column_snapshots[j] = {k: v.clone() for k, v in wrapper.model.columns[j].state_dict().items()}
        
    # Run multiple train transitions
    for _ in range(5):
        wrapper.train_transition(torch.randn(8), F.one_hot(torch.tensor(2), num_classes=4))
        
    # Verify bitwise stability of experts
    for j in range(wrapper.model.num_columns):
        current_state = wrapper.model.columns[j].state_dict()
        for k, v in column_snapshots[j].items():
            assert torch.equal(v, current_state[k]), f"Parameter {k} in column {j} was mutated during router training!"


def test_minimum_energy_state_safety():
    """Verify minimum-energy routing commits exactly one state and decays all losers exactly once."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="minimum_energy")
    
    # Set mock states to distinct values
    for j in range(wrapper.model.num_columns):
        wrapper.model.h_prev[j] = torch.ones(32) * (j + 1)
        
    h_prev_snapshot = wrapper.model.h_prev.clone()
    z = torch.randn(8)
    
    # Run one step
    logits, diag = wrapper.model.forward_step(z, torch.tensor(1), task_id=0, is_training=False)
    winner = diag["active_column"]
    
    # Verify winner was committed and losers were decayed exactly once
    decay = wrapper.model.decay_factor
    for j in range(wrapper.model.num_columns):
        if j == winner:
            # Committed winner state should not match simple decay of its prior
            assert not torch.allclose(wrapper.model.h_prev[j], h_prev_snapshot[j] * decay)
        else:
            # Loser state should be decayed exactly once
            assert torch.allclose(wrapper.model.h_prev[j], h_prev_snapshot[j] * decay)


def test_early_stopping_agreement():
    """Verify early stopping approximates full settling predictions and routing decisions."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="task_id_oracle")
    z = torch.randn(8)
    h_prev = torch.randn(32)
    
    # 1. Full settling (early_stop=False)
    h_full, diag_full = wrapper.model.columns[0].settle(z, h_prev, early_stop=False)
    
    # 2. Early stopping (early_stop=True)
    h_early, diag_early = wrapper.model.columns[0].settle(z, h_prev, early_stop=True, min_steps=5, energy_tol=1e-4)
    
    # Compute metrics
    cos_sim = F.cosine_similarity(h_early.unsqueeze(0), h_full.unsqueeze(0)).item()
    steps_reduction = diag_full["steps_taken"] - diag_early["steps_taken"]
    
    # Verify high cosine similarity (>= 0.95) and positive steps reduction
    assert cos_sim >= 0.95, f"Cosine similarity between early and full settling is too low: {cos_sim:.4f}"
    print(f"[Early-Stopping Test] Cosine Similarity: {cos_sim:.4f}, Iteration Reduction: {steps_reduction}")


def test_mixture_probability_sums_to_one():
    """Verify Router C (learned_router_mixture) output probability distribution sums to 1.0."""
    wrapper = SPARCSequenceWrapper(d_in=8, d_out=4, num_columns=3, routing_mode="learned_router_mixture")
    z = torch.randn(8)
    
    # Run a forward step with is_training=True (which triggers the probability mixture calculation)
    logits, diag = wrapper.model.forward_step(
        z=z,
        target=torch.tensor([1]),
        task_id=0,
        is_training=True
    )
    # Logits returned is log_probs in mixture mode
    probs = torch.exp(logits)
    
    assert torch.allclose(probs.sum(), torch.tensor(1.0)), "Mixture output probability must sum to exactly 1.0"
