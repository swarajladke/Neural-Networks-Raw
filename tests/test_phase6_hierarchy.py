"""
Raw AGNIS — tests/test_phase6_hierarchy.py

Phase 6 Deep Stacked Hierarchy tests.
"""

import pytest
import torch
import math

from agnis.core.predictive_hierarchy import PredictiveHierarchy
from agnis.memory.replay_buffer import LatentReplayBuffer
from agnis.neurogenesis.routing import NoveltyAttributor
from agnis.sequence.sequence_wrapper import DeepSeqAgnisModel
from agnis.utils.config import ExperimentConfig


def test_predictive_hierarchy_init():
    """Verify initialization and matrix shapes in stacked PredictiveHierarchy."""
    d_input = 24
    layer_dims = [32, 16, 8]
    k_sparse = [4, 2, 1]
    commit_strides = [1, 2, 4]

    hierarchy = PredictiveHierarchy(
        d_input=d_input,
        layer_dims=layer_dims,
        k_sparse_per_layer=k_sparse,
        commit_strides=commit_strides,
        use_precision_gating=True,
    )

    assert hierarchy.n_layers == 3
    assert list(hierarchy.layer_dims) == layer_dims

    # Verify shapes
    assert hierarchy._E(0).shape == (32, 24)
    assert hierarchy._E(1).shape == (16, 32)
    assert hierarchy._E(2).shape == (8, 16)

    assert hierarchy._D_inter(0).shape == (24, 32)
    assert hierarchy._D_inter(1).shape == (32, 16)
    assert hierarchy._D_inter(2).shape == (16, 8)

    assert hierarchy._R(0).shape == (32, 32)
    assert hierarchy._R(1).shape == (16, 16)
    assert hierarchy._R(2).shape == (8, 8)


def test_predictive_hierarchy_forward():
    """Verify settling and multi-rate commitment in PredictiveHierarchy."""
    d_input = 20
    layer_dims = [16, 8]
    k_sparse = [2, 1]
    commit_strides = [1, 3]

    hierarchy = PredictiveHierarchy(
        d_input=d_input,
        layer_dims=layer_dims,
        k_sparse_per_layer=k_sparse,
        commit_strides=commit_strides,
    )

    s = torch.randn(d_input)

    # Step t=0 (both commit)
    z_states = hierarchy.forward(s, t=0)
    assert len(z_states) == 2
    assert z_states[0].shape == (16,)
    assert z_states[1].shape == (8,)

    # Verify sparsity
    assert (z_states[0] != 0).sum().item() <= 2
    assert (z_states[1] != 0).sum().item() <= 1

    # Step t=1 (only layer 0 commits, layer 1 holds z_prev)
    z_prev_1_before = hierarchy._z_prev(1).clone()
    z_states_t1 = hierarchy.forward(s, t=1)
    assert torch.equal(hierarchy._z(1), z_states[1])  # Layer 1 held its state


def test_predictive_hierarchy_updates():
    """Verify gated weight updates in PredictiveHierarchy."""
    d_input = 12
    layer_dims = [16, 8]
    k_sparse = [2, 1]
    commit_strides = [1, 2]

    hierarchy = PredictiveHierarchy(
        d_input=d_input,
        layer_dims=layer_dims,
        k_sparse_per_layer=k_sparse,
        commit_strides=commit_strides,
        use_precision_gating=True,
    )

    # Initial weights
    D0_orig = hierarchy._D_inter(0).clone()
    E1_orig = hierarchy._E(1).clone()

    s = torch.randn(d_input)
    hierarchy.forward(s, t=0)
    hierarchy.update_all_weights(s, t=0)

    # Weight check after update (D0 should have changed because t=0 commits both)
    assert not torch.equal(hierarchy._D_inter(0), D0_orig)
    assert not torch.equal(hierarchy._E(1), E1_orig)


def test_predictive_hierarchy_neurogenesis():
    """Verify growth expands all state vectors and routing logic selects correct target layer."""
    d_input = 10
    layer_dims = [16, 8]
    k_sparse = [2, 1]
    commit_strides = [1, 2]

    hierarchy = PredictiveHierarchy(
        d_input=d_input,
        layer_dims=layer_dims,
        k_sparse_per_layer=k_sparse,
        commit_strides=commit_strides,
    )

    # Pre-growth checks
    assert hierarchy.layer_dims == [16, 8]
    assert hierarchy._E(0).shape == (16, 10)
    assert hierarchy._D_inter(1).shape == (16, 8)
    assert hierarchy._E(1).shape == (8, 16)

    # Grow a unit at layer 0
    hierarchy.grow_units(l=0, current_input=torch.randn(10), residual_error=torch.randn(10))

    # Post-growth checks (layer 0 dimension expanded to 17)
    assert hierarchy.layer_dims == [17, 8]
    assert hierarchy._E(0).shape == (17, 10)
    # The layer above (l=1) should have scaled its incoming row/column matrices
    assert hierarchy._D_inter(1).shape == (17, 8)
    assert hierarchy._E(1).shape == (8, 17)
    assert hierarchy._error_accum(1).shape == (17,)

    # Verify routing logic
    routing = NoveltyAttributor(n_layers=2, max_units_per_layer=[20, 10])
    routing.update([0.8, 0.4])  # layer 0 error high, layer 1 error low
    # rho^1 = 0.4 / 0.8 = 0.5 (equal or above 0.5 triggers escalation to parent)
    target = routing.route_growth([True, True], [17, 8])
    assert target == 1  # Escalates to layer 1


def test_latent_replay_buffer_balanced():
    """Verify domain-balanced reservoir sampling in LatentReplayBuffer."""
    buffer = LatentReplayBuffer(max_per_domain=3)

    # Add 10 snapshots for domain 0
    for i in range(10):
        buffer.add(task_id=0, latent_states=[torch.tensor([i])])

    # Verify reservoir sampling capped size at max_per_domain = 3
    assert len(buffer.buffers[0]) == 3

    # Add 2 snapshots for domain 1
    buffer.add(task_id=1, latent_states=[torch.tensor([10.0])])
    buffer.add(task_id=1, latent_states=[torch.tensor([20.0])])

    assert len(buffer.buffers[1]) == 2
    assert buffer.size == 5

    # Sample batch
    batch = buffer.sample(batch_size=4)
    assert len(batch) == 4


def test_deep_seq_agnis_model_training():
    """Verify DeepSeqAgnisModel wrapper train step, prediction, and sleep consolidation."""
    config = ExperimentConfig()
    config.hierarchy.enabled = True
    config.hierarchy.layer_dims = [16, 8]
    config.hierarchy.k_sparse_per_layer = [2, 1]
    config.hierarchy.commit_strides = [1, 2]
    config.neurogenesis.enabled = True

    model = DeepSeqAgnisModel(d_in=5, d_out=5, config=config)

    x = torch.randn(5)
    y = torch.randn(5)

    # Train transition
    out = model.train_transition(x, y)
    assert "error" in out

    # Predict
    pred = model.predict_no_state_update(x)
    assert pred.shape == (5,)
    assert abs(pred.sum().item() - 1.0) < 1e-5  # normalized softmax

    # Sleep
    sleep_out = model.sleep()
    # If sleep consolidates, should output start/end errors
    assert isinstance(sleep_out, dict)


def test_dim_below_sync_after_growth():
    """Regression: growing layer 1 then resetting state must not crash.

    Before the fix, _dim_below was stale after grow_units, so
    reset_state() would allocate error_accum(2) with the wrong size,
    causing a RuntimeError on the next forward pass.
    """
    d_input = 24
    layer_dims = [32, 16, 8]
    k_sparse = [4, 2, 1]
    commit_strides = [1, 2, 4]

    h = PredictiveHierarchy(
        d_input=d_input,
        layer_dims=list(layer_dims),
        k_sparse_per_layer=list(k_sparse),
        commit_strides=commit_strides,
        use_precision_gating=True,
    )

    x = torch.randn(d_input)
    h.forward(x, t=0)

    # Grow a unit at layer 1 (dim goes 16 -> 17)
    residual = torch.randn(layer_dims[0])  # dim_below for layer 1
    h.grow_units(1, residual, residual)
    assert h.layer_dims[1] == layer_dims[1] + 1
    assert h._dim_below[2] == layer_dims[1] + 1

    # The critical sequence: reset then forward again
    h.reset_state()
    h.forward(x, t=1)  # would crash without _dim_below sync
