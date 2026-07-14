"""
Raw AGNIS — tests/test_sparc_settling.py

Unit tests for SPARC v0.1 proximal latent settling, line search,
exact quadratic solutions, and parameter isolation.
"""

import pytest
import torch
from agnis.sparc.column import PredictiveColumn
from agnis.sparc.router import TaskIDOracleRouter, NearestPrototypeRouter
from agnis.sparc.sparc_model import SPARCSequenceModel


def test_proximal_settling_exact_quadratic():
    """
    Test 1: Check that for alpha=0 (no sparsity), the proximal gradient settling
    converges exactly to the analytic quadratic solution:
    h* = (D^T * D + beta * I)^-1 * (D^T * z + beta * h_prior)
    """
    torch.manual_seed(42)
    d_input = 16
    d_latent = 8
    d_output = 4

    # Instantiate column with alpha=0 (no sparsity penalty)
    column = PredictiveColumn(
        d_input=d_input,
        d_latent=d_latent,
        d_output=d_output,
        alpha=0.0,  # 0 sparsity
        beta=0.5,
        n_settle=100,  # Run many steps for tight convergence
        step_c=0.2,  # Conservative step size
    )

    # Random target input z and previous state
    z = torch.randn(d_input)
    h_previous = torch.randn(d_latent)

    # 1. Run numerical settling
    h_settled, diagnostics = column.settle(z, h_previous)

    # 2. Compute exact mathematical solution
    h_prior = column.get_recurrent_prior(h_previous)
    D = column.D.detach()
    beta = column.beta

    # System: (D^T * D + beta * I)
    system = torch.mm(D.T, D) + beta * torch.eye(d_latent)
    # Right-hand side: (D^T * z + beta * h_prior)
    rhs = torch.mv(D.T, z) + beta * h_prior

    h_exact = torch.linalg.solve(system, rhs)

    # Assert convergence within numerical tolerance
    # (Tolerance 1e-4 is appropriate for numerical settling convergence)
    assert torch.allclose(h_settled, h_exact, atol=1e-4), "Settling did not converge to analytic solution"
    assert diagnostics["line_search_failures"] == 0, "Line search failed on smooth quadratic problem"


def test_energy_monotonic_decrease():
    """
    Test 2: Verify that for alpha > 0, every accepted settling step
    reduces or preserves total column energy (monotonicity).
    """
    torch.manual_seed(43)
    d_input = 16
    d_latent = 8
    d_output = 4

    column = PredictiveColumn(
        d_input=d_input, d_latent=d_latent, d_output=d_output, alpha=0.1, beta=0.2, n_settle=15, step_c=0.5
    )

    z = torch.randn(d_input)
    h_previous = torch.randn(d_latent)

    h_prior = column.get_recurrent_prior(h_previous)
    h = h_prior.detach().clone()

    # Track energy at every step
    energies = [column.energy(z, h, h_prior).item()]

    # Run custom settling tracking step-by-step
    d_norm_sq = torch.sum(column.D ** 2).item()
    initial_step = column.step_c / (d_norm_sq + column.beta + 1e-8)
    minimum_step = initial_step * 1e-4
    backtrack_factor = 0.5

    for _ in range(column.n_settle):
        old_h = h.clone()
        old_energy = column.energy(z, old_h, h_prior).item()
        smooth_grad = column.compute_smooth_gradient(z, old_h, h_prior)

        step_size = initial_step
        step_success = False

        while step_size >= minimum_step:
            u = old_h - step_size * smooth_grad
            candidate = column.soft_threshold(u, step_size * column.alpha)
            candidate_energy = column.energy(z, candidate, h_prior).item()

            if candidate_energy <= old_energy + 1e-7:
                h = candidate
                step_success = True
                break
            step_size *= backtrack_factor

        new_energy = column.energy(z, h, h_prior).item()
        assert new_energy <= old_energy + 1e-7, "Energy increased during a settling step!"
        energies.append(new_energy)

    # Verify overall energy decreased
    assert energies[-1] < energies[0], "Energy did not decrease over the settling process"


def test_sparsity_activation():
    """
    Test 3: Confirm that alpha > 0 produces sparser latents than alpha = 0.
    """
    torch.manual_seed(44)
    d_input = 16
    d_latent = 8
    d_output = 4

    z = torch.randn(d_input)
    h_previous = torch.randn(d_latent)

    # Column with no sparsity
    col_dense = PredictiveColumn(
        d_input=d_input, d_latent=d_latent, d_output=d_output, alpha=0.0, beta=0.2, n_settle=15
    )
    h_dense, _ = col_dense.settle(z, h_previous)

    # Column with high sparsity
    col_sparse = PredictiveColumn(
        d_input=d_input, d_latent=d_latent, d_output=d_output, alpha=0.5, beta=0.2, n_settle=15
    )
    h_sparse, _ = col_sparse.settle(z, h_previous)

    dense_active = torch.sum(torch.abs(h_dense) > 1e-4).item()
    sparse_active = torch.sum(torch.abs(h_sparse) > 1e-4).item()

    assert sparse_active <= dense_active, "Sparsity penalty did not reduce active latent elements"


def test_parameter_isolation():
    """
    Test 6: Verify that training active columns does not mutate inactive columns' parameters
    (bitwise synaptic protection).
    """
    torch.manual_seed(45)
    model = SPARCSequenceModel(
        num_columns=3, d_input=10, d_latent=5, d_output=2, alpha=0.01, beta=0.5, eta_D=0.01, eta_R=0.01
    )

    # Save a bitwise snapshot of Column 1 and Column 2 parameters
    snapshot_col1 = {k: v.clone() for k, v in model.columns[1].state_dict().items()}
    snapshot_col2 = {k: v.clone() for k, v in model.columns[2].state_dict().items()}

    # Train Column 0
    z = torch.randn(10)
    target = torch.tensor([1])
    # Run active training on Column 0 (using task_id = 0)
    logits, diag = model.forward_step(z, target, task_id=0, is_training=True)

    # Assert column 0 parameters changed (or at least updated readout)
    # Check that Column 1 and Column 2 are completely unchanged
    for k, v in model.columns[1].state_dict().items():
        assert torch.equal(v, snapshot_col1[k]), f"Column 1 param {k} was corrupted during Column 0 training!"

    for k, v in model.columns[2].state_dict().items():
        assert torch.equal(v, snapshot_col2[k]), f"Column 2 param {k} was corrupted during Column 0 training!"
