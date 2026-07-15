# SPARC Research Checkpoint & Transition Guide

This document provides a complete end-to-end context, architectural review, and progress status of the **SPARC (Sparse Predictive Adversarial/Associative Recurrent Columns)** project for transition to your new machine.

---

## 1. Project Context & Overarching Goal
The goal of this project is to build a continual sequence learning model (**Raw AGNIS / SPARC**) that:
1. **Acquires sequence patterns** without representation collapse (the monolithic recurrent system collapsed completely to a **4% random guessing floor**).
2. **Mitigates catastrophic forgetting** during sequential training across distinct domains (animals, objects, emotions, actions).
3. Achieves **task-ID-free inference** by dynamically routing inputs to specialized predictive coding columns.

---

## 2. Completed Milestones & Findings

### Phase 0: Baseline Freeze
* **Milestone:** Integrated simple RNN/GRU, EWC, and Replay baselines into the comparative sweep framework.
* **Key Finding:** Monolithic deep AGNIS stack collapses to `4.0%` accuracy across all seeds due to representational overlap. Mainstream recurrent baselines achieve `38%–44%` evaluation accuracy but suffer from `20%–32%` catastrophic forgetting.

### Phase 1A: Oracle & Nonparametric Routing
* **Milestone:** Implemented `PredictiveColumn` (proximal gradient settling, backtracking line search, Lipschitz step size) and nonparametric routers (`NearestPrototypeRouter`).
* **Key Finding:** Broke the 4% collapse floor! Oracle and prototype-routing configurations reached **`45.2%` peak accuracy** and maintained **`37.8%` final evaluation accuracy** with forgetting restricted to **`7.3%–7.5%`** (a ~70% reduction in forgetting relative to baselines).

### Phase 1B: Differentiable Learned Routing (v0.2 Spec)
* **Milestone:** Implemented differentiable top-1 learned routing, robust Median/MAD energy calibration, causal context input features, and state-safe minimum-energy evaluations.
* **Key Finding:** Resolved all autograd and state leakages. All **144 unit tests** pass successfully, validating parameter freezing, context boundary resets, and optimizer parameter ownership.

---

## 3. Core Repository File structure
When you resume on your new laptop, these are the key modules:

* **[column.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/src/agnis/sparc/column.py):** Implements `PredictiveColumn` with proximal settling, backtracking line-search, and consecutive-convergence early stopping.
* **[learned_router.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/src/agnis/sparc/learned_router.py):** Implements `DifferentiableTopKRouter` using deterministic straight-through top-1 routing and project-context mapping.
* **[minimum_energy_router.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/src/agnis/sparc/minimum_energy_router.py):** Implements `MinimumEnergyRouter` with robust Median/MAD calibration floors and state-safe clone-based candidate settling.
* **[sparc_model.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/src/agnis/sparc/sparc_model.py):** Coordinates column banks, routes inputs, and implements Router C mixture logsumexp probability structures.
* **[sequence_wrapper.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/src/agnis/sequence/sequence_wrapper.py):** Connects SPARC with sequential benchmarks, managing task-transition router optimizer updates, calibration, and cached route distillation.
* **[test_sparc_routing.py](file:///c:/Users/Helios/Desktop/Neural-Networks-Raw/tests/test_sparc_routing.py):** Automated unit test suite verifying v0.2 routing safety constraints.

---

## 4. Current Verification Benchmarks (Pass Gates)
Before proceeding to new features, verify that your new laptop runs:

1. **Unit Tests:**
   ```bash
   PYTHONPATH=src:. pytest tests/
   ```
2. **Smoke Test Command:**
   ```bash
   PYTHONPATH=src:. python experiments/phase6_deep_stack/run_deep_benchmark.py --model sparc_energy_distilled --smoke
   ```

---

## 5. Next Steps
Once the Kaggle sweep is complete, analyze:
* **Router A vs B vs C:** Does the distilled self-supervised router (B) match the supervised task-label router (A) and mixture-loss router (C)?
* **Routing Regret:** Track `task_loss_routing_regret` to ensure it approaches zero.
* **Early-Stopping:** Compare the runtime of early-stopping vs full-settling configurations.
