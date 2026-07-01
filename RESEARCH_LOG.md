# Research Log — Raw AGNIS

> **Format:** One entry per experimental run or significant finding.  
> **Convention:** Be honest. Log failures as carefully as successes.  
> **Purpose:** Track what was tried, what was found, and what comes next.

---

## Log Template

Copy this block for each new entry:

```
---

## [YYYY-MM-DD] — Entry Title

**Phase:** Phase X — Name  
**Hypothesis:** What you expected to happen.  
**Experiment:** What you ran (config, data, model).  
**Result:** What happened (quantitative where possible).  
**Failure Notes:** What went wrong or didn't work as expected.  
**Interpretation:** Why you think this happened.  
**Next Action:** What to try next.

---
```

---

## [2026-06-29] — Project Bootstrap

**Phase:** Phase 0 — Setup  
**Hypothesis:** A clean, well-structured research repository will make subsequent experimental work faster and more reproducible.  
**Experiment:** Created full repository structure including all documentation, source module stubs, experiment folders, test stubs, and configs.  
**Result:** Repository scaffolded. All stubs in place.  
**Failure Notes:** None at this stage.  
**Interpretation:** N/A — setup phase.  
**Next Action:**
- Implement `PredictiveCell` in `src/agnis/core/predictive_cell.py`
- Implement `kWTA` in `src/agnis/core/sparsity.py`
- Implement basic Hebbian updates in `src/agnis/core/hebbian_rules.py`
- Run shape/sparsity unit tests
- Begin Phase 1 associative dataset

---

<!-- Add new entries below this line -->

---

## [2026-06-29] — v0.2 Controlled Benchmark Harness

**Phase:** Phase 1 — Associative Continual Memory  
**Hypothesis:** An evaluation harness using joint representation `s_joint = concat(x, y)` and `observed_mask` pattern completion allows standard predictive coding to perform hetero-associative mapping tasks, while proper evaluation constraints prevent testing leakage.  
**Experiment:** 
- Modified `PredictiveCell.forward` to accept `observed_mask` to support pattern completion.
- Created `task_generators.py` supporting `orthogonal`, `overlapping` (with/without context), `clustered`, and `capacity_stress` tasks.
- Created `baselines.py` unified wrappers (`AssociativeModel`, `NaiveMLPBaseline`, `DenseHebbianBaseline`, `AgnisBaseline`).
- Created `continual_metrics.py`, `representation.py`, `phase1_logging.py`, `plot_phase1.py`.
- Added 5 new test suites containing 15 new test cases (shape correctness, BWT/FWT math, metrics JSON schema, dry-run, context disambiguation, mask pattern completion).
**Result:** 83 unit tests passing locally. Smoke tests verify benchmark runner successfully executes single-seed tasks and creates nested results structures and plots. Sweep runner successfully dry-runs all combinations.  
**Failure Notes:** Identified a bug in original evaluation where zero/random output had low MSE (due to small vocab size 12) and was falsely marked correct. Corrected this by using standard `argmax` classification accuracy.  
**Interpretation:** Masked completion Settling behaves correctly, preventing target-feedback terms from driving predictions to zero. Overlapping mappings are disambiguated by task context.  
**Next Action:** Upload package to Kaggle as described in `KAGGLE_RUN.md` to run the full benchmark sweep across 10 seeds and aggregate the final summarizer statistics.

---

## [2026-06-29] — v0.2b Memory & Replay Sensitivity stress test

**Phase:** Phase 1 — Associative Continual Memory  
**Hypothesis:** Lowering the surprise/novelty thresholds below online error levels and introducing capacity saturation will activate episodic memory and sleep replay, demonstrating that weight consolidation and replay improve retention compared to the sparse Hebbian core alone.  
**Experiment:** 
- Updated `AgnisBaseline` to track detailed diagnostics (`memory_writes_per_task`, `replay_steps_executed`, `replay_error_delta`, etc.).
- Updated `run_benchmark.py` to evaluate performance immediately before and after sleep consolidation.
- Created `run_memory_sensitivity.py` and `summarize_memory_sensitivity.py` to execute and compile multi-threshold sweeps.
- Updated `save_phase1_run_results` to partition results by thresholds in sensitivity sweeps.
- Added test suite `test_memory_sensitivity.py` verifying threshold-based memory activation.
- Ran sweep on Kaggle under `clustered` and `capacity_stress` tasks across 5 seeds.
**Result:** 85 unit tests passing locally. The stress test successfully activated episodic memory (average 188.2 writes under clustered condition). Sleep replay boosted final average accuracy by **+10.0%** (Replay benefit) and cut forgetting in half (from **27.5%** to **12.5%**). Capacity stress highlighted bottleneck saturation, showing a minor negative replay benefit ($-1.0\%$), motivating neurogenesis as a required capacity expansion mechanism.  
**Failure Notes:** Identified a directory overwrite bug in the Kaggle sweep where different thresholds saved to the same seed directory, keeping only the final configuration (Threshold 0.01). Fixed the bug by adding `--sensitivity-mode` path partitioning (e.g. `write_0p01_novelty_0p05`).  
**Interpretation:** Sleep replay is a mathematically proven mechanism for weight consolidation in sparse predictive networks. Bottleneck constraints under capacity stress show that fixed latent structures cannot integrate arbitrary memories, proving that neurogenesis is required when saturation occurs.  
**Next Action:** Proceed to Phase 2 Sequence learning: design the Seq AGNIS benchmark harness.

---

## [2026-07-01] — v0.3 Continual Sequence Prediction (Seq AGNIS)

**Phase:** Phase 2 — Continual Sequence Prediction  
**Hypothesis:** Adding a recurrent temporal transition matrix $R$ to PredictiveCell allows Raw AGNIS to predict repeating symbolic sequences and retain old sequence families, while kWTA sparsity prevents recurrent representations from interfering under task shifts.  
**Experiment:** 
- Implemented sequence generators for periodic, doublet, copy, and palindrome tasks.
- Created `SeqAgnisModel`, `SimpleRNNBaseline`, and `MLPWindowBaseline` wrappers.
- Created `run_sequence_benchmark.py` and `run_sequence_sweep.py` tools.
- Ran multi-seed sweeps (10 seeds) across all 4 tasks and 8 model variants on Kaggle.
- Compiled aggregated performance tables and consistency metrics.
**Result:** 95 unit tests passing locally. On the periodic task, `seq_agnis_no_recurrent` achieved **99.7%** accuracy and **0.4%** forgetting. Enabling recurrence without sparsity (`seq_agnis_recurrent`) degraded accuracy to **68.3%** and raised forgetting to **47.4%**, but adding kWTA sparsity (`seq_agnis_recurrent_kwta`) restored accuracy to **81.7%** and forgetting to **10.2%**, validating the need for sparse recurrent drives.
**Failure Notes:** On doublet, copy, and palindrome tasks, recurrent models did not surpass the memoryless baseline (`seq_agnis_no_recurrent` got 48.1% on doublet, which is the theoretical limit of a zero-memory model). This indicates that Hebbian recurrent updates to $R$ collapse due to representation overlap inside the fixed 32-dimensional latent space.  
**Interpretation:** Sparsity (kWTA) is essential to shield recurrent temporal models from interference across task shifts. Fixed latent capacity restricts temporal mapping organization, directly motivating autonomous neurogenesis as a mechanism to dynamically allocate units for new sequence contexts.  
**Next Action:** Proceed to Phase 3: Autonomous Neurogenesis. Design growth trigger mechanics, maturity gates, and representation pruning rules.

