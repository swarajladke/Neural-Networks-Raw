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
