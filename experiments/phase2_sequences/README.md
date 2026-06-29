# Phase 2 — Seq AGNIS (Continual Sequence Prediction)

This directory contains the benchmark runner, sweep orchestrator, generators, plotting, and analysis tools for **Phase 2 Sequence Prediction**.

## Objective
Evaluate whether Raw AGNIS can learn and retain temporal structure continually across sequence task shifts using a recurrent predictive state $R$, comparing it to recurrent and sliding-window baselines.

## File Contents
- `run_sequence_benchmark.py`: Main sequence prediction orchestrator.
- `run_sequence_sweep.py`: Multi-seed multi-model sweep runner.
- `sequence_generators.py` (via `src/agnis/sequence/sequence_tasks.py`): Sequence cycle generators.
- `plot_phase2.py`: Visualizes accuracy matrices and recurrent state drive curves.
- `summarize_phase2.py`: Combines sweep seeds into aggregated summaries.
- `KAGGLE_RUN_PHASE2.md`: Guide to executing sequence prediction sweeps on Kaggle.

## Local Validation
Do not execute long sweeps locally. Run the lightweight test suite and a single smoke test:
```bash
python -m pytest tests/ -q
python experiments/phase2_sequences/run_sequence_benchmark.py --condition periodic --model seq_agnis_recurrent --seed 0 --config configs/phase2_smoke.yaml --smoke
```
