# Phase 1 — Associative Continual Memory

## Goal
Prove that Raw AGNIS can learn sequential associative mappings with less catastrophic forgetting than simple baselines.

## Tasks
- Task 0: A→B, C→D (one-hot vectors, vocab size 12)
- Task 1: E→F, G→H
- Task 2: I→J, K→L

## Protocol
1. Train Task 0 → Evaluate Task 0
2. Train Task 1 → Evaluate Task 0, Task 1
3. Train Task 2 → Evaluate Task 0, Task 1, Task 2

## Models Compared
- Naive MLP (sequential backprop, no memory) — WORST CASE BASELINE
- Dense Hebbian (Hebbian, no sparsity) — MECHANISM BASELINE
- Raw AGNIS (no sparsity) — ABLATION: sparsity removed
- Raw AGNIS (no memory) — ABLATION: fast memory removed
- Raw AGNIS (full) — FULL MODEL

## Running

```bash
# From project root
python experiments/phase1_associative/run_phase1.py

# With custom settings
python experiments/phase1_associative/run_phase1.py --seed 42 --n_repeats 100 --d_z 64 --k_sparse 6

# Multiple seeds
for seed in 0 1 2 3 4; do
    python experiments/phase1_associative/run_phase1.py --seed $seed --results_dir results/phase1/seed_$seed/
done
```

## Success Criteria
- Raw AGNIS (full) avg_forgetting < Naive MLP avg_forgetting ✓
- Raw AGNIS (full) avg_forgetting < Dense Hebbian avg_forgetting ✓ (sparsity helps)
- Raw AGNIS (no sparsity) forgetting > Raw AGNIS (full) forgetting ✓ (sparsity ablation)
- Raw AGNIS (no memory) forgetting > Raw AGNIS (full) forgetting ✓ (memory ablation)

## Output Files
- `results/phase1/forgetting_curves_{model}.png` — per-model forgetting curves
- `results/phase1/forgetting_comparison.png` — bar chart comparison
