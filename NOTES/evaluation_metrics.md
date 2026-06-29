# Evaluation Metrics — Raw AGNIS

> **Purpose:** Define all metrics used to evaluate Raw AGNIS across all phases.  
> **Principle:** Every claim must have a measurement. Every component must have an ablation.

---

## Core Continual Learning Metrics

### 1. Task Accuracy
$$A_{i,t} = \text{accuracy of model on task } i \text{ after training on task } t$$

Reported as a matrix where rows are tasks, columns are evaluation checkpoints.

### 2. Forgetting (per task)
$$F_i = \max_{t \leq T_i} A_{i,t} - A_{i,T}$$

The drop from peak performance on task i to final performance on task i.  
- `T_i` = last time task i was trained  
- `T` = final evaluation checkpoint  
- Always ≥ 0. Zero = no forgetting.

### 3. Average Forgetting
$$\bar{F} = \frac{1}{N-1} \sum_{i=1}^{N-1} F_i$$

Averaged over all tasks except the last (which has not yet been forgotten).

### 4. Backward Transfer (BWT)
$$\text{BWT} = \frac{1}{N-1} \sum_{i=1}^{N-1} \left( A_{i,T} - A_{i,T_i} \right)$$

Negative BWT = forgetting. Positive BWT = later training helps old tasks (rare but possible with replay).

### 5. Forward Transfer (FWT)
$$\text{FWT} = \frac{1}{N-1} \sum_{i=2}^{N} \left( A_{i, T_{i-1}} - A_{i}^{\text{random}} \right)$$

How much has training on previous tasks helped on the new task before seeing any examples.  
Requires a "random initialization" baseline accuracy `A_i^random`.

### 6. Adaptation Speed
Number of training examples needed to reach 80% accuracy on a new task (per task).  
Lower = faster adaptation.

---

## Prediction Quality Metrics

### 7. Prediction Error (MSE)
$$\text{MSE}_l = \frac{1}{d} \sum_{j=1}^{d} e_j^2 = \frac{1}{d} |s - \hat{s}|^2$$

Mean squared reconstruction/prediction error at layer l.

### 8. Prediction Error EMA
$$\text{EMA\_error}_l(t) = (1 - \alpha) \cdot \text{EMA\_error}_l(t-1) + \alpha \cdot \text{MSE}_l(t)$$

Smooth moving average of prediction error. Used in growth score and novelty detection.

---

## Representation Quality Metrics

### 9. Sparsity Level
$$\text{sparsity}_l = \frac{\text{# zero units in } a_l}{d_{z,l}}$$

Fraction of units inactive after kWTA. Reported per layer.

### 10. Representation Overlap
For two tasks i and j, measure the cosine similarity between their average activations:
$$\text{overlap}_{ij} = \frac{\bar{a}_i \cdot \bar{a}_j}{|\bar{a}_i| |\bar{a}_j|}$$

High overlap → high interference risk. Ablation: overlap should decrease when kWTA is enabled.

### 11. Settling Convergence
Number of settling steps required to reach stable `z` (change < threshold).  
High settling steps = model is confused or conflicted.

---

## Memory Metrics

### 12. Memory Utilization
`# stored prototypes / max_memory_capacity`

### 13. Memory Retrieval Accuracy
Fraction of retrieval queries that return the correct (nearest) prototype.

### 14. Replay Benefit
$$\text{replay\_benefit} = \bar{F}_{\text{no\_replay}} - \bar{F}_{\text{with\_replay}}$$

Positive = replay helps. Zero = replay has no effect.

---

## Neurogenesis Metrics

### 15. Units Born (cumulative)
Total number of new units created across all tasks.

### 16. Units Pruned (cumulative)
Total number of units removed across all tasks.

### 17. Net Growth
`units_born - units_pruned` — net capacity change.

### 18. Maturity Distribution
Histogram of maturity values across all units at each checkpoint.
Expected: bimodal — mature units (maturity ≈ 1) and young units (maturity ≈ 0–0.3).

### 19. Unit Specialization
For each new unit, fraction of its top-10 activations that belong to the task that triggered its birth.

### 20. Growth Score Trace
Logged value of `G_l(t)` over time. Used to understand when and why growth triggered.

---

## Ablation Table Template

| Mechanism | Average Forgetting (↓) | Task 1 Acc | Task 2 Acc | Task 3 Acc | Sparsity | Units |
|---|---|---|---|---|---|---|
| Naive MLP baseline | ? | ? | ? | ? | 100% | fixed |
| Dense Hebbian baseline | ? | ? | ? | ? | 100% | fixed |
| Raw AGNIS — no kWTA | ? | ? | ? | ? | 100% | fixed |
| Raw AGNIS — no memory | ? | ? | ? | ? | ? | fixed |
| Raw AGNIS — no replay | ? | ? | ? | ? | ? | fixed |
| Raw AGNIS — no recurrent | ? | ? | ? | ? | ? | fixed |
| Raw AGNIS — full (fixed cap) | ? | ? | ? | ? | ? | fixed |
| Raw AGNIS + neurogenesis | ? | ? | ? | ? | ? | growing |
| Raw AGNIS + neurogenesis + pruning | ? | ? | ? | ? | ? | controlled |

Fill in with actual numbers at each phase.

---

## Reporting Conventions

1. All metrics reported as mean ± std over **5 random seeds**.
2. Forgetting curves plotted as task accuracy vs. training step (not just endpoint).
3. Sparsity level logged every 100 steps.
4. Memory utilization logged every 100 steps.
5. Growth score logged every step (for neurogenesis phases).
6. All results stored in `results/phase{N}/` with seed-indexed files.
7. Summary CSVs produced at end of each experiment.

---

*Last updated: 2026-06-29*
