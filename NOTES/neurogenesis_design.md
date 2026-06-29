# Neurogenesis Design — Raw AGNIS

> **Purpose:** Detailed design specification for autonomous structural growth in Raw AGNIS.  
> **Phase:** Phase 3 — Raw AGNIS Neurogenesis  
> **Prerequisite:** Fixed-capacity Raw AGNIS must work on Phases 1 and 2 first.  
> **Status:** Design complete. Implementation planned for Week 5.

---

## Motivation

Fixed-capacity neural networks face a fundamental trade-off in continual learning:
- **Too small:** Not enough capacity to learn new patterns without overwriting old ones.
- **Too large:** Excessive interference between many partially-used representations.

**Neurogenesis** resolves this by growing capacity only when it is needed — when existing units cannot explain persistent novelty without causing interference.

The key constraint is: **growth must be controlled**. Without pruning, neurogenesis leads to bloat. Without maturity gating, new units destabilize old representations.

---

## Growth Score

The growth score `G_l(t)` at layer `l` at time `t` aggregates multiple signals that together indicate whether new capacity is needed:

$$G_l(t) = \alpha \cdot \underbrace{\text{EMA}(\text{error}_l)}_{\text{persistent error}} + \beta \cdot \underbrace{\text{novelty}_l}_{\text{input surprise}} + \gamma \cdot \underbrace{\text{uncertainty}_l}_{\text{settling instability}} + \delta \cdot \underbrace{\text{interference}_l}_{\text{representation overlap}} - \kappa \cdot \underbrace{\text{coverage}_l}_{\text{memory fullness}} - \lambda \cdot \underbrace{\text{cost}_l}_{\text{current size}}$$

### Component Definitions

| Component | Computation | Intuition |
|---|---|---|
| `EMA(error_l)` | Exponential moving average of `mean(|e_l|)` | High persistent error = need more capacity |
| `novelty_l` | EMA of `|e_l| - EMA(|e_l|)` when positive | Recent input much more surprising than average |
| `uncertainty_l` | Variance of `z_l` across settling steps | Unstable settling = representation conflict |
| `interference_l` | Avg cosine similarity between current `a_l` and k-nearest stored activations | High overlap = new input conflicts with stored ones |
| `coverage_l` | `# filled memory slots / max_memory_slots` | Full memory = less room for new episodes |
| `cost_l` | `current_units_l / max_units_l` | Growth cost — penalizes large existing capacity |

### Hyperparameters (Phase 3 starting values)
```yaml
growth_alpha: 1.0    # persistent error weight
growth_beta: 0.5     # novelty weight
growth_gamma: 0.3    # uncertainty weight
growth_delta: 0.4    # interference weight
growth_kappa: 0.3    # memory coverage penalty
growth_lambda: 0.2   # capacity cost penalty
growth_threshold: 0.6
growth_consecutive_n: 10  # consecutive observations above threshold to trigger
```

---

## Birth Condition

```python
if EMA(G_l, window=50) > growth_threshold:
    consecutive_count += 1
else:
    consecutive_count = 0

if consecutive_count >= growth_consecutive_n:
    trigger_neurogenesis(layer=l)
    consecutive_count = 0
```

**Why EMA instead of instantaneous G?** A single surprising input should not trigger growth. Growth should only happen when persistent, sustained difficulty exceeds what existing capacity can handle.

---

## New Unit Initialization

When a new unit `j` is born at layer `l`:

### Weight Initialization

**Generative weights** (column of D corresponding to new unit):
$$D[:, j] = \frac{e_{\text{residual}}}{|e_{\text{residual}}|}$$

The new unit is initialized to explain the current residual error. It is "born" from what the existing network cannot predict.

**Recognition weights** (row of E corresponding to new unit):
$$E[j, :] = \frac{s_{\text{current}}}{|s_{\text{current}}|}$$

The new unit learns to recognize the current input pattern.

**Recurrent weights** (row of R corresponding to new unit):
$$R[j, :] \sim \mathcal{N}(0, \sigma_R^2), \quad \sigma_R = 0.01$$

Small random stable values. The new unit should not disrupt temporal dynamics.

**Lateral connections** (column/row of L corresponding to new unit):
$$L[j, :] = L[:, j] = \text{sparse}, \quad \sim \mathcal{U}(-0.1, 0)$$

Sparse inhibitory connections. New unit participates in competition but weakly.

### State Variables

```python
new_unit.maturity = 0.0          # no influence yet
new_unit.plasticity = 1.0        # maximally plastic
new_unit.importance = 0.0        # no history
new_unit.usage_count = 0         # never used
new_unit.birth_time = t          # for age tracking
new_unit.probation_until = t + N_probation  # evaluation window
```

---

## Maturity Gate

New units do not immediately contribute to the full latent representation. Their effective contribution is gated by their maturity score:

$$z_{\text{eff}, j} = \text{maturity}_j \cdot \varphi(z_j)$$

### Maturity Update

After each forward pass, update maturity based on error reduction:

```python
error_before = mean(|e|)   # before new unit contributes
error_after  = mean(|e'|)  # after new unit contributes

maturity_j += eta_maturity * max(0, error_before - error_after)
maturity_j  = clip(maturity_j, 0.0, 1.0)
```

**Key properties:**
- Maturity only increases if the unit reduces prediction error.
- Maturity never decreases (units cannot un-earn their influence).
- A unit that consistently fails to reduce error stays at low maturity and will be pruned.

### Maturity Trajectory (expected)

| Time (after birth) | Expected Maturity | Notes |
|---|---|---|
| t+0 | 0.00 | Born with no influence |
| t+10 | ~0.05 | Early specialization, small contribution |
| t+50 | ~0.30 | Unit finding its niche |
| t+200 | ~0.70 | Well-established, meaningful contribution |
| t+500+ | ~1.00 | Fully mature, integrated |

If maturity does not reach 0.1 by `probation_until`, the unit is a pruning candidate.

---

## Usage and Importance Tracking

### Per-Unit Usage
$$\text{usage}_j \leftarrow (1 - \alpha_u) \cdot \text{usage}_j + \alpha_u \cdot \mathbb{1}[a_j > 0]$$

Exponential moving average of whether unit j was active in this forward pass.

### Per-Weight Importance
$$\text{importance}_{ij} \leftarrow (1 - \alpha_I) \cdot \text{importance}_{ij} + \alpha_I \cdot |\Delta W_{ij}|$$

EMA of absolute weight changes. High-importance weights change frequently and substantially.

### Per-Unit Redundancy
$$\text{redundancy}_j = \max_{k \neq j} \frac{D[:, j] \cdot D[:, k]}{|D[:, j]| \cdot |D[:, k]|}$$

Maximum cosine similarity between unit j's generative weights and any other unit's generative weights. High redundancy means two units are doing essentially the same thing.

---

## Pruning

Pruning is applied on a schedule (every `prune_interval` steps, not continuously).

### Pruning Condition

Unit `j` is a pruning candidate if **all three** conditions hold:

```python
is_low_usage       = usage_j < usage_threshold          # e.g., 0.05
is_low_importance  = max(importance_j[:]) < importance_threshold  # e.g., 0.01
is_high_redundancy = redundancy_j > redundancy_threshold  # e.g., 0.90
```

Additionally, units past their probation period with maturity < 0.05 are prune candidates regardless of the above.

### Pruning Action

1. Remove unit j from D, E, R, L (delete corresponding rows/columns)
2. Remove unit j's state variables
3. Log: `{unit_id: j, birth_time: ..., death_time: t, maturity: ..., reason: ...}`

### Merging

If two units have redundancy > `merge_threshold` (e.g., 0.95), merge them:

```python
D[:, keep] = normalize(D[:, keep] + D[:, merge])
E[keep, :] = normalize(E[keep, :] + E[merge, :])
importance[keep] = max(importance[keep], importance[merge])
usage[keep] = max(usage[keep], usage[merge])
# Delete 'merge' unit
```

---

## Routing and Layer Assignment

When neurogenesis is triggered:
- By default, add new units to the layer with the highest growth score.
- Later versions may add entire micro-columns (small groups of units at multiple layers).

For Phase 3, start with single-unit growth at a single layer. Multi-layer column growth is a Phase 3+ extension.

---

## Neurogenesis Logging

Every neurogenesis event should be logged with:

```json
{
  "event": "birth",
  "unit_id": 42,
  "layer": 0,
  "time": 1250,
  "trigger_growth_score": 0.73,
  "error_at_birth": 0.42,
  "novelty_at_birth": 0.38,
  "interference_at_birth": 0.61
}
```

Every pruning event:
```json
{
  "event": "prune",
  "unit_id": 42,
  "layer": 0,
  "time": 3100,
  "maturity_at_death": 0.02,
  "usage_at_death": 0.01,
  "reason": "probation_failed"
}
```

---

## Experiments (Phase 3)

### Experiment 3.1: Does neurogenesis reduce persistent error?
- Run Phase 1 tasks with fixed-capacity Raw AGNIS (baseline) and neurogenesis-enabled Raw AGNIS.
- Measure: prediction error at end of each task for both models.
- Expected: neurogenesis-enabled model has lower persistent error on novel tasks.

### Experiment 3.2: Do new units specialize?
- After neurogenesis events, track which inputs each new unit activates on.
- Measure: cosine similarity between new unit's receptive field and task-specific inputs.
- Expected: new units activate preferentially on inputs from the task that triggered their birth.

### Experiment 3.3: Does maturity gating prevent destabilization?
- Compare: immediate full-influence new units vs maturity-gated new units.
- Measure: forgetting on old tasks immediately after neurogenesis.
- Expected: maturity-gated units cause less forgetting on old tasks.

### Experiment 3.4: Does pruning control growth?
- Run all three phases with neurogenesis enabled.
- Measure: total unit count over time with and without pruning.
- Expected: pruning keeps total unit count bounded.

### Experiment 3.5: Does neurogenesis reduce forgetting vs fixed capacity?
- Full comparison: Fixed capacity vs neurogenesis + pruning on Phase 1 + 2 combined task sequence.
- Report average forgetting for both.
- Expected: neurogenesis + pruning < fixed-capacity forgetting.

---

## Success Criteria (Phase 3)

| Criterion | Measurement | Pass Threshold |
|---|---|---|
| Neurogenesis reduces persistent error | Error at end of task 3 | Must be < fixed-capacity baseline |
| New units specialize | Activation selectivity | > 50% of new unit's activations on trigger-task inputs |
| Maturity gating prevents destabilization | Old task accuracy immediately post-birth | Must not drop > 5% more than no-neurogenesis case |
| Pruning controls growth | Total unit count | Must not exceed 2× initial capacity |
| Forgetting comparison | Average forgetting F_avg | Neurogenesis < fixed-capacity |

---

## Failure Modes to Watch

| Failure Mode | Detection | Response |
|---|---|---|
| Growth explosion | Total units > max budget | Tighten growth threshold, increase λ_cost |
| Maturity never increases | All new units stay at maturity ≈ 0 | Check error_before/error_after computation |
| Pruning kills useful units | Old task accuracy drops after prune | Increase importance_threshold |
| NaN in D[:, j] after init | Test for NaN in birth routine | Add normalization safety (avoid zero residual) |
| Growth never triggers | G_l always below threshold | Check interference and novelty computation |

---

*Last updated: 2026-06-29*
