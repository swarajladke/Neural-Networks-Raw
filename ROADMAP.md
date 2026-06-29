# Raw AGNIS — Research Roadmap

> **Philosophy:** Mechanism-first. Do not scale before the mechanisms work.  
> **Claim:** "Reduced catastrophic forgetting" — until strong evidence supports stronger claims.  
> **Updated:** 2026-06-29

---

## Overview

```
Phase 0: Project Setup          [Complete]
Phase 1: Associative Memory     [Complete]
Phase 2: Sequence Learning      [Active]
Phase 3: Neurogenesis           [Planned]
Phase 4: Character Language     [Planned]
Phase 5: TinyStories Mini       [Conditional]
```

Each phase must demonstrate measurable benefits before the next begins.  
No phase is skipped. No mechanism is added without ablation.

---

## Phase 0 — Project Setup

**Goal:** Create a clean, reproducible research foundation.

**Status:** ✅ In Progress

### Deliverables
- [x] README.md — vision, philosophy, structure
- [x] ROADMAP.md — this document
- [x] RESEARCH_LOG.md — experimental log template
- [x] NOTES/ folder — design documents
- [x] `src/agnis/` — module structure
- [x] `experiments/` — phase experiment folders
- [x] `tests/` — unit test stubs
- [x] `configs/` — YAML hyperparameter files
- [x] `results/` — output folder

### Exit Criteria
- All source files importable
- All test stubs in place
- All configs parseable
- README explains project clearly

---

## Phase 1 — Raw AGNIS Micro (Associative Continual Memory)

**Goal:** Prove reduced catastrophic forgetting on sequential associative mapping tasks under multiple difficulty conditions.

**Status:** ✅ Completed (v0.2/v0.2b Swaps & Checkpoint Report Saved)

### Difficulty Conditions (Phase 1A–D)
1. **Phase 1A: Orthogonal associations** (Sanity check)
   - Disjoint input and target one-hot vectors across tasks.
2. **Phase 1B: Overlapping associations** (Interference test)
   - Same inputs map to different targets across tasks. Evaluates overwrite/interference under conflicting associations. Tested with and without task-specific context vectors.
3. **Phase 1C: Similarity-clustered associations** (Representation-overlap test)
   - Inputs are perturbed cluster center prototypes in continuous space, mapping to discrete one-hot targets.
4. **Phase 1D: Capacity stress test** (Neurogenesis-preparation test)
   - Orthogonal/overlapping tasks run under constrained latent bottlenecks ($d_z \in \{4, 8\}$) to force representation saturation.

### Core Mechanisms (this phase)
- Predictive state vector `z` settled using observed-input `observed_mask` pattern completion.
- Reconstruction/prediction error `e = observed_mask * (s - D @ a)`
- Local Hebbian update for `D` (generative) and `E` (recognition)
- kWTA sparse activation
- Fast episodic memory (surprise-based write, nearest-neighbor cosine retrieval)
- Sleep consolidation phases (offline replay)
- Double-update protection (no learning/writes during evaluation)

### Baselines & Ablations
- `mlp`: Sequential backprop MLP fine-tuning
- `dense_hebbian`: Dense associative memory
- `agnis_dense`: Raw AGNIS without kWTA sparsity, memory, or replay
- `agnis_kwta`: Raw AGNIS with kWTA sparsity, no memory or replay
- `agnis_memory`: Raw AGNIS with kWTA sparsity and fast memory, no replay
- `agnis_replay`: Raw AGNIS with kWTA sparsity, fast memory, and replay
- `agnis_full_fixed`: Full fixed-capacity model (kWTA + memory + replay)

### Metrics
- Accuracy matrix: argmax classification accuracy per task over checkpoint boundaries
- Forgetting per task and average forgetting
- Backward Transfer (BWT) and Forward Transfer (FWT)
- Mean prediction error and latent active fraction
- Final memory size and memory hit rate
- Representation overlap (cosine similarity of task prototypes) and interference risk score

### Success Criteria
- `agnis_full_fixed` average forgetting < `mlp` average forgetting
- `agnis_kwta` average forgetting < `agnis_dense` average forgetting (sparsity helps)
- `agnis_full_fixed` average forgetting < `agnis_kwta` average forgetting (memory/replay helps)
- Replay/sparsity mitigates interference under overlapping and capacity stress conditions.

### Exit Criteria / Next Steps
- Execute multi-seed sweeps on Kaggle to collect rigorous evidence.
- Once fixed-capacity benefits are proven, transition to **v0.3 Neurogenesis** to add autonomous structural growth.

---

## Phase 2 — Raw AGNIS Seq (Continual Sequence Prediction)

**Goal:** Prove reduced forgetting on sequential symbolic sequence learning.

**Status:** 🚧 Active (Designing Seq AGNIS benchmark harness)

### Tasks
Train and evaluate sequentially on:

| Sequence Family | Example | Notes |
|---|---|---|
| Repeating triplet | ABCABCABC... | Simple periodic |
| Doublet repeat | AABBCCAABB... | Grouped repetition |
| Numeric cycle | 123123123... | Digit periodic |
| Alternating | ABABAB... | Simple 2-alternation |
| Bracket-like | ABBAABBA... | Palindrome-like |
| Copy | ABCDABCD... | Copy of 4-window |

### New Mechanisms (this phase)
- Recurrent temporal state `z_prev`
- Recurrent Hebbian update for `R`
- Temporal settling with recurrent drive
- Replay of sequence prototypes during sleep phase
- Novelty detection via prediction error magnitude

### Metrics (addition to Phase 1)
- Next-symbol prediction accuracy per sequence family
- Sequence-level forgetting curve
- Replay benefit: `forgetting_no_replay - forgetting_with_replay`
- Recurrent benefit: `forgetting_no_recurrent - forgetting_with_recurrent`

### Baselines (addition to Phase 1)
- Simple RNN (backprop-through-time)
- Echo State Network (random recurrent reservoir)

### Success Criteria
- Raw AGNIS sequence forgetting < Simple RNN forgetting ✓
- Replay measurably reduces forgetting ✓
- Recurrent state improves next-symbol prediction ✓

---

## Phase 3 — Raw AGNIS Neurogenesis (Autonomous Structural Growth)

**Goal:** Demonstrate that autonomous unit birth reduces persistent prediction error and improves continual learning without uncontrolled growth.

**Status:** 🔲 Planned (begins after Phase 2 success)

### Growth Score

$$G_l(t) = \alpha \cdot \text{EMA}(\text{error}_l) + \beta \cdot \text{novelty}_l + \gamma \cdot \text{uncertainty}_l + \delta \cdot \text{interference}_l - \kappa \cdot \text{coverage}_l - \lambda \cdot \text{cost}_l$$

Growth triggers if `EMA(G_l) > threshold` for N consecutive observations.

### New Mechanisms (this phase)
- Growth score computation
- Unit birth with error-residual initialization
- Maturity gate: new units contribute proportional to maturity
- Usage tracking per unit
- Importance tracking per weight
- Redundancy detection (cosine similarity between unit weights)
- Pruning: remove low-usage, low-importance, high-redundancy units
- Merging: combine nearly-identical units

### New Unit Initialization
```
D[:, new] = normalize(residual_error)
E[new, :] = normalize(current_input)
R[new, :] = small random stable values
L[new, :] = sparse inhibitory connections
maturity_new = 0.0
plasticity_new = 1.0
importance_new = 0.0
```

### Maturity Update
```
maturity_j += eta_m * max(0, error_before_j - error_after_j)
```

### Pruning Condition
Remove unit j if:
- `usage_j < usage_threshold` AND
- `importance_j < importance_threshold` AND
- `max_cosine_similarity_to_peers > redundancy_threshold`

### Metrics (addition to Phase 2)
- Unit count over time (growth timeline)
- Error reduction per new unit
- Maturity trajectory per new unit
- Units pruned vs units born
- Forgetting: fixed-capacity vs neurogenesis-enabled

### Success Criteria
- Neurogenesis reduces persistent prediction error ✓
- New units specialize in novel patterns (measurable via activation analysis) ✓
- Maturity gate prevents destabilization of old representations ✓
- Pruning controls total capacity growth ✓
- Neurogenesis-enabled model forgetting < fixed-capacity model forgetting ✓

---

## Phase 4 — Raw AGNIS Char (Character-Level Continual Language)

**Goal:** Test Raw AGNIS on small, streaming, character-level text domains without catastrophic forgetting.

**Status:** 🔲 Planned (begins after Phase 3 success)

> ⚠️ **Note:** We do NOT claim GPT-like fluency. We measure retention and adaptation.

### Tasks (sequential text domains)
| Domain | Content | Notes |
|---|---|---|
| Simple stories | Tiny narrative text | 2–3 unique word types |
| Code-like | `if x: y else z` patterns | Structured syntax |
| Math-like | `1 + 2 = 3` expressions | Numeric patterns |
| Dialogue-like | `A: hi B: hello` exchanges | Turn-taking patterns |

### Metrics
- Character prediction accuracy per domain
- Character-level perplexity per domain
- Forgetting per domain after new domain training
- Adaptation speed (samples to reach 80% accuracy on new domain)
- Replay benefit (retention with vs without replay)
- Neurogenesis events per domain shift
- Capacity growth (total units) over all domains
- Qualitative sample quality (not primary metric)

### Success Criteria
- Better character retention than RNN baseline after domain shifts ✓
- Replay and neurogenesis show measurable benefit under domain shift ✓
- Character-level generation stable on old domains after new training ✓
- No catastrophic collapse on any domain ✓

---

## Phase 5 — TinyStories Mini (Conditional)

**Goal:** Very small-scale generative test after mechanisms are proven in Phases 1–4.

**Status:** 🔲 Conditional — only if Phases 1–4 show consistent benefits.

### Conditions to Trigger Phase 5
- Phase 1: Raw AGNIS forgetting < MLP on associative tasks ✓
- Phase 2: Sequence retention > RNN on at least 3 families ✓
- Phase 3: Neurogenesis demonstrably reduces persistent error ✓
- Phase 4: Character retention > RNN after domain shifts ✓

### Scope
- Small subset of TinyStories dataset
- Character-level or small vocabulary (512–1024 tokens) only
- Small model only (< 10M parameters in neurogenesis-grown state)
- Evaluation on continual learning metrics, not BLEU/perplexity comparison to GPT

### What Phase 5 Is NOT
- Not a GPT-2 replacement experiment
- Not a large-scale language model evaluation
- Not a fluency benchmark
- Not a scaling experiment

---

## Long-Term Vision (Post Phase 5)

If Phases 1–5 succeed, future directions include:

1. **Native continual text generation** — character or small-token generation with proven forgetting resistance
2. **Multi-modal continual learning** — symbolic + numeric + text streams
3. **Active learning integration** — model requests more examples from confusing inputs
4. **Hierarchical neurogenesis** — grow entire columns, not just single units
5. **Comparison to EWC, PackNet, ProgressiveNets** — formal continual learning benchmark comparison
6. **Possible integration with Hybrid AGNIS** — use Raw AGNIS mechanisms to improve the predictive memory bridge in Hybrid AGNIS

---

## Timeline (First 8 Weeks)

| Week | Focus | Key Output |
|---|---|---|
| 1 | Setup + Core | PredictiveCell, kWTA, Hebbian, unit tests |
| 2 | Phase 1 Experiment | Phase 1 dataset, trainer, forgetting metric, baselines |
| 3 | Memory | Fast memory, replay, forgetting curves |
| 4 | Recurrence | Recurrent state, Phase 2 start |
| 5 | Growth | Growth score, unit birth, maturity gate |
| 6 | Pruning | Pruning, ablations, neurogenesis evaluation |
| 7 | Char | Character-level toy tasks |
| 8 | Consolidate | Research report, stop/go decision |

---

## Versioning Convention

| Tag | Meaning |
|---|---|
| `v0.x` | Phase 0 — setup and scaffolding |
| `v1.x` | Phase 1 — associative memory |
| `v2.x` | Phase 2 — sequence learning |
| `v3.x` | Phase 3 — neurogenesis |
| `v4.x` | Phase 4 — character language |
| `v5.x` | Phase 5 — TinyStories mini |

---

*Last updated: 2026-06-29*
