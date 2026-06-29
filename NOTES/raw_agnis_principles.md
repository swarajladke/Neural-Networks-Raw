# Raw AGNIS Core Principles

> These principles guide every design decision in the Raw AGNIS project.  
> When in doubt, return to these principles before adding complexity.

---

## 1. Predictive

**Principle:** The network should always be predicting something. Learning is driven by prediction error, not by a loss function imposed from outside.

**In practice:**
- Every layer predicts the layer below it (generative pathway)
- Every layer updates to better encode the layer above it (recognition pathway)
- Prediction error `e = s - s_hat` is the central signal
- High error means surprise. Low error means mastery.

**Why:** Predictive coding is biologically plausible, local, and naturally produces a signal that is both a learning driver and a novelty detector. It does not require a global loss function.

---

## 2. Sparse

**Principle:** At any given moment, only a small fraction of units should be active.

**In practice:**
- Use k-Winners-Take-All (kWTA): keep top-k activations, suppress the rest
- Typical sparsity: 5–15% of units active
- Lateral inhibition enforces sparsity within layers
- Sparse representations reduce overlap between representations of different inputs
- Overlap reduction directly reduces catastrophic interference

**Why:** If two tasks activate the same units, training on one degrades the other. Sparse, separated representations minimize this overlap. This is well-supported in neuroscience and in machine learning (sparse autoencoders, capsule networks, etc.).

**Ablation:** Always test with and without kWTA to verify sparsity actually helps.

---

## 3. Local

**Principle:** Weight updates should depend only on locally available information — pre-synaptic activity, post-synaptic activity, and local error.

**In practice:**
- Generative Hebbian: `ΔD = η_D * outer(e, a)`
- Recognition Hebbian: `ΔE = η_E * outer((z - E@s), s)`
- Recurrent Hebbian: `ΔR = η_R * outer(z, z_prev)`
- Plasticity gates are local per-weight scalars
- No global loss gradient propagation through the entire network

**Why:** Global backprop requires storing the entire computation graph and propagating gradients across all layers. This is expensive, biologically implausible, and creates tight inter-layer coupling that makes continual learning harder. Local rules are more modular and naturally support incremental updates.

**Caveat:** PyTorch is used for implementation convenience. Global backprop may be used in baselines and for validation. But the core Raw AGNIS mechanisms should not rely on global backprop for their primary learning signal.

---

## 4. Memory-Based

**Principle:** Not all information can be learned in weights alone. Fast episodic memory provides a separate, rapidly-writable store for novel or surprising events.

**In practice:**
- **Fast memory:** Key-value prototype store. Write when prediction error or novelty is high.
- **Slow weights:** Updated gradually through repeated exposure and replay.
- **Two timescales:** Fast memory catches rare events. Slow weights capture statistics.

**Why:** Biological brains use hippocampus (fast) and cortex (slow). This separation allows both rapid one-shot storage and gradual generalization. Without fast memory, a single-stream continual learner either forgets old patterns (if learning rate is high) or learns new ones slowly (if learning rate is low).

---

## 5. Neurogenic

**Principle:** If existing capacity cannot explain persistent novelty, grow new capacity.

**In practice:**
- Track prediction error per layer over time (EMA)
- If error remains high despite learning and memory, trigger neurogenesis
- New units initialized from residual error (they are designed to explain what existing units cannot)
- New units start with high plasticity, low maturity
- New units earn influence by reducing prediction error over repeated exposures

**Why:** Fixed-capacity networks eventually saturate. Rather than preemptively overprovisioning capacity (which wastes resources and increases interference), grow capacity only when necessary.

**Key constraint:** Growth must be measurable. Every neurogenesis event must be logged and its effect on prediction error tracked.

---

## 6. Consolidating

**Principle:** Memories that prove useful across many experiences should migrate from fast episodic memory to stable slow weights.

**In practice:**
- During sleep/replay phases, sample high-importance memories from fast store
- Reconstruct inputs from memory prototypes
- Apply slow weight updates (reduced learning rate, reduced plasticity)
- Protect weights with high importance scores from large updates

**Why:** Fast memory has finite capacity and is vulnerable to overwrite. Important patterns should be "burned in" to stable weights over time. This mirrors hippocampal-cortical consolidation in neuroscience.

---

## 7. Pruning-Aware

**Principle:** Growth without pruning leads to uncontrolled capacity bloat. Every unit that is born can also die.

**In practice:**
- Track per-unit: usage count, importance score, contribution to error reduction
- Track per-unit-pair: redundancy (cosine similarity of weight vectors)
- Prune units that are: low-usage AND low-importance AND high-redundancy
- Merge nearly-identical units into a single representative unit
- Apply pruning on a schedule (not continuously — avoid thrashing)

**Why:** Without pruning, neurogenesis creates an ever-growing network. Pruning controls capacity, removes dead units that add noise, and enforces a useful inductive bias toward efficient representation.

---

## 8. Continually Learning

**Principle:** Every mechanism must contribute to the core goal: learning new things without catastrophically forgetting old ones.

**In practice:**
- Continual learning is the primary evaluation criterion at every phase
- No mechanism is added without measuring its effect on forgetting
- Forgetting metric: `F_i = max_prev_acc_i - current_acc_i`
- Average forgetting reported across all tasks at each phase boundary
- Ablation table required before claiming any mechanism helps

**Why:** It is easy to build a complex system. It is hard to build one where each part provably helps. The discipline of measuring forgetting at every step forces honest assessment.

---

## Summary Table

| Principle | Implementation | Measurable By |
|---|---|---|
| Predictive | State settling, prediction error | Error curves |
| Sparse | kWTA, lateral inhibition | % active units |
| Local | Hebbian update rules | Weight change localization |
| Memory-based | FastMemory, replay buffer | Memory size, retrieval accuracy |
| Neurogenic | Growth score, unit birth | Unit count, error reduction |
| Consolidating | Sleep trainer, slow weights | Retention after sleep |
| Pruning-aware | Usage/importance/redundancy tracking | Units born vs pruned |
| Continually learning | Forgetting metric, task accuracy | F_i, avg forgetting |

---

*Last updated: 2026-06-29*
