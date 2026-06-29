# Insights from Hybrid AGNIS

> **Purpose:** Document which ideas, observations, and lessons from the Hybrid AGNIS project (Neural-Networks/) are transferable to Raw AGNIS (Neural-Networks-Raw/), and which are explicitly not transferred.  
> **Important:** No code is shared. Only conceptual insights.

---

## Background

Hybrid AGNIS is a practical near-term architecture that uses a frozen GPT-2/LLM as a fluent generation backbone, with a Hebbian predictive coding core providing continual memory through gated hidden-state injection. It was designed to be usable now while LLMs are the dominant paradigm.

Raw AGNIS is a different project: a standalone continual learner with no LLM dependency, designed to work from first principles.

---

## Transferable Ideas (Concepts Only, Not Code)

### 1. Predictive Hierarchy Design
**What was learned:** Organizing computation as a hierarchy of prediction-error-driven layers (bottom-up error, top-down prediction) is a viable and tractable architecture. The settling dynamics (iterative state updates toward reduced error) are implementable in PyTorch with reasonable stability.

**Raw AGNIS implication:** Use the same conceptual structure for `PredictiveCell` and `PredictiveHierarchy`. Each layer predicts the layer below and updates to minimize local reconstruction error.

### 2. Local Hebbian Update Rules
**What was learned:** Simple outer-product Hebbian rules (`ΔW = η * pre ⊗ post_error`) produce meaningful weight updates. The key is defining "pre" and "post_error" carefully — post_error should be related to prediction error at that layer, not just post-synaptic activation.

**Raw AGNIS implication:** Use `ΔD = η_D * outer(e, a)` and `ΔE = η_E * outer((z - E@s), s)` as the primary update rules. These are the cleanest form of local Hebbian learning in a predictive coding framework.

### 3. Latent Settling Dynamics
**What was learned:** The iterative settling of a latent state `z` toward reduced prediction error is stable when the step size `η_z` is small and activation functions are bounded (e.g., tanh). Instability arises when lateral or recurrent connections are too strong or not initialized carefully.

**Raw AGNIS implication:**
- Use bounded activations (tanh or normalized ReLU)
- Use small `η_z` (0.01–0.1 range)
- Apply gradient clipping or norm clipping during settling
- Watch for NaN during settling in tests

### 4. Lateral Inhibition / kWTA
**What was learned:** k-Winners-Take-All sparsity is easy to implement and measurably separates representations. The key parameter is k (or equivalently, the sparsity fraction). Too sparse → representational collapse. Too dense → high interference.

**Raw AGNIS implication:**
- Start with k = 10% of units (configurable)
- Test k in range [5%, 25%] in ablations
- Implement as a hard mask (top-k values kept, others zeroed)
- Expose `sparsity_level` as a logged metric at every step

### 5. Recurrent Temporal State
**What was learned:** A recurrent connection `z_t = f(z_{t-1})` provides memory of the previous timestep. In Hybrid AGNIS, this was used to condition the predictive state on temporal context. It stabilized sequence-level representations.

**Raw AGNIS implication:** Add recurrent state to `PredictiveCell` in Phase 2. Use `d_time = R @ z_prev` as a recurrent drive term. Update `R` with `ΔR = η_R * outer(z, z_prev)`.

### 6. Novelty Detection via Prediction Error
**What was learned:** High prediction error is a reliable novelty signal. When the model encounters something it has not seen before, error spikes. This can be used to gate memory writes (write to fast memory only when error is high) and to gate plasticity (allow large updates only when error is high).

**Raw AGNIS implication:**
- Use `novelty = EMA(|e|)` as the primary novelty signal
- Gate fast memory writes: write if `novelty > write_threshold`
- Gate plasticity: `plasticity_ij = sigmoid(a * novelty - b * importance_ij - c * age_ij)`

### 7. Prediction Error as Learning Driver
**What was learned:** Using prediction error directly as the learning signal (rather than a separate loss function) ties learning tightly to the model's current predictive accuracy. It naturally slows learning on well-mastered inputs and accelerates it on novel ones.

**Raw AGNIS implication:** This is the foundation of the Raw AGNIS update rules. The error signal `e = s - D @ a` directly drives both the generative (`ΔD`) and recognition (`ΔE`) updates.

### 8. Memory / Replay Ideas
**What was learned:** Simple prototype-based memory (store compressed representations of high-error events) with cosine-similarity retrieval is surprisingly effective for associative recall. Replay during a "sleep" phase (offline from the main learning loop) avoids interference between replay updates and online learning.

**Raw AGNIS implication:**
- Implement `FastMemory` as a key-value store with cosine retrieval
- Separate `OnlineTrainer` (live task updates) from `SleepTrainer` (replay-based consolidation)
- Replay in sleep phase with reduced plasticity to protect old representations

### 9. Plasticity Control
**What was learned:** Uniform plasticity leads to interference. Weight-specific plasticity (some weights should be frozen/rigid, others plastic) allows the model to protect important representations while staying plastic for new learning. Importance-weighted plasticity (less important weights stay more plastic) is a simple, effective heuristic.

**Raw AGNIS implication:**
- Track `importance_ij = EMA(|ΔW_ij|)` per weight
- Apply `plasticity_ij = sigmoid(... - c * importance_ij ...)`
- High-importance weights (frequently and meaningfully updated) become rigid
- Low-importance weights remain plastic for new learning

### 10. Failure Lessons from Gate/Bridge Instability
**What was learned:** The gate mechanism in Hybrid AGNIS (controlling how much the predictive memory state influences the frozen LLM hidden states) was a significant source of instability. When gate values became saturated or collapsed, the predictive core effectively disconnected from or completely dominated the LLM output. Gating mechanism design requires very careful initialization and monitoring.

**Raw AGNIS implication:**
- For maturity gating in neurogenesis: initialize maturity at 0, increase only through demonstrated error reduction. Never allow new units to immediately influence the full representation.
- For plasticity gating: monitor plasticity distribution. If plasticity collapses near 0 globally, the model stops learning.
- Always log gate/maturity/plasticity statistics during training.

---

## Non-Transferable Components

These components exist in Hybrid AGNIS (Neural-Networks/) and must NOT appear in Raw AGNIS:

| Component | Reason Not Transferred |
|---|---|
| GPT-2 model and hooks | Raw AGNIS has no LLM dependency |
| `transformers` library usage | No transformer architecture in Raw AGNIS |
| Hidden-state injection (`model.inject_hidden`) | No frozen backbone to inject into |
| LoRA / PEFT adapters | No frozen model to adapt |
| Large tokenizer (GPT-2 tokenizer, 50k vocab) | Raw AGNIS uses symbols or characters |
| Prompt engineering / soft prompts | Not applicable |
| Large-scale pretraining assumptions | Raw AGNIS starts from scratch |
| GPT-like fluency metrics (BLEU, etc.) | Not the primary goal of Raw AGNIS |
| Any code importing `torch.nn.modules.activation` for transformer attention | Not relevant |
| Multi-head self-attention | Not used in Raw AGNIS core |

---

## Key Lessons Summary

| Lesson | Impact on Raw AGNIS |
|---|---|
| Settling stability requires bounded activations | Use tanh, clip norms |
| kWTA separates representations measurably | Core mechanism, ablate |
| Prediction error = novelty signal naturally | Gate memory writes and plasticity on error |
| Replay in separate sleep phase avoids interference | Implement `SleepTrainer` as distinct loop |
| Gate/maturity mechanisms need careful init | Maturity starts at 0, grows only on merit |
| Importance-weighted plasticity works | Track `importance_ij`, gate updates |
| Recurrent state stabilizes sequence representations | Add in Phase 2, ablate |
| Measure forgetting at every phase | Core evaluation loop |

---

## Research Connection

Hybrid AGNIS and Raw AGNIS are linked through research notes only. Future results from Raw AGNIS may:
- Inform improvements to the Hebbian predictive memory in Hybrid AGNIS
- Validate whether mechanisms proven on toy tasks transfer to LLM-bridged architectures
- Provide a theoretical baseline for understanding what the Hybrid AGNIS predictive core is actually doing

But there is no shared code. The two projects maintain separate codebases, separate philosophies, and separate evaluation criteria.

---

*Last updated: 2026-06-29*
