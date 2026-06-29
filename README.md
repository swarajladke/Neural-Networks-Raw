# Raw AGNIS — Autonomous Generative Neuroplastic Intelligence System

> **Status:** Phase 2 — Seq AGNIS Continual Sequence Prediction  
> **Last Updated:** 2026-06-29  
> **Research Phase:** Active benchmark design and implementation

---

## What Is Raw AGNIS?

Raw AGNIS is a **standalone continual learning neural architecture** built from first principles. It is not a language model competitor. It is not a transformer variant. It is a rigorous mechanism-first research system designed to answer one question:

> *Can a neural network learn new things continuously without catastrophically forgetting old ones?*

Raw AGNIS pursues this through a combination of:
- **Predictive coding** — the network predicts its input; error drives learning
- **Local Hebbian updates** — no global backprop dependency for core mechanisms
- **Sparse representations** — k-Winners-Take-All (kWTA) reduces interference
- **Fast episodic memory** — novel/high-error experiences stored as prototypes
- **Slow semantic consolidation** — repeated useful patterns migrate to stable weights
- **Replay / sleep phases** — offline replay of important memories consolidates knowledge
- **Autonomous neurogenesis** — grow new units when capacity is exhausted
- **Maturity-gated integration** — new units earn influence only by reducing error
- **Pruning and merging** — remove dead or redundant capacity

---

## What Raw AGNIS Is NOT

| Raw AGNIS | NOT Raw AGNIS |
|---|---|
| Standalone continual learner | GPT competitor |
| Mechanism-first architecture | Scale-first system |
| Local Hebbian core | Pure backpropagation-only system |
| Sparse predictive coding | Dense transformer |
| Proven on toy tasks first | Pre-trained on large corpora first |
| Claims "reduced forgetting" | Claims "beats LLMs" |

---

## Relationship to Hybrid AGNIS

This project is entirely **separate** from the Neural-Networks project (Hybrid AGNIS).

| Dimension | Hybrid AGNIS | Raw AGNIS |
|---|---|---|
| **Repository** | `Neural-Networks/` | `Neural-Networks-Raw/` |
| **Core idea** | Hebbian bridge to frozen GPT-2 | Standalone continual learner |
| **LLM dependency** | Yes (frozen GPT-2) | None |
| **Generation goal** | Fluent language via LLM | Mechanism proof first |
| **Architecture** | Predictive coding + LoRA/injection | Predictive coding + neurogenesis |
| **Primary goal** | Near-term practical system | Long-term research system |
| **Current phase** | Active development | Phase 0 bootstrap |

**Transferable insights from Hybrid AGNIS** (ideas only, no shared code):
- Predictive hierarchy design
- Local Hebbian update intuitions
- Latent settling dynamics
- Lateral inhibition / kWTA
- Recurrent temporal state
- Novelty detection signals
- Prediction-error as learning driver
- Memory/replay ideas
- Plasticity control mechanisms
- Failure lessons from gate/bridge instability

**Not transferred:**
- GPT-2 hooks or transformer code
- Hidden-state injection
- LoRA/adapters
- Large tokenizer assumptions
- LLM fluency assumptions

---

## Project Roadmap (Summary)

| Phase | Name | Goal | Status |
|---|---|---|---|
| 0 | Setup | Repository, docs, structure | ✅ Complete |
| 1 | Micro | Continual associative memory | ✅ Complete |
| 2 | Seq | Continual sequence prediction | 🚧 Active |
| 3 | Neurogenesis | Autonomous structural growth | 🔲 Planned |
| 4 | Char | Character-level continual language | 🔲 Planned |
| 5 | TinyStories | Small-scale generative test | 🔲 Conditional |

See [ROADMAP.md](./ROADMAP.md) for full detail.

---

## Repository Structure

```
Neural-Networks-Raw/
├── README.md
├── ROADMAP.md
├── RESEARCH_LOG.md
├── NOTES/
│   ├── raw_agnis_principles.md
│   ├── insights_from_hybrid_agnis.md
│   ├── predictive_coding_equations.md
│   ├── neurogenesis_design.md
│   └── evaluation_metrics.md
├── src/
│   └── agnis/
│       ├── core/           # PredictiveCell, settling, Hebbian, sparsity
│       ├── memory/         # FastMemory, replay buffer, consolidation
│       ├── neurogenesis/   # Growth controller, birth, maturity, pruning
│       ├── sequence/       # Recurrent state, sequence predictor, char model
│       ├── training/       # Online trainer, sleep trainer, curriculum
│       ├── evaluation/     # Forgetting metrics, baselines, probes
│       └── utils/          # Config, logging, visualization
├── experiments/
│   ├── phase1_associative/
│   ├── phase2_sequences/
│   ├── phase3_neurogenesis/
│   ├── phase4_char_language/
│   └── phase5_tinystories/
├── configs/
│   ├── phase1_micro.yaml
│   ├── phase2_seq.yaml
│   ├── phase3_neurogenesis.yaml
│   └── phase4_char.yaml
├── tests/
│   ├── test_predictive_cell.py
│   ├── test_hebbian_update.py
│   ├── test_sparsity.py
│   ├── test_fast_memory.py
│   ├── test_forgetting.py
│   └── test_neurogenesis.py
├── notebooks/
│   ├── 01_associative_memory_demo.ipynb
│   ├── 02_sequence_learning_demo.ipynb
│   └── 03_neurogenesis_visualization.ipynb
└── results/
    └── .gitkeep
```

---

## First Experiment: Phase 1 Associative Continual Learning

**Goal:** Prove that Raw AGNIS can learn sequential associative mappings with less catastrophic forgetting than simple baselines.

**Tasks (sequential):**
- Task 1: `A→B`, `C→D`
- Task 2: `E→F`, `G→H`
- Task 3: `I→J`, `K→L`

**Protocol:**
1. Train Task 1 → Evaluate Task 1
2. Train Task 2 → Evaluate Task 1, Task 2
3. Train Task 3 → Evaluate Task 1, Task 2, Task 3

**Baselines:**
- Naive MLP (sequential backprop)
- Dense Hebbian associative memory
- Raw AGNIS without sparsity (ablation)

**Success criteria:**
- Raw AGNIS forgetting < naive MLP forgetting
- Sparsity clearly reduces interference
- Forgetting curves are measurable and reproducible

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Neural-Networks-Raw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies and editable package
pip install -e .
```

---

## Running Benchmark & Tests

> ⚠️ **Compute Warning:** Do NOT run heavy multi-seed sweeps locally. All long-running sweeps should be run on Kaggle. See [KAGGLE_RUN.md](experiments/phase1_associative/KAGGLE_RUN.md) for details.

### 1. Run Unit Tests (Local Validation)
```bash
python -m pytest tests/ -q
```

### 2. Local Smoke Test (Fast validation)
```bash
python experiments/phase1_associative/run_benchmark.py \
  --condition orthogonal \
  --model agnis_kwta \
  --seed 0 \
  --config configs/phase1_smoke.yaml \
  --smoke
```

### 3. Kaggle Sweep Run (High compute)
```bash
python experiments/phase1_associative/run_sweep.py \
  --config configs/kaggle_phase1.yaml \
  --conditions orthogonal overlapping clustered capacity_stress \
  --models mlp dense_hebbian agnis_dense agnis_kwta agnis_memory agnis_replay agnis_full_fixed \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

## Core Research Claims (Honest)

We claim only what we can measure:

| Claim | Evidence Required |
|---|---|
| "Raw AGNIS shows **reduced** catastrophic forgetting" | Must beat naive MLP on Phase 1 |
| "Sparsity reduces interference" | Ablation: with vs without kWTA |
| "Replay improves retention" | Ablation: with vs without replay |
| "Neurogenesis reduces persistent error" | Phase 3 ablation study |

We do **NOT** yet claim:
- "Solves catastrophic forgetting"
- "Matches or beats transformer-based systems"
- "Achieves fluent generation"

---

## Success Criteria by Phase

**Phase 1:** Raw AGNIS forgetting < MLP baseline on associative tasks  
**Phase 2:** Raw AGNIS sequence retention > simple RNN baseline  
**Phase 3:** Neurogenesis demonstrably reduces persistent prediction error  
**Phase 4:** Character prediction stable on old domains after new domain training  
**Phase 5 (conditional):** Only if Phases 1–4 show consistent mechanism-level benefits

---

## Stop/Go Criteria

**Continue if:**
- Beats simple baselines on forgetting in Phase 1 or 2
- Sparsity clearly reduces interference
- Replay measurably improves retention
- Neurogenesis reduces persistent error
- Growth is controllable with pruning

**Pause if:**
- Cannot beat simple baselines on toy tasks
- Most time goes to debugging generation instead of continual learning
- Requires large compute before showing mechanism results
- Significantly delays Hybrid AGNIS progress

---

## Research Philosophy

1. Do not scale before the mechanisms work.
2. Do not chase GPT-like fluency first.
3. Do not overbuild.
4. Every mechanism must be measurable.
5. Every claim must have a baseline.
6. Every component must have an ablation.
7. Use "reduced catastrophic forgetting" until strong evidence supports stronger claims.
8. Start small → prove memory → prove sequences → prove neurogenesis → then scale.

---

## Project Lead

Personal research project. Connected to Hybrid AGNIS work in Neural-Networks repository as a parallel research track.

---

*"The goal is not to build something big. The goal is to build something that actually works."*
