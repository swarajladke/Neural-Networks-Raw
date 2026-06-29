# Predictive Coding Equations — Raw AGNIS

> **Purpose:** Mathematical specification of the core update rules and settling dynamics used in Raw AGNIS.  
> **Convention:** All vectors are column vectors. `@` denotes matrix multiplication. `⊗` or `outer()` denotes outer product.  
> **Reference:** Rao & Ballard (1999), Friston (2005), Millidge et al. (2021), and related predictive coding literature.

---

## Variable Definitions

| Symbol | Shape | Description |
|---|---|---|
| `s` | `(d_in,)` | Input (stimulus) vector at current timestep |
| `z` | `(d_z,)` | Latent state vector (internal representation) |
| `a` | `(d_z,)` | Activation: `a = φ(z)` |
| `φ` | — | Activation function (tanh or bounded nonlinearity) |
| `φ'` | — | Derivative of φ |
| `D` | `(d_in, d_z)` | Generative (decoding) weight matrix |
| `E` | `(d_z, d_in)` | Recognition (encoding) weight matrix |
| `R` | `(d_z, d_z)` | Recurrent weight matrix |
| `L` | `(d_z, d_z)` | Lateral inhibition matrix (sparse, typically negative off-diagonal) |
| `z_prev` | `(d_z,)` | Latent state from previous timestep |
| `ŝ` | `(d_in,)` | Predicted reconstruction of input |
| `e` | `(d_in,)` | Prediction error vector |
| `η_z` | scalar | Settling step size |
| `η_D` | scalar | Generative weight learning rate |
| `η_E` | scalar | Recognition weight learning rate |
| `η_R` | scalar | Recurrent weight learning rate |
| `ρ` | scalar | Recurrent drive strength |
| `λ_lat` | scalar | Lateral inhibition strength |
| `λ_sparse` | scalar | L1 sparsity penalty strength |

---

## 1. Prediction and Reconstruction

### Activation
$$a = \varphi(z)$$

### Predicted Reconstruction
$$\hat{s} = D \cdot a$$

### Prediction Error
$$e = s - \hat{s} = s - D \cdot a$$

*This is the core signal. High error means the model is surprised. Low error means the model has mastered this input.*

---

## 2. State Settling

The latent state `z` is updated iteratively to minimize prediction error while integrating multiple drives: recognition (bottom-up), feedback (top-down), recurrent (temporal), and lateral (inhibitory).

### Drive Terms

**Recognition drive** (bottom-up encoding signal):
$$d_{\text{rec}} = E \cdot s - z$$

*Pulls z toward the encoding of the current input.*

**Feedback drive** (top-down error signal):
$$d_{\text{fb}} = D^T \cdot e \cdot \varphi'(z)$$

*Propagates prediction error back from the reconstruction space into latent space. Element-wise multiplication with φ'(z).*

**Recurrent drive** (temporal context):
$$d_{\text{time}} = R \cdot z_{\text{prev}}$$

*Injects information from the previous timestep's latent state.*

**Lateral drive** (inhibitory competition):
$$d_{\text{lat}} = L \cdot a$$

*Enforces competitive suppression. L contains negative off-diagonal values to create mutual inhibition.*

### Settling Update (one step)
$$z \leftarrow z + \eta_z \left( d_{\text{rec}} + d_{\text{fb}} + \rho \cdot d_{\text{time}} + \lambda_{\text{lat}} \cdot d_{\text{lat}} - \lambda_{\text{sparse}} \cdot \text{sign}(z) \right)$$

This is iterated for `T_settle` steps (typically 5–20) per input presentation.

### kWTA Sparsity (applied after each settling step or at end)
$$a_k = \begin{cases} a_k & \text{if } a_k \text{ is in top-}k \text{ by magnitude} \\ 0 & \text{otherwise} \end{cases}$$

Only the k largest-magnitude activations survive. All others are zeroed. This is applied to `a = φ(z)` and may feed back into subsequent settling steps.

---

## 3. Hebbian Weight Updates

All weight updates are applied **after** settling is complete (i.e., after `z` has converged or completed its settling steps).

### Generative (Decoding) Update
$$\Delta D = \eta_D \cdot e \otimes a = \eta_D \cdot \text{outer}(e, a)$$

*Shape: `(d_in, d_z)`. The generative matrix learns to reconstruct `s` from `a`. High-error directions in input space are strengthened for active units.*

### Recognition (Encoding) Update
$$\Delta E = \eta_E \cdot (z - E \cdot s) \otimes s = \eta_E \cdot \text{outer}(z - E \cdot s, s)$$

*Shape: `(d_z, d_in)`. The recognition matrix learns to encode `s` into `z`. The error term `(z - E@s)` is the difference between the settled latent state and the direct linear encoding.*

### Recurrent Update
$$\Delta R = \eta_R \cdot z \otimes z_{\text{prev}} = \eta_R \cdot \text{outer}(z, z_{\text{prev}})$$

*Shape: `(d_z, d_z)`. The recurrent matrix learns temporal associations between consecutive latent states.*

---

## 4. Plasticity-Gated Update

For continual learning, we do not want all weights to update equally. Weights that are already important for well-learned patterns should be more resistant to change.

### Importance Tracking
$$\text{importance}_{ij} \leftarrow (1 - \alpha_I) \cdot \text{importance}_{ij} + \alpha_I \cdot |\Delta W_{ij}|$$

*EMA of absolute weight changes. High-importance weights are those that change frequently and substantially.*

### Plasticity
$$\text{plasticity}_{ij} = \sigma\!\left( a_p \cdot \text{novelty} + b_p \cdot \text{uncertainty} - c_p \cdot \text{importance}_{ij} - d_p \cdot \text{age}_{ij} \right)$$

Where:
- `novelty = EMA(|e|)` — how surprising is the current input?
- `uncertainty = Var(z)` across settling steps — how unstable is the latent state?
- `importance_ij` — how critical is this weight for past learning?
- `age_ij` — how long since this weight was meaningfully updated?
- `σ` = sigmoid function

### Plasticity-Gated Weight Update
$$\Delta W_{ij} = \eta \cdot \text{plasticity}_{ij} \cdot \text{pre}_i \cdot \text{post\_error}_j$$

*Each weight updates proportionally to its plasticity. Important weights barely move. New or unused weights move freely.*

---

## 5. Memory Write Condition

$$\text{write to fast memory if:} \quad \text{novelty} > \theta_{\text{write}} \quad \text{OR} \quad |e| > \theta_{\text{error}}$$

Where:
- `novelty = EMA(|e|)` 
- `θ_write`, `θ_error` are configurable thresholds

Memory entry stored as:
```
key   = a          (sparse latent activation — compact, distinctive)
value = s          (full input — for reconstruction during replay)
error = |e|        (magnitude of prediction error at write time)
importance = 0.0   (initialized to 0, updated based on retrieval frequency)
timestamp = t
usage_count = 0
```

---

## 6. Memory Retrieval

Given a query activation `a_q`, find the nearest stored prototype:

$$k^* = \arg\max_{k \in \text{Memory}} \frac{a_q \cdot \text{key}_k}{|a_q| \cdot |\text{key}_k|}$$

Retrieve `value_{k^*}` as the recalled input.

---

## 7. Replay (Sleep Phase)

During the sleep/consolidation phase, sample `M` memories from fast memory (weighted by importance):

For each sampled memory `(key_m, value_m)`:
1. Reconstruct: set `s = value_m`
2. Run settling to convergence
3. Compute `e = s - D @ a`
4. Apply weight updates with reduced plasticity: `η_sleep = η * γ_sleep` where `γ_sleep < 1`
5. Protect high-importance weights: skip update if `importance_ij > θ_protect`

---

## 8. Neurogenesis Growth Score

$$G_l(t) = \alpha \cdot \text{EMA}(\text{error}_l) + \beta \cdot \text{novelty}_l + \gamma \cdot \text{uncertainty}_l + \delta \cdot \text{interference}_l - \kappa \cdot \text{coverage}_l - \lambda \cdot \text{cost}_l$$

Where:
- `error_l` = mean reconstruction error at layer l
- `novelty_l` = EMA of `|e_l|` over recent inputs
- `uncertainty_l` = variance of latent state during settling
- `interference_l` = estimated cosine similarity between current and stored activations (high overlap = high interference risk)
- `coverage_l` = fraction of memory slots utilized (high coverage = low need to grow)
- `cost_l` = current number of units / max budget (growth cost)

**Growth condition:**
$$\text{trigger neurogenesis if:} \quad \text{EMA}(G_l) > \theta_{\text{grow}} \quad \text{for } N_{\text{consec}} \text{ consecutive observations}$$

---

## 9. New Unit Contribution (Maturity Gate)

When a new unit `j` is born, its contribution to the overall latent representation is:

$$z_{\text{eff}, j} = \text{maturity}_j \cdot \varphi(z_j)$$

**Maturity update** (incremented only when the unit demonstrably reduces error):

$$\text{maturity}_j \leftarrow \text{maturity}_j + \eta_m \cdot \max\!\left(0,\ e_{\text{before},j} - e_{\text{after},j}\right)$$

**Maturity clipping:** `maturity_j ∈ [0, 1]`

A new unit starts at `maturity = 0` (no influence) and approaches `maturity = 1` only if it consistently reduces prediction error.

---

## 10. Summary of Key Equations

| Equation | Role |
|---|---|
| `ŝ = D @ a` | Prediction / reconstruction |
| `e = s - ŝ` | Prediction error (core signal) |
| `z ← z + η_z(d_rec + d_fb + ρ·d_time + λ_lat·d_lat - λ_s·sign(z))` | State settling |
| `ΔD = η_D · outer(e, a)` | Generative Hebbian update |
| `ΔE = η_E · outer(z - E@s, s)` | Recognition Hebbian update |
| `ΔR = η_R · outer(z, z_prev)` | Recurrent Hebbian update |
| `ΔW_ij = η · plasticity_ij · pre_i · post_error_j` | Plasticity-gated update |
| `importance_ij ← EMA(|ΔW_ij|)` | Importance tracking |
| `maturity_j ← maturity_j + η_m · max(0, e_before_j - e_after_j)` | Maturity update |
| `G_l = α·error + β·novelty + ... - κ·coverage - λ·cost` | Growth score |

---

*Last updated: 2026-06-29*
