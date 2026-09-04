# Gaussian State Representation & Feature Specification

## 1. Conceptual Architecture: Persistent State vs Frame-Local Observations

To eliminate index aliasing and guarantee causality in learned utility estimation ($s_i(t) \to U_i^\star$), state signals are strictly divided into two categories:

1. **Frame-Local Observations $o_i(t)$**: Quantities measured solely from the current frame rendering and pre-intervention scene geometry.
2. **Persistent State $s_i^{\text{persist}}(t)$**: Historical and lifecycle signals maintained across time by `GaussianStateStore` indexed by unique, immutable `persistent_id`.

$$\text{Pipeline: } o_i(t) \longrightarrow \text{GaussianStateStore.update\_frame} \longrightarrow s_i(t) \longrightarrow \text{Utility Estimator}$$

```
                RGB-D frame t
                     │
                     ▼
             GaussianModel G_t
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Gaussian geometry      Rendering / attribution
          │                     │
          └──────────┬──────────┘
                     ▼
              State signals o_i(t)
                     │
                     ▼
             GaussianStateStore
                     │
       persistent identity + history
                     │
                     ▼
              feature vector s_i(t)
                     │
                     ▼
             Utility estimator
```

---

## 2. Invariants & Lifecycle Semantics

### Invariant A — ID Uniqueness
$$i \ne j \implies \text{persistent\_id}_i \ne \text{persistent\_id}_j$$
- `_next_id` is monotonically non-decreasing. Re-allocation or reuse of old IDs is prohibited.

### Invariant B — Reorder and Compaction Invariance
$$\pi(\mathbf{G}) \implies \mathbf{s}_{\pi(i)} = \mathbf{s}_i$$
- When tensor indices shift due to pruning, sorting, or compaction, state signals follow the `persistent_id`, never the physical tensor index.

### Separation of Concerns: Engine Execution vs Research State (Point D)
- **`GaussianModel._state`**: Low-level engine execution flags (`UNSTABLE=0, STABLE=1, FROZEN=2, PRUNED=3`). Dictates rasterizer memory compaction and CUDA kernel filtering.
- **`GaussianStateStore`**: Research-level persistent state, identity provenance, continuous signals (EMAs, drift, age, staleness), and budget priority tiers (`Tier 0..3`).
- **Synchronization Invariant**: At all lifecycle boundaries (`initialize_from_points`, `compact`, `add_gaussians`, `reorder`), the invariant `model.num_gaussians == model.state_store.num_gaussians` is strictly asserted.

### Pruning Lifecycle
- Pruned Gaussians are evicted from the active state tensors and archived in `_pruned_registry` with `pruned_frame` and historical lineage intact for retrospective lineage queries (`get_lineage(id)`).

### Densification Lineage & Dimension Validation (Point C)
- Spawning $K$ children from parents $P$ assigns unique IDs $C_1 \dots C_K$ with `parent_id = P.persistent_id` (not tensor index).
- **Dimension Invariant**: `add_gaussians(new_params, parent_indices, n_children_per_parent)` strictly validates that $N_{\text{new}} = |\text{parent\_indices}| \times n_{\text{children\_per\_parent}}$ (or $N_{\text{new}} = |\text{parent\_indices}|$ when $1$-to-$1$), preventing parameter vs state store count desynchronization.
- **Initialization Policy**: `fresh` (per `protocol_v1.yaml`), children initialize with prior uncertainty ($0.5$), zero EMA, age $0$. (Alternative `inherit` copies parent EMA).

### Exact Temporal Semantics & Feature Unification (Point B)
- **Age**: $\text{age}_i(t) = t - t_{\text{creation}, i}$
- **Staleness**: $\text{staleness}_i(t) = t - t_{\text{last\_update}, i}$
- **EMA Decay**: $\text{EMA}_t = \beta \text{EMA}_{t-1} + (1 - \beta) x_t$ with $\beta = 0.90$.
- **Temporal Factor Unification**: The canonical temporal state factors are `position_drift` ($d_i(t) = \|\mu_i(t) - \mu_i(t-1)\|_2$) and `residual_drift_ema` ($r_i(t) = \text{EMA}(|e_i(t) - e_i(t-1)|)$). `temporal_drift` is retained strictly as a property alias for `position_drift` to prevent feeding duplicate collinear features into downstream utility estimators.

### Visibility Semantics (Point A)
- **`visibility_count`**: Exact integer/float count of screen pixels influenced by Gaussian $g_i$ ($V_i(t) = \#\{\text{pixels influenced}\}$ from attribution footprint).
- **`ema_visibility`**: Identity-preserving exponential moving average of visibility over time.


---

## 3. Feature Availability Matrix

| Feature | Mathematical Definition | Signal Source | Frame Timing | Persistent? | Normalization | Used by Model? | Potential Leakage? |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `rgb_error` | $\|I(\mu_i) - I_{gt}(\mu_i)\|_1$ | Rasterizer / Screen residual | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None (Pre-intervention) |
| `ema_rgb` | $\beta \text{EMA}_{t-1} + (1-\beta) e_{\text{rgb}}(t)$ | `GaussianStateStore` | Historical ($t-1 \to t$) | Yes | Yes (Z-score) | Yes | None |
| `depth_error` | $\|D(\mu_i) - D_{gt}(\mu_i)\|_1$ | Rasterizer / Depth residual | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None (Pre-intervention) |
| `ema_depth` | $\beta \text{EMA}_{t-1} + (1-\beta) e_{\text{depth}}(t)$ | `GaussianStateStore` | Historical ($t-1 \to t$) | Yes | Yes (Z-score) | Yes | None |
| `gradient_norm` | $\|\nabla_{\mu} \mathcal{L}_{\text{pre}}\|_2$ | Pre-opt Autograd backward | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None (Pre-intervention) |
| `gradient_ema` | $\beta \text{EMA}_{t-1} + (1-\beta) g(t)$ | `GaussianStateStore` | Historical | Yes | Yes (Z-score) | Yes | None |
| `visibility_count` | $\sum_u \mathbb{I}[w_{u, i} > 0]$ | Attribution Footprint | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None |
| `ema_visibility` | $\beta \text{EMA}_{t-1} + (1-\beta) V(t)$ | `GaussianStateStore` | Historical | Yes | Yes (Z-score) | Yes | None |
| `influence_mass` | $\sum_u \alpha_i T_i(u)$ | Surface-aware attribution | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None |
| `ema_influence` | $\beta \text{EMA}_{t-1} + (1-\beta) M(t)$ | `GaussianStateStore` | Historical | Yes | Yes (Z-score) | Yes | None |
| `position_drift` | $\|\mu_i(t) - \mu_i(t-1)\|_2$ | Position delta | Pre-scheduling ($t$) | Yes (Derived) | Yes (Z-score) | Yes | None |
| `residual_drift_ema`| $\beta \text{EMA}_{t-1} + (1-\beta) \|e_t - e_{t-1}\|$ | Error delta EMA | Historical | Yes (Derived) | Yes (Z-score) | Yes | None |
| `uncertainty_var` | Prior confidence variance | `GaussianStateStore` | Persistent | Yes | Yes (Z-score) | Yes | None |
| `age` | $t - t_{\text{creation}, i}$ | `GaussianStateStore` | Lifecycle | Yes | Yes (Z-score) | Yes | None |
| `staleness` | $t - t_{\text{last\_update}, i}$ | `GaussianStateStore` | Lifecycle | Yes | Yes (Z-score) | Yes | None |
| `projected_area` | $\pi r_x r_y$ (Screen footprint) | Geometry projection | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None |
| `tile_footprint` | Cardinality of touched 16x16 tiles | Tile binner | Pre-scheduling ($t$) | No (Obs) | Yes (Z-score) | Yes | None |

---

## 4. Execution Pipeline & Leakage Elimination

To strictly prevent future information leakage:
1. **Step 1 (Render)**: Render current state at frame $t$.
2. **Step 2 (Observe)**: Compute residuals $e_{\text{rgb}}, e_{\text{depth}}$, influence mass $M$, and pre-intervention gradient norms $g_{\text{pre}}$.
3. **Step 3 (Update StateStore)**: Pass pre-intervention signals into `GaussianStateStore.update_frame()`.
4. **Step 4 (Extract Feature Vector)**: Construct $s_i(t)$ by concatenating normalized observations and persistent state signals.
5. **Step 5 (Utility Prediction)**: Learned Two-Head model infers $\hat{u}_i(t) = \hat{q}_i(t) / \hat{c}_i(t)$.
6. **Step 6 (Schedule & Optimize)**: Scheduler selects subset $S_t \subseteq \{1 \dots N\}$, optimizer updates $S_t$.
7. **Step 7 (Record Optimization)**: Record `optimized_mask` in `GaussianStateStore.last_update_frames`.

**Leakage Audit Guarantee**: No post-intervention metric (e.g. gradient after optimization step, delta quality achieved by that step) is ever exposed to Step 5.

---

## 5. Normalization Provenance Contract

Per `configs/protocol_v1.yaml` (`pre_fusion_normalization: true`):
- Normalization statistics $(\mu_j, \sigma_j)$ are computed **strictly on the canonical training split (frames 0–40)**.
- Parameters are persisted in `results/statistics/normalization.json`.
- Validation (frames 41–60) and Test (`tum_fr2_xyz`) splits apply frozen training statistics without re-fitting.
