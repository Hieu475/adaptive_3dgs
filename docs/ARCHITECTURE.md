# Architecture Overview: Adaptive 3D Gaussian Splatting

This document details the end-to-end architecture of the **Adaptive 3D Gaussian Splatting under Compute Budget** framework, highlighting the separation of concerns between the **Advanced Machine Learning** core and the **AI Systems** optimization layer.

---

## 1. End-to-End Pipeline

```
                     RGB-D Sensor Stream (I_t, D_t)
                                   │
                                   ▼
             3D Gaussian Scene Representation G_t = {g_1, ..., g_N}
              (Position μ_i, Covariance Σ_i, Opacity α_i, Color c_i)
                                   │
                                   ▼
                Pre-Intervention Feature Extraction (s_i ∈ R^11)
            (Photometric, Depth, Gradient, Visibility, Uncertainty)
                                   │
                                   ▼
                Learned Utility Predictor (Two-Head MLP / P4)
                ├── Branch 1: Expected Reconstruction Gain ΔQ_hat
                └── Branch 2: Expected Execution Time Cost C_hat
                                   │
                                   ▼
              Budget-Constrained Knapsack Selection (S_B ⊆ G_t)
                     max Σ ΔQ_hat_i  s.t.  Σ C_hat_i ≤ B_t
                                   │
                                   ▼
              Selective Optimization Engine (SelectiveAdam)
              ├── Active: Gradient descent applied only to S_B
              └── Passive: FrozenBackgroundCache preserves unselected map
                                   │
                                   ▼
              High-Fidelity Online Map & Novel View Synthesis Q(t)
```

---

## 2. Decoupling: Advanced ML vs AI Systems

### 2.1. Advanced Machine Learning Layer
- **Causal Counterfactual Oracle ($U_i^\star = \Delta Q_i / C_i$)**: Evaluates ground-truth marginal utility via isolated trial interventions, proving that **$20.5\%$** of unconstrained gradient updates degrade reconstruction quality ($U_i^\star < 0$).
- **State Representation ($s_i \in \mathbb{R}^{11}$)**: Pre-fusion feature extraction with train-only normalization anchoring.
- **Two-Head Decoupled Architecture**: Decouples scalar quality from execution duration, trained via margin-weighted pairwise ranking.
- **Contextual Interaction Modeling (Phase 6)**:
  - Formulates conditional marginal utility $U^*(i \mid S_t)$ where $S_t$ is the set of already-selected Gaussians.
  - Characterizes sub-additivity in volume rendering ($\rho(\text{IoU}, |I|) = 0.5357$).
  - **Empirical Scientific Resolution (Case B)**: While contextual interactions attenuate utility magnitude, candidate priority ranks remain substantially invariant ($\bar{\rho}_{\text{rank}} = \mathbf{0.8916} \pm \mathbf{0.1104}$, Kendall $\bar{\tau} = \mathbf{0.8043}$, Top-5 overlap = $\mathbf{80.0\%}$).

### 2.2. AI Systems Layer
- **Budget Enforcement**: Guarantees that selective optimization adheres to a rigid frame time budget $B_t$ (e.g., $15.0\text{ ms}$).
- **Selective Optimization Runtime**: `SelectiveAdam` operates on sparse index masks; `FrozenBackgroundCache` eliminates redundant alpha compositing passes for inactive regions.
- **Runtime Profile & Bottleneck Analysis**:
  Profiling across pipeline stages reveals a critical systems finding:

  $$T_{\text{stage}} = T_{\text{feat}} + T_{\text{ctx}} + T_{\text{MLP}} + T_{\text{sel}} + T_{\text{opt}} = 924.59\text{ ms}$$

  | Stage | Latency | Share | Systems Observation |
  | :--- | :---: | :---: | :--- |
  | **Feature Extraction** | $225.79\text{ ms}$ | 24.4% | Screen attribution rendering and error mass reduction |
  | **Context Construction** | $66.84\text{ ms}$ | 7.2% | Spatial KNN query and pairwise screen-space IoU projection |
  | **Model Inference (MLP)** | **$1.12\text{ ms}$** | **0.1%** | **PyTorch forward pass is practically negligible** |
  | **Subset Selection** | $76.36\text{ ms}$ | 8.3% | Iterative adaptive greedy sort across $|S_B|$ steps ($420\times$ vs static) |
  | **Gaussian Optimization** | $554.48\text{ ms}$ | 60.0% | GPU CUDA backward kernels and Adam parameter updates |

  > **Key Systems Takeaway**: *Model inference itself is cheap ($1.12\text{ ms}$); state construction and selection orchestration dominate adaptive overhead.*

---

## 3. Single Source of Truth & Provenance

All quantitative statements in this repository are verified by our 417-test regression suite (`pytest -q`) and anchored to:
- `results/phase6_context_utility/manifest.json` (Phase 6 Single Source of Truth)
- `results/phase6_context_utility/rank_stability_analysis.json` (Rank Stability & Coverage)
- `results/phase6_context_utility/ablation/ablation_summary.json` (8-Variant Architecture Ladder)
- `results/phase6_context_utility/runtime_breakdown.json` (Per-Stage Runtime Breakdown)

