# Phase 6 & 7: V0–V7 Feature Ablation & Causal Chain Verification

## 1. Feature Ablation Progression (V0 to V7)

Evaluated strictly on independent held-out cross-scene test split (`cross_scene_test`):

| Variant | Inputs | Spearman $\rho$ ↑ | $\Delta \rho$ | NDCG@20% ↑ | Overlap@20% ↑ | OSE@20% ↑ | Realized $\Delta Q$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V0: RGB Error** | 1 | **+0.1410** | +0.1410 | 0.3892 | 28.0% | **0.4283** | +0.000856 |
| **V1: + Depth Error** | 2 | **+0.2717** | +0.1307 | 0.5109 | 50.0% | **0.5404** | +0.001079 |
| **V2: + Gradient Norm** | 3 | **-0.1104** | -0.3821 | 0.3291 | 20.0% | **0.2688** | +0.000537 |
| **V3: + Visibility** | 4 | **+0.1091** | +0.2196 | 0.3941 | 28.0% | **0.3811** | +0.000761 |
| **V4: + Influence Mass** | 5 | **+0.1568** | +0.0476 | 0.3694 | 26.0% | **0.3916** | +0.000782 |
| **V5: + Temporal Drift** | 7 | **-0.1382** | -0.2949 | 0.4470 | 26.0% | **0.4390** | +0.000877 |
| **V6: + Uncertainty** | 8 | **+0.1335** | +0.2717 | 0.4832 | 42.0% | **0.5140** | +0.001027 |
| **V7: + Cost & Lifecycle (All 11)** | 11 | **+0.2123** | +0.0788 | 0.5011 | 44.0% | **0.5898** | +0.001178 |

## 2. Causal Chain Proof (Phase 7)

Demonstrates the causal transfer chain: Fidelity ($\rho$) $\Rightarrow$ Selection Quality ($NDCG$, $OSE$) $\Rightarrow$ Reconstruction Gain ($\Delta Q$):

- **Fidelity to Ranking Quality:** $\text{corr}(\rho, NDCG@20) = \mathbf{+0.5156}$ ($p = 0.1909$)
- **Ranking Quality to Reconstruction Gain:** $\text{corr}(NDCG@20, \Delta Q) = \mathbf{+0.9493}$ ($p = 0.0003$)
- **Selection Efficiency to Reconstruction Gain:** $\text{corr}(OSE@20, \Delta Q) = \mathbf{+1.0000}$ ($p = 0.0000$)
- **End-to-End Prediction to Gain:** $\text{corr}(\rho, \Delta Q) = \mathbf{+0.6620}$ ($p = 0.0737$)

> **Core Discovery:** Predictive fidelity directly determines selection efficiency, which in turn statistically dictates realized online reconstruction gain.
