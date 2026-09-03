# Phase 4: Learned Marginal Utility & State Factor Analysis Report

## 1. Univariate Predictive Power Analysis (Point XV - Level A)

| State Variable ($x_j$) | Spearman $\rho(x_j, U^\star)$ | p-value | Significance |
|:---|:---:|:---:|:---:|
| **rgb_error** | +0.1529 | 0.0536 | Not Significant |
| **depth_error** | +0.0355 | 0.6563 | Not Significant |
| **visibility** | -0.0117 | 0.8829 | Not Significant |
| **influence_mass** | -0.0109 | 0.8907 | Not Significant |
| **temporal_drift** | +nan | nan | Not Significant |
| **uncertainty** | -0.1127 | 0.1559 | Not Significant |
| **gradient_norm** | +0.0011 | 0.9886 | Not Significant |
| **projected_area** | -0.0109 | 0.8907 | Not Significant |
| **age** | -0.0538 | 0.4991 | Not Significant |
| **update_frequency** | +nan | nan | Not Significant |

## 2. Conditional Incremental Information (Point XV - Level B)

| Model Variant | Inputs | Spearman $\rho$ ↑ | $\Delta \rho$ | NDCG@20% ↑ | Overlap@20% ↑ | OSE@20% ↑ | Absolute Regret ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V0: Error Only** | 2 | **+0.3837** | +0.3837 | 0.9462 | 37.5% | **0.5795** | +0.194511 |
| **V1: + Visibility** | 3 | **+0.3985** | +0.0148 | 0.9563 | 50.0% | **0.7309** | +0.124475 |
| **V2: + Influence** | 4 | **+0.3644** | -0.0341 | 0.9506 | 50.0% | **0.6382** | +0.167371 |
| **V3: + Temporal Drift** | 5 | **+0.2038** | -0.1606 | 0.9457 | 37.5% | **0.5695** | +0.199149 |
| **V4: + Uncertainty** | 6 | **+0.3043** | +0.1006 | 0.9508 | 37.5% | **0.5561** | +0.205348 |
| **V5: + Gradient Norm** | 7 | **+0.2447** | -0.0597 | 0.9450 | 37.5% | **0.5445** | +0.210693 |
| **V6: + Projected Area** | 8 | **+0.2493** | +0.0047 | 0.9429 | 37.5% | **0.5045** | +0.229224 |
| **V7: Full State** | 10 | **+0.2366** | -0.0128 | 0.9561 | 25.0% | **0.4723** | +0.244094 |

## 3. Model Architecture & Loss Comparison (Points XXIV & XXV)

| Model Architecture | Loss Objective | Spearman $\rho(U^\star)$ ↑ | NDCG@20% ↑ | OSE@20% ↑ | $\text{MAE}(\Delta Q)$ ↓ | $\text{MAE}(C)$ ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear Two-Head (Regression)** | Decoupled Smooth-L1 | **-0.2897** | 0.9192 | **0.2557** | 0.032215 | 44.35 ms |
| **MLP Two-Head (Regression)** | Decoupled Smooth-L1 | **+0.3418** | 0.9607 | **0.5184** | 0.032706 | 8.49 ms |
| **Linear Two-Head (Ranking)** | Pairwise + Pointwise | **+0.2964** | 0.9584 | **0.6748** | 0.113590 | 45.85 ms |
| **MLP Two-Head (Pairwise+Pointwise Ranking - Ours)** | Pairwise + Pointwise | **+0.0644** | 0.9332 | **0.4734** | 0.069883 | 7.73 ms |

## 4. Geometry Stratum Breakdown (Point XXVII)

| Geometry Stratum | Interventions (N) | Mean Oracle $U^\star$ | $\rho(\text{Error}, U^\star)$ | $\rho(\text{Heuristic}, U^\star)$ | $\rho(\text{Learned Ours}, U^\star)$ ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **flat** | 40 | +0.000026 | +0.0859 | +0.1580 | **+0.3842** |
| **edge** | 40 | +0.000511 | -0.0689 | -0.0135 | **+0.6186** |
| **texture** | 40 | +0.000514 | +0.1390 | +0.1375 | **+0.4139** |
| **depth_discontinuity** | 40 | +0.000383 | +0.3101 | +0.1704 | **+0.4805** |
