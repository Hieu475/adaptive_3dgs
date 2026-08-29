# R32 Rigorous Cost Calibration Report

Evaluated with randomized multi-seed protocol across $N=20,000$ Gaussians.

| Metric | Model A (Linear Count) | Model B (Feature-Aware) | Model C (Stage-Level) |
|:---|:---:|:---:|:---:|
| **Formulation** | $T_0 + \beta M$ | $T_0 + \beta_1 M + \beta_2 A + \beta_3 \text{Inf}$ | $\Sigma_s (a_s + b_s M)$ |
| **Fixed Overhead ($T_0$)** | 892.081 ms | 966.241 ms | 892.081 ms |
| **Goodness of Fit ($R^2$)** | **0.1090** | **0.2019** | **0.1090** |
| **MAE** | **2464.535 ms** | **2464.637 ms** | **2464.535 ms** |
| **MAPE** | **927.64%** | **855.85%** | **927.64%** |

