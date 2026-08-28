# R32 Rigorous Cost Calibration Report

Evaluated with randomized multi-seed protocol across $N=20,000$ Gaussians.

| Metric | Model A (Linear Count) | Model B (Feature-Aware) |
|:---|:---:|:---:|
| **Formulation** | $T_0 + \beta M$ | $T_0 + \beta_1 M + \beta_2 A + \beta_3 \text{Inf}$ |
| **Fixed Overhead ($T_0$)** | 59.628 ms | 59.298 ms |
| **Goodness of Fit ($R^2$)** | **0.9051** | **0.9157** |
| **MAE** | **29.110 ms** | **26.719 ms** |
| **MAPE** | **39.75%** | **38.34%** |

