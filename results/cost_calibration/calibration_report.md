# R32 Rigorous Cost Calibration Report

Evaluated with randomized multi-seed protocol across $N=20,000$ Gaussians.

| Metric | Model A (Linear Count) | Model B (Feature-Aware) |
|:---|:---:|:---:|
| **Formulation** | $T_0 + \beta M$ | $T_0 + \beta_1 M + \beta_2 A + \beta_3 \text{Inf}$ |
| **Fixed Overhead ($T_0$)** | 161.231 ms | 162.053 ms |
| **Goodness of Fit ($R^2$)** | **0.5918** | **0.6291** |
| **MAE** | **41.084 ms** | **38.229 ms** |
| **MAPE** | **19.06%** | **17.92%** |

