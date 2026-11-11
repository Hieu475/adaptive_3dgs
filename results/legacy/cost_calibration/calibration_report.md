# R32 Rigorous Cost Calibration Report

Evaluated with isolated pure optimization timing across $N=25,000$ Gaussians (80% Train, 20% Held-Out Test).

| Metric | Model A (Linear Count) | Model B (Workload-Aware) | Model C (Stage-Level Sum) |
|:---|:---:|:---:|:---:|
| **Formulation** | $T_0 + \beta M$ | $T_0 + \beta_1 M + \beta_2 A + \beta_3 P$ | $T_{rend}(M) + T_{bwd}(M) + T_{opt}(M)$ |
| **Fixed Overhead ($T_0$)** | 34.387 ms | 32.746 ms | 34.387 ms |
| **Out-of-Sample $R^2$** | **0.8990** | **0.8909** | **0.8990** |
| **Test MAE (ms)** | **21.048 ms** | **22.570 ms** | **21.048 ms** |
| **Test RMSE (ms)** | **25.504 ms** | **26.500 ms** | **25.504 ms** |
| **Test sMAPE (%)** | **37.50%** | **36.67%** | **37.50%** |

