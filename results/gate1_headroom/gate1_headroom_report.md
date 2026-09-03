# Gate 1 & Headroom Verification Report

## 1. Gate 1: Measurability & Variance of Marginal Utility

- **Status:** PASSED ✅
- **Sample Size:** 60 visible interventions
- **Utility Variance:** $\text{Var}(U^\star) = 0.000000$
- **Utility Range:** $[-0.0003, +0.0017]$ (Mean: $+0.0003 \pm 0.0004$)
- **Negative Utility Proportion:** 8.3% (5/60 candidates degraded quality upon intervention)

## 2. Optimization Headroom ($H$) & Policy Selection at $K = 12$

- **Headroom $H$ (Joint Gain):** **+0.000149** (Strictly Positive ✅)
- **Headroom $\Delta$PSNR:** **+0.0014 dB**

| Policy | Realized $\Delta Q$ | Realized $\Delta$PSNR | Oracle Selection Efficiency ($OSE$) ↑ | Selection Regret ($R_K$) ↓ |
|:---|:---:|:---:|:---:|:---:|
| **Oracle Upper Bound ($S^\star_K$)** | **+0.000338** | **+0.0027 dB** | **1.000** | **0.0000** |
| **Heuristic (Pre-fusion Norm)** | +0.000179 | +0.0011 dB | **0.530** | +0.000159 |
| **Error-Only Top-$K$** | +0.000050 | -0.0002 dB | 0.148 | +0.000288 |
| **Random Baseline** | +0.000189 | +0.0013 dB | 0.558 | +0.000149 |

## 3. Empirical Group Additivity ($R_{add}$)

- **Group Size $g=4$:** $R_{add} = 0.2249$
- **Group Size $g=16$:** $R_{add} = 0.0048$
- *Interpretation:* $R_{add} < 1.0$ quantitatively confirms diminishing marginal returns / occlusion overlap in concurrent Gaussian optimization.
