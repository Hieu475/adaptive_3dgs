# Factorial Ablation Study: Causal Component Isolation

**Scene:** `tum_fr1_room` | **Frames:** 50 | **Seeds:** [42] | **Budget:** 15.0 ms

## 1. Full 2×3 Factorial Matrix

| ID | Selection Policy | Throttling | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Map Size ($N$) | FPS |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | `error_only` | **OFF** | 18.31 ± 0.00 | 15.54 ± 0.00 | 0.7039 | 55,934 | 8.7 |
| **B1** | `error_influence` | **OFF** | 17.52 ± 0.00 | 16.54 ± 0.00 | 0.6917 | 57,665 | 8.9 |
| **B2** | `learned_utility` | **OFF** | 14.88 ± 0.00 | 13.59 ± 0.00 | 0.6160 | 58,378 | 16.9 |
| **B3** | `error_only` | **ON** | 18.12 ± 0.00 | 14.99 ± 0.00 | 0.7012 | 49,131 | 9.0 |
| **B4** | `error_influence` | **ON** | 17.34 ± 0.00 | 16.17 ± 0.00 | 0.6893 | 51,666 | 9.0 |
| **B5** | `learned_utility` | **ON** | 14.83 ± 0.00 | 13.86 ± 0.00 | 0.6170 | 53,199 | 16.9 |

## 2. Causal Effect Decomposition

| Research Question | Factorial Contrast | Empirical Effect ($\\Delta$ Mean PSNR) | Scientific Interpretation |
|:---|:---:|:---:|:---|
| **Does Throttling rescue raw Error?** | $B_3 - B_0$ | **-0.18 dB** | Pure compute conservation effect |
| **Does Throttling rescue Error×Influence?** | $B_4 - B_1$ | **-0.18 dB** | Impact of map containment on sensitivity ranking |
| **Does Learned Utility beat Heuristic (No Throttle)?** | $B_2 - B_1$ | **-2.64 dB** | Intrinsic value of learned utility without throttling |
| **Does Learned Utility add value with Throttling?** | $B_5 - B_4$ | **-2.51 dB** | Marginal gain of learned utility in controlled regime |
| **Interaction (Super-additivity / Synergy)** | $(B_5 - B_4) - (B_2 - B_1)$ | **+0.14 dB** | Non-linear interaction between model & throttling |

## 3. Core Scientific Conclusion

> [!NOTE]
> **Observed Gains:** Throttling effect = -0.18 dB, Learned utility marginal gain = -2.51 dB.