# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44, 45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=8**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `no_op` | 19.06 ± 0.01 | 14.88 ± 0.01 | 0.6920 | 90.0 | 23,523 |
| `error_only` | 24.04 ± 0.17 | 20.12 ± 0.28 | 0.8838 | 9.8 | 73,382 |
| `error_influence` | 25.03 ± 0.08 | 21.83 ± 0.14 | 0.9083 | 9.5 | 105,438 |
| `error_influence_temporal` | 24.94 ± 0.07 | 21.61 ± 0.14 | 0.9079 | 9.5 | 106,391 |
| `ours` | 27.64 ± 0.04 | 28.82 ± 0.09 | 0.9231 | 11.1 | 23,764 |
| `full` | 27.85 ± 0.01 | 29.34 ± 0.08 | 0.9245 | 9.6 | 90,581 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `no_op` | +8.58 dB | [+8.54, +8.62] | 0.0078 | 0.0391 ✅ | 154.45 | +13.94 dB | 0.0391 | +0.2311 |
| OURS vs `error_only` | +3.59 dB | [+3.48, +3.73] | 0.0078 | 0.0391 ✅ | 18.40 | +8.70 dB | 0.0391 | +0.0393 |
| OURS vs `error_influence` | +2.61 dB | [+2.53, +2.68] | 0.0078 | 0.0391 ✅ | 22.48 | +6.99 dB | 0.0391 | +0.0148 |
| OURS vs `error_influence_temporal` | +2.70 dB | [+2.64, +2.76] | 0.0078 | 0.0391 ✅ | 27.97 | +7.21 dB | 0.0391 | +0.0152 |
| OURS vs `full` | -0.21 dB | [-0.24, -0.18] | 0.0078 | 0.0391 ✅ | -4.22 | -0.52 dB | 0.0391 | -0.0014 |

## 3. Headroom Recovery & Substrate Metrics

- **Gap to Unconstrained FULL:** **0.21 dB** (27.85 vs 27.64 dB) with a **3.8× smaller map**.
- **Headroom Recovery Ratio:** $\eta = \frac{Q_{\mathrm{OURS}} - Q_{\mathrm{no\_op}}}{Q_{\mathrm{full}} - Q_{\mathrm{no\_op}}} = \frac{27.64 - 19.06}{27.85 - 19.06} = \mathbf{97.6\%}$.
  *(Note: PSNR is logarithmic dB; we report normalized headroom recovery rather than the unnormalized ratio).* 

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```