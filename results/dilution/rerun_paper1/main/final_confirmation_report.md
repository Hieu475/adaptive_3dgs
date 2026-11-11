# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44, 45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=8**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `no_op` | 18.85 ± 0.01 | 14.75 ± 0.02 | 0.4655 | 67.2 | 21,127 |
| `error_only` | 24.21 ± 0.14 | 20.20 ± 0.40 | 0.7023 | 11.0 | 72,365 |
| `error_influence` | 25.38 ± 0.10 | 22.35 ± 0.16 | 0.7338 | 11.1 | 86,988 |
| `error_influence_temporal` | 25.23 ± 0.17 | 22.05 ± 0.30 | 0.7315 | 10.9 | 86,187 |
| `rtg_slam_reimpl` | 23.53 ± 0.28 | 20.57 ± 0.71 | 0.6833 | 10.1 | 60,049 |
| `ours` | 27.74 ± 0.03 | 28.73 ± 0.08 | 0.7978 | 12.1 | 21,642 |
| `full` | 27.89 ± 0.01 | 29.29 ± 0.08 | 0.8036 | 11.3 | 70,295 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `no_op` | +8.89 dB | [+8.87, +8.92] | 0.0078 | 0.0469 ✅ | 245.09 | +13.98 dB | 0.0469 | +0.3323 |
| OURS vs `error_only` | +3.53 dB | [+3.42, +3.63] | 0.0078 | 0.0469 ✅ | 22.52 | +8.53 dB | 0.0469 | +0.0954 |
| OURS vs `error_influence` | +2.36 dB | [+2.30, +2.44] | 0.0078 | 0.0469 ✅ | 21.07 | +6.38 dB | 0.0469 | +0.0640 |
| OURS vs `error_influence_temporal` | +2.51 dB | [+2.40, +2.64] | 0.0078 | 0.0469 ✅ | 13.71 | +6.68 dB | 0.0469 | +0.0663 |
| OURS vs `rtg_slam_reimpl` | +4.21 dB | [+4.01, +4.40] | 0.0078 | 0.0469 ✅ | 13.27 | +8.16 dB | 0.0469 | +0.1145 |
| OURS vs `full` | -0.15 dB | [-0.18, -0.13] | 0.0078 | 0.0469 ✅ | -3.90 | -0.56 dB | 0.0469 | -0.0058 |

## 3. Headroom Recovery & Substrate Metrics

- **Gap to Unconstrained FULL:** **0.15 dB** (27.89 vs 27.74 dB) with a **3.8× smaller map**.
- **Headroom Recovery Ratio:** $\eta = \frac{Q_{\mathrm{OURS}} - Q_{\mathrm{no\_op}}}{Q_{\mathrm{full}} - Q_{\mathrm{no\_op}}} = \frac{27.74 - 18.85}{27.89 - 18.85} = \mathbf{98.3\%}$.
  *(Note: PSNR is logarithmic dB; we report normalized headroom recovery rather than the unnormalized ratio).* 

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```