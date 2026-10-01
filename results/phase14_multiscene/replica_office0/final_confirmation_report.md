# Final Confirmation Benchmark Report

**Scene:** `replica_office0` | **Frames:** 30 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 23.99 ± 0.01 | 28.99 ± 0.07 | 0.8838 | 10.0 | 51,879 |
| `rtg_slam_reimpl` | 29.49 ± 0.05 | 37.89 ± 0.09 | 0.9400 | 10.0 | 55,272 |
| `error_only` | 29.48 ± 0.01 | 37.88 ± 0.10 | 0.9397 | 9.8 | 55,268 |
| `full` | 30.30 ± 0.04 | 38.41 ± 0.03 | 0.9467 | 9.2 | 55,233 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | -5.49 dB | [-5.54, -5.42] | 0.2500 | 0.7500 ❌ | -84.45 | -8.90 dB | 0.7500 | -0.0562 |
| OURS vs `error_only` | -5.48 dB | [-5.50, -5.46] | 0.2500 | 0.7500 ❌ | -296.43 | -8.89 dB | 0.7500 | -0.0559 |
| OURS vs `full` | -6.31 dB | [-6.34, -6.25] | 0.2500 | 0.7500 ❌ | -116.05 | -9.42 dB | 0.7500 | -0.0629 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```