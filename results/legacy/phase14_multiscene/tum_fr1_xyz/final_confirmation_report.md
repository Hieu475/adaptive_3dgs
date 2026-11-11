# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_xyz` | **Frames:** 30 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 23.45 ± 0.01 | 26.53 ± 0.02 | 0.9701 | 11.8 | 16,545 |
| `rtg_slam_reimpl` | 21.84 ± 0.00 | 23.87 ± 0.01 | 0.9627 | 8.9 | 22,430 |
| `error_only` | 23.42 ± 0.01 | 26.17 ± 0.04 | 0.9705 | 9.5 | 22,367 |
| `full` | 23.56 ± 0.01 | 26.63 ± 0.03 | 0.9711 | 15.0 | 22,142 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | +1.61 dB | [+1.61, +1.62] | 0.2500 | 0.7500 ❌ | 275.43 | +2.66 dB | 0.7500 | +0.0074 |
| OURS vs `error_only` | +0.03 dB | [+0.02, +0.05] | 0.2500 | 0.7500 ❌ | 1.89 | +0.36 dB | 0.7500 | -0.0004 |
| OURS vs `full` | -0.11 dB | [-0.13, -0.09] | 0.2500 | 0.7500 ❌ | -5.47 | -0.10 dB | 0.7500 | -0.0010 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```