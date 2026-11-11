# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 30 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 28.66 ± 0.01 | 30.04 ± 0.01 | 0.7613 | 16.9 | 13,331 |
| `rtg_slam_reimpl` | 27.69 ± 0.01 | 28.20 ± 0.04 | 0.7365 | 9.3 | 13,729 |
| `error_only` | 28.68 ± 0.00 | 30.07 ± 0.00 | 0.7618 | 17.2 | 13,576 |
| `full` | 28.68 ± 0.01 | 30.03 ± 0.08 | 0.7616 | 16.9 | 13,560 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | +0.98 dB | [+0.96, +0.99] | 0.2500 | 0.7500 ❌ | 65.55 | +1.85 dB | 0.7500 | +0.0248 |
| OURS vs `error_only` | -0.01 dB | [-0.03, -0.00] | 0.2500 | 0.7500 ❌ | -0.99 | -0.03 dB | 0.7500 | -0.0005 |
| OURS vs `full` | -0.01 dB | [-0.01, -0.01] | 0.2500 | 0.7500 ❌ | -3.19 | +0.01 dB | 0.7500 | -0.0003 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```