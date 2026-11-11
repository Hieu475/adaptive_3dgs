# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 10 | **Seeds:** [42] | **Budget:** 15.0 ms | **n=1**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 28.69 ± 0.00 | 29.98 ± 0.00 | 0.7527 | 16.4 | 12,819 |
| `rtg_slam_reimpl` | 28.07 ± 0.00 | 28.79 ± 0.00 | 0.7276 | 9.2 | 12,991 |
| `error_only` | 28.69 ± 0.00 | 29.83 ± 0.00 | 0.7532 | 18.7 | 12,901 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | +0.62 dB | [+0.62, +0.62] | 1.0000 | 1.0000 ❌ | 0.00 | +1.19 dB | 1.0000 | +0.0251 |
| OURS vs `error_only` | +0.00 dB | [+0.00, +0.00] | 1.0000 | 1.0000 ❌ | 0.00 | +0.15 dB | 1.0000 | -0.0005 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```