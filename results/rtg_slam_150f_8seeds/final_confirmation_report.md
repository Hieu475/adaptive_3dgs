# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44, 45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=8**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 27.67 ± 0.04 | 28.89 ± 0.11 | 0.9232 | 10.7 | 23,397 |
| `rtg_slam_reimpl` | 23.05 ± 0.26 | 20.04 ± 0.88 | 0.8663 | 8.0 | 67,230 |
| `error_only` | 24.07 ± 0.14 | 20.01 ± 0.37 | 0.8850 | 8.9 | 74,640 |
| `error_influence` | 25.04 ± 0.06 | 21.89 ± 0.09 | 0.9084 | 8.8 | 106,928 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | +4.62 dB | [+4.44, +4.82] | 0.0078 | 0.0234 ✅ | 15.45 | +8.85 dB | 0.0234 | +0.0568 |
| OURS vs `error_only` | +3.59 dB | [+3.48, +3.70] | 0.0078 | 0.0234 ✅ | 21.19 | +8.88 dB | 0.0234 | +0.0381 |
| OURS vs `error_influence` | +2.63 dB | [+2.56, +2.68] | 0.0078 | 0.0234 ✅ | 29.79 | +6.99 dB | 0.0234 | +0.0148 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```