# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 27.69 ± 0.01 | 28.92 ± 0.07 | 0.9230 | 11.1 | 23,007 |
| `rtg_slam_reimpl` | 23.33 ± 0.07 | 20.13 ± 0.91 | 0.8693 | 8.5 | 63,053 |
| `error_only` | 24.17 ± 0.09 | 20.22 ± 0.13 | 0.8867 | 9.3 | 74,445 |
| `error_influence` | 25.09 ± 0.05 | 21.89 ± 0.10 | 0.9083 | 9.1 | 104,918 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `rtg_slam_reimpl` | +4.35 dB | [+4.26, +4.45] | 0.2500 | 0.7500 ❌ | 44.37 | +8.79 dB | 0.7500 | +0.0538 |
| OURS vs `error_only` | +3.52 dB | [+3.38, +3.60] | 0.2500 | 0.7500 ❌ | 28.95 | +8.71 dB | 0.7500 | +0.0363 |
| OURS vs `error_influence` | +2.60 dB | [+2.52, +2.67] | 0.2500 | 0.7500 ❌ | 33.69 | +7.03 dB | 0.7500 | +0.0147 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```