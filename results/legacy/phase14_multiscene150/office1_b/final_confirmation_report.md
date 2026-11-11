# Final Confirmation Benchmark Report

**Scene:** `replica_office1` | **Frames:** 100 | **Seeds:** [45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=5**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 29.22 ± 0.07 | 28.89 ± 0.09 | 0.9346 | 8.0 | 41,352 |
| `error_only` | 33.45 ± 0.02 | 35.09 ± 0.06 | 0.9730 | 7.7 | 74,067 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -4.23 dB | [-4.29, -4.17] | 0.0625 | 0.0625 ❌ | -55.03 | -6.20 dB | 0.0625 | -0.0384 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```