# Final Confirmation Benchmark Report

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44, 45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=8**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `no_op` | 13.45 ± 0.02 | 11.75 ± 0.04 | 0.4378 | 61.3 | 119,257 |
| `error_only` | 15.54 ± 0.02 | 12.77 ± 0.10 | 0.5708 | 8.0 | 105,845 |
| `error_influence` | 16.37 ± 0.06 | 13.53 ± 0.26 | 0.7379 | 7.8 | 150,000 |
| `error_influence_temporal` | 16.28 ± 0.07 | 13.51 ± 0.19 | 0.7345 | 7.7 | 150,000 |
| `ours` | 19.24 ± 0.05 | 17.64 ± 0.50 | 0.8433 | 8.7 | 39,250 |
| `full` | 21.03 ± 0.04 | 19.31 ± 0.68 | 0.8767 | 5.8 | 150,000 |

## 2. Paired Significance Tests (OURS vs each baseline)

| Comparison | Δ Mean PSNR | 95% CI | p-value | Cohen's d | Δ Final PSNR | p-value | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `no_op` | +5.80 dB | [+5.76, +5.83] | 0.0078 ✅ | 111.39 | +5.89 dB | 0.0078 | +0.4055 |
| OURS vs `error_only` | +3.70 dB | [+3.65, +3.74] | 0.0078 ✅ | 54.98 | +4.87 dB | 0.0078 | +0.2725 |
| OURS vs `error_influence` | +2.87 dB | [+2.83, +2.92] | 0.0078 ✅ | 38.42 | +4.11 dB | 0.0078 | +0.1054 |
| OURS vs `error_influence_temporal` | +2.97 dB | [+2.92, +3.01] | 0.0078 ✅ | 41.87 | +4.13 dB | 0.0078 | +0.1088 |
| OURS vs `full` | -1.78 dB | [-1.83, -1.75] | 0.0078 ✅ | -29.00 | -1.67 dB | 0.0078 | -0.0334 |

## 3. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```