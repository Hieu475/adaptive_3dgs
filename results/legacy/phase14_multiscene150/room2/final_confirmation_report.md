# Final Confirmation Benchmark Report

**Scene:** `replica_room2` | **Frames:** 99 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 22.86 ± 0.08 | 23.59 ± 0.16 | 0.8897 | 8.9 | 77,780 |
| `error_only` | 25.88 ± 0.01 | 26.34 ± 0.12 | 0.9428 | 7.9 | 146,168 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -3.02 dB | [-3.14, -2.94] | 0.2500 | 0.2500 ❌ | -29.00 | -2.75 dB | 0.2500 | -0.0531 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```