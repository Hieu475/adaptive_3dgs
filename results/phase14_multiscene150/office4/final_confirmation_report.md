# Final Confirmation Benchmark Report

**Scene:** `replica_office4` | **Frames:** 99 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 23.58 ± 0.06 | 20.54 ± 0.05 | 0.9493 | 11.0 | 80,556 |
| `error_only` | 25.93 ± 0.03 | 21.43 ± 0.04 | 0.9671 | 10.1 | 141,344 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -2.35 dB | [-2.40, -2.32] | 0.2500 | 0.2500 ❌ | -49.98 | -0.89 dB | 0.2500 | -0.0178 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```