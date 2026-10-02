# Final Confirmation Benchmark Report

**Scene:** `replica_office3` | **Frames:** 99 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 22.45 ± 0.01 | 22.56 ± 0.02 | 0.9344 | 10.8 | 76,684 |
| `error_only` | 24.47 ± 0.02 | 23.19 ± 0.03 | 0.9602 | 9.6 | 149,991 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -2.02 dB | [-2.05, -1.98] | 0.2500 | 0.2500 ❌ | -53.70 | -0.62 dB | 0.2500 | -0.0258 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```