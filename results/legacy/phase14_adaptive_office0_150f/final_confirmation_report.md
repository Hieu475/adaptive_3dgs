# Final Confirmation Benchmark Report

**Scene:** `replica_office0` | **Frames:** 150 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 26.86 ± 0.02 | 26.12 ± 0.05 | 0.9245 | 10.7 | 78,799 |
| `error_only` | 30.18 ± 0.01 | 28.10 ± 0.06 | 0.9585 | 10.0 | 115,769 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -3.33 dB | [-3.34, -3.31] | 0.2500 | 0.2500 ❌ | -204.59 | -1.99 dB | 0.2500 | -0.0340 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```