# Final Confirmation Benchmark Report

**Scene:** `replica_office1` | **Frames:** 100 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 29.29 ± 0.06 | 29.01 ± 0.08 | 0.9353 | 10.9 | 41,759 |
| `error_only` | 33.47 ± 0.03 | 35.09 ± 0.08 | 0.9733 | 10.7 | 73,891 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -4.18 dB | [-4.24, -4.13] | 0.2500 | 0.2500 ❌ | -73.03 | -6.08 dB | 0.2500 | -0.0380 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```