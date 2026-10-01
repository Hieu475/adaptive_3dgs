# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_rpy` | **Frames:** 100 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 19.92 ± 0.03 | 24.47 ± 0.08 | 0.8725 | 8.4 | 45,454 |
| `error_only` | 21.06 ± 0.00 | 24.10 ± 0.02 | 0.8988 | 8.2 | 62,002 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -1.13 dB | [-1.16, -1.11] | 0.2500 | 0.2500 ❌ | -43.55 | +0.36 dB | 0.2500 | -0.0263 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```