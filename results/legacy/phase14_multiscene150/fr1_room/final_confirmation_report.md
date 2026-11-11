# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_room` | **Frames:** 100 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 15.53 ± 0.03 | 12.16 ± 0.01 | 0.6264 | 8.0 | 91,791 |
| `error_only` | 16.54 ± 0.02 | 12.38 ± 0.07 | 0.6538 | 7.9 | 96,550 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -1.01 dB | [-1.03, -0.99] | 0.2500 | 0.2500 ❌ | -53.88 | -0.22 dB | 0.2500 | -0.0274 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```