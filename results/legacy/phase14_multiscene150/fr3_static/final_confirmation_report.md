# Final Confirmation Benchmark Report

**Scene:** `tum_fr3_sitting_static` | **Frames:** 100 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 25.96 ± 0.03 | 30.67 ± 0.02 | 0.6684 | 17.4 | 13,452 |
| `error_only` | 25.88 ± 0.01 | 29.32 ± 0.27 | 0.6718 | 11.5 | 18,060 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | +0.08 dB | [+0.04, +0.11] | 0.2500 | 0.2500 ❌ | 2.24 | +1.35 dB | 0.2500 | -0.0034 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```