# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_xyz` | **Frames:** 150 | **Seeds:** [45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=5**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 23.47 ± 0.01 | 25.40 ± 0.06 | 0.9587 | 7.4 | 38,743 |
| `error_only` | 21.11 ± 0.03 | 22.01 ± 0.14 | 0.9413 | 6.8 | 80,616 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | +2.36 dB | [+2.34, +2.38] | 0.0625 | 0.0625 ❌ | 84.74 | +3.38 dB | 0.0625 | +0.0174 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```