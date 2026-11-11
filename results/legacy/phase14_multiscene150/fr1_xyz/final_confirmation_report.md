# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 23.47 ± 0.00 | 25.38 ± 0.02 | 0.9587 | 8.7 | 38,743 |
| `error_only` | 21.07 ± 0.03 | 22.06 ± 0.15 | 0.9409 | 8.1 | 80,782 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | +2.41 dB | [+2.37, +2.44] | 0.2500 | 0.2500 ❌ | 70.84 | +3.32 dB | 0.2500 | +0.0178 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```