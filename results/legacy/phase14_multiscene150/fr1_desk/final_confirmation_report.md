# Final Confirmation Benchmark Report

**Scene:** `tum_fr1_desk` | **Frames:** 150 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 22.55 ± 0.01 | 21.37 ± 0.20 | 0.9211 | 8.1 | 27,394 |
| `error_only` | 22.34 ± 0.02 | 18.69 ± 0.11 | 0.9135 | 7.9 | 55,671 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | +0.21 dB | [+0.17, +0.23] | 0.2500 | 0.2500 ❌ | 6.89 | +2.68 dB | 0.2500 | +0.0075 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```