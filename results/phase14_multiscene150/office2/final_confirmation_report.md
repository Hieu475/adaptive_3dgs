# Final Confirmation Benchmark Report

**Scene:** `replica_office2` | **Frames:** 99 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 22.08 ± 0.01 | 20.81 ± 0.05 | 0.9471 | 10.9 | 70,105 |
| `error_only` | 24.66 ± 0.03 | 24.82 ± 0.13 | 0.9684 | 10.0 | 117,279 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -2.59 dB | [-2.61, -2.53] | 0.2500 | 0.2500 ❌ | -51.01 | -4.01 dB | 0.2500 | -0.0213 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```