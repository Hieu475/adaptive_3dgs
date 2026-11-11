# Final Confirmation Benchmark Report

**Scene:** `replica_office0` | **Frames:** 150 | **Seeds:** [45, 46, 47, 48, 49] | **Budget:** 15.0 ms | **n=5**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 26.84 ± 0.04 | 26.05 ± 0.05 | 0.9244 | 10.7 | 78,840 |
| `error_only` | 30.16 ± 0.02 | 27.98 ± 0.12 | 0.9583 | 10.0 | 116,080 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -3.31 dB | [-3.36, -3.27] | 0.0625 | 0.0625 ❌ | -55.94 | -1.93 dB | 0.0625 | -0.0339 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```