# Final Confirmation Benchmark Report

**Scene:** `replica_room1` | **Frames:** 100 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 20.42 ± 0.01 | 21.58 ± 0.08 | 0.8342 | 11.3 | 46,990 |
| `error_only` | 24.28 ± 0.01 | 24.55 ± 0.03 | 0.9011 | 10.3 | 59,292 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -3.86 dB | [-3.88, -3.84] | 0.2500 | 0.2500 ❌ | -171.88 | -2.97 dB | 0.2500 | -0.0669 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```