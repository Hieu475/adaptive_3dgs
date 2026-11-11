# Final Confirmation Benchmark Report

**Scene:** `replica_room0` | **Frames:** 99 | **Seeds:** [42, 43, 44] | **Budget:** 15.0 ms | **n=3**

## 1. Policy Performance Summary

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | FPS | N_final |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ours` | 20.85 ± 0.02 | 19.68 ± 0.17 | 0.8964 | 10.4 | 82,158 |
| `error_only` | 22.78 ± 0.01 | 15.93 ± 0.48 | 0.9317 | 9.4 | 150,000 |

## 2. Paired Significance Tests (Family of Comparisons against OURS, Holm-Corrected)

| Comparison | Δ Mean PSNR | 95% CI | Raw p | Holm p | Paired Cohen's $d_z$ | Δ Final PSNR | Final Holm p | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OURS vs `error_only` | -1.93 dB | [-1.96, -1.89] | 0.2500 | 0.2500 ❌ | -54.61 | +3.75 dB | 0.2500 | -0.0353 |

## 4. OURS Definition

```
OURS = Error-Influence-Temporal selection + Coverage Throttling + Backlog Throttling
       NO warmup (ablation: -0.75 dB)
       NO adaptive-K (ablation: -1.54 dB with learned utility)
       NO learned utility predictor
       Fixed K=5 micro-steps
```