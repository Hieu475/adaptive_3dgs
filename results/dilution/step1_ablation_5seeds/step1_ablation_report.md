# Stage 1 Component Ablation Study Report (5 Seeds x 150 Frames)

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** [42, 43, 44, 45, 46] | **Budget:** 15.0 ms

## 1. Condition Aggregate Performance

| Condition | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Mean K | Final Map Size | FPS |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `A0_old_substrate` | 16.07 ± 0.53 | 12.94 ± 1.55 | 0.6969 ± 0.0579 | 5.00 | 140,997 | 7.4 |
| `A1_plus_warmup` | 15.32 ± 0.50 | 12.76 ± 0.68 | 0.6344 ± 0.0533 | 5.00 | 140,038 | 7.8 |
| `A2_plus_coverage_throttling` | 17.11 ± 0.35 | 14.31 ± 0.27 | 0.7699 ± 0.0077 | 5.00 | 44,864 | 8.7 |
| `A3_plus_backlog_throttling` | 17.15 ± 0.27 | 14.68 ± 0.28 | 0.7713 ± 0.0075 | 5.00 | 43,824 | 8.8 |
| `A4_all` | 14.53 ± 1.32 | 12.97 ± 1.35 | 0.5652 ± 0.1343 | 2.00 | 38,317 | 12.7 |

## 2. Statistical Significance Tests vs A0 Baseline (Two-Sided Wilcoxon + Holm Multiplicity Correction)

| Comparison | Δ Mean PSNR [95% CI] | p_Holm | Cohen's d | Δ Final PSNR [95% CI] | p_Holm | Cohen's d | Δ SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A1_plus_warmup vs A0** | -0.75 dB [-1.14, -0.37] | 0.2500 | -1.55 | -0.18 dB [-1.19, +0.94] | 1.0000 | -0.13 | -0.0625 |
| **A2_plus_coverage_throttling vs A0** | +1.04 dB [+0.47, +1.59] | 0.2500 | 1.49 | +1.37 dB [+0.43, +2.62] | 0.2500 | 0.97 | +0.0730 |
| **A3_plus_backlog_throttling vs A0** | +1.08 dB [+0.55, +1.56] | 0.2500 | 1.72 | +1.75 dB [+0.49, +3.25] | 0.3750 | 1.03 | +0.0744 |
| **A4_all vs A0** | -1.54 dB [-2.78, -0.31] | 0.2500 | -1.00 | +0.03 dB [-1.01, +0.83] | 1.0000 | 0.02 | -0.1316 |