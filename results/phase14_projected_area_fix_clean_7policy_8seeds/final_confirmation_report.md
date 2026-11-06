# Clean Phase-14 confirmation: seven policies

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** 42–49 | **Budget:** 15 ms modeled optimization budget | **Backend:** gsplat | **Pose:** ground-truth W2C

The six-policy run and the RTG-SLAM reimplementation run were executed separately from the same clean source commit. Seeds are the replicate unit.

| Policy | Mean PSNR | Final PSNR | Mean SSIM | Mean depth L1 | FPS | Final map N |
|:--|--:|--:|--:|--:|--:|--:|
| no_op | 19.06 ± 0.01 | 14.88 ± 0.01 | 0.6920 | 0.03496 | 90.0 | 23,523 |
| error_only | 24.04 ± 0.17 | 20.12 ± 0.28 | 0.8838 | 0.02221 | 9.8 | 73,383 |
| error_influence | 25.03 ± 0.08 | 21.83 ± 0.14 | 0.9083 | 0.01531 | 9.5 | 105,438 |
| error_influence_temporal | 24.94 ± 0.07 | 21.61 ± 0.14 | 0.9079 | 0.01513 | 9.5 | 106,392 |
| rtg_slam_reimpl | 23.16 ± 0.24 | 20.04 ± 1.07 | 0.8669 | 0.02622 | 8.9 | 65,960 |
| ours | 27.64 ± 0.04 | 28.82 ± 0.09 | 0.9231 | 0.01129 | 11.1 | 23,765 |
| full | 27.85 ± 0.01 | 29.34 ± 0.08 | 0.9245 | 0.01107 | 9.6 | 90,582 |

## Paired contrasts with OURS

Wilcoxon tests use eight per-seed trajectory summaries. Holm correction is applied across the six baseline contrasts; bootstrap intervals use the harness implementation.

| Baseline | Δ mean PSNR | 95% CI | Raw p | Holm p | Δ final PSNR | Holm p final |
|:--|--:|--:|--:|--:|--:|--:|
| no_op | +8.58 dB | [+8.54, +8.62] | 0.0078 | 0.0469 | +13.94 dB | 0.0469 |
| error_only | +3.59 dB | [+3.48, +3.73] | 0.0078 | 0.0469 | +8.70 dB | 0.0469 |
| error_influence | +2.61 dB | [+2.53, +2.68] | 0.0078 | 0.0469 | +6.99 dB | 0.0469 |
| error_influence_temporal | +2.70 dB | [+2.64, +2.76] | 0.0078 | 0.0469 | +7.21 dB | 0.0469 |
| rtg_slam_reimpl | +4.48 dB | [+4.31, +4.64] | 0.0078 | 0.0469 | +8.79 dB | 0.0469 |
| full | -0.21 dB | [-0.24, -0.18] | 0.0078 | 0.0469 | -0.52 dB | 0.0469 |

## System interpretation

Headroom recovery above no-op is 97.6%; OURS is 0.21 dB below FULL with 3.81× fewer Gaussians.
OURS mean end-to-end frame time is 90.1 ms. The 15 ms value is only the modeled optimization budget, so this run does not establish real-time operation.

## Protocol notes

Ground-truth poses isolate mapping; no ATE/RPE claim is made. This bundle combines two runs from the same source commit and records both raw run sets; the complete merged table is generated from per-seed outputs without treating frames as independent replicates.
