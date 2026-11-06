# Coverage-throttling decomposition

## replica_office0 (n=4)

Baseline (error_off) PSNR 30.62

| condition | dPSNR [95% CI] | dDepth L1 | map (k) | spawn/frame |
|---|---|--:|--:|--:|
| error_coverage | -0.67 [-0.69, -0.66] | +0.0154 | 65.2 | 465 |
| cov_spawn_only | -0.67 [-0.69, -0.65] | +0.0153 | 65.3 | 466 |
| cov_thresh_only | +0.02 [+0.00, +0.03] | +0.0001 | 112.0 | 937 |
| spawn_cap_always20 | -0.72 [-0.77, -0.69] | +0.0188 | 66.0 | 473 |
| spawn_cap_always50 | -0.11 [-0.14, -0.08] | +0.0023 | 85.4 | 669 |
| error_budget_prop | +2.89 [+2.88, +2.90] | -0.0075 | 107.6 | 893 |
| cov_plus_prop | +1.32 [+1.30, +1.33] | +0.0067 | 63.7 | 449 |

## replica_office1 (n=4)

Baseline (error_off) PSNR 33.87

| condition | dPSNR [95% CI] | dDepth L1 | map (k) | spawn/frame |
|---|---|--:|--:|--:|
| error_coverage | -0.88 [-0.90, -0.84] | +0.0112 | 39.9 | 457 |
| cov_spawn_only | -0.91 [-0.93, -0.88] | +0.0110 | 40.4 | 462 |
| cov_thresh_only | -0.01 [-0.05, +0.02] | +0.0001 | 83.1 | 929 |
| spawn_cap_always20 | -1.14 [-1.19, -1.06] | +0.0176 | 40.9 | 464 |
| spawn_cap_always50 | -0.11 [-0.12, -0.10] | +0.0015 | 57.8 | 646 |
| error_budget_prop | +1.54 [+1.52, +1.56] | -0.0053 | 82.7 | 928 |
| cov_plus_prop | -0.08 [-0.12, -0.03] | +0.0076 | 39.0 | 449 |

## tum_fr1_xyz (n=4)

Baseline (error_off) PSNR 21.57

| condition | dPSNR [95% CI] | dDepth L1 | map (k) | spawn/frame |
|---|---|--:|--:|--:|
| error_coverage | +1.70 [+1.64, +1.75] | -0.0043 | 37.0 | 174 |
| cov_spawn_only | +1.55 [+1.50, +1.60] | -0.0040 | 38.6 | 184 |
| cov_thresh_only | +0.18 [+0.10, +0.25] | -0.0003 | 68.3 | 384 |
| spawn_cap_always20 | +1.54 [+1.50, +1.59] | -0.0042 | 38.5 | 183 |
| spawn_cap_always50 | +0.54 [+0.46, +0.58] | -0.0018 | 59.5 | 325 |
| error_budget_prop | +2.96 [+2.92, +3.00] | -0.0070 | 79.1 | 456 |
| cov_plus_prop | +2.82 [+2.80, +2.85] | -0.0071 | 37.8 | 179 |

## tum_fr2_xyz (n=4)

Baseline (error_off) PSNR 24.21

| condition | dPSNR [95% CI] | dDepth L1 | map (k) | spawn/frame |
|---|---|--:|--:|--:|
| error_coverage | +3.36 [+3.28, +3.43] | -0.0086 | 22.2 | 68 |
| cov_spawn_only | +3.25 [+3.21, +3.31] | -0.0084 | 24.2 | 81 |
| cov_thresh_only | +1.12 [+0.98, +1.27] | -0.0004 | 63.2 | 341 |
| spawn_cap_always20 | +3.30 [+3.21, +3.40] | -0.0087 | 23.7 | 78 |
| spawn_cap_always50 | +1.56 [+1.40, +1.71] | -0.0027 | 47.6 | 237 |
| error_budget_prop | +3.69 [+3.65, +3.73] | -0.0102 | 70.0 | 387 |
| cov_plus_prop | +3.67 [+3.63, +3.74] | -0.0099 | 20.9 | 59 |
