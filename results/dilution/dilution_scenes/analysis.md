# Scene-level proportional budget and coverage throttling

| scene | frames | n seeds | base PSNR | prop. budget dPSNR [95% CI] | coverage dPSNR [95% CI] |
|---|--:|--:|--:|---|---|
| replica_office2 | 99 | 3 | 24.72 | +2.76 [+2.73, +2.78] | -0.08 [-0.10, -0.03] |
| replica_office3 | 99 | 3 | 24.40 | +3.55 [+3.54, +3.56] | -0.26 [-0.30, -0.21] |
| replica_office4 | 99 | 3 | 25.47 | +3.68 [+3.66, +3.71] | -0.04 [-0.09, +0.01] |
| replica_room0 | 99 | 3 | 22.60 | +2.54 [+2.53, +2.56] | -0.19 [-0.24, -0.16] |
| replica_room1 | 99 | 3 | 24.18 | +2.12 [+2.09, +2.16] | -0.25 [-0.30, -0.18] |
| replica_room2 | 99 | 3 | 25.56 | +3.09 [+3.06, +3.12] | -1.64 [-1.73, -1.52] |
| tum_fr1_desk | 99 | 3 | 23.22 | +0.92 [+0.92, +0.93] | -0.16 [-0.22, -0.07] |
| tum_fr1_room | 99 | 3 | 16.69 | +1.74 [+1.74, +1.75] | -0.13 [-0.13, -0.12] |
| tum_fr1_rpy | 99 | 3 | 21.10 | +2.06 [+2.04, +2.10] | +0.04 [-0.05, +0.09] |
| tum_fr3_sitting_static | 99 | 3 | 26.44 | +0.02 [+0.01, +0.03] | +0.08 [+0.02, +0.15] |
| tum_fr3_structure_texture_near | 99 | 3 | 27.29 | +2.70 [+2.68, +2.72] | +2.54 [+2.53, +2.56] |
| replica_office0 | 150 | 8 | 30.63 | +2.88 [+2.86, +2.91] | -0.68 [-0.70, -0.66] |
| replica_office1 | 150 | 8 | 33.84 | +1.56 [+1.54, +1.59] | -0.87 [-0.90, -0.84] |
| tum_fr2_xyz | 150 | 8 | 24.16 | +3.73 [+3.64, +3.81] | +3.53 [+3.38, +3.70] |
| tum_fr1_xyz | 150 | 8 | 21.54 | +2.99 [+2.96, +3.03] | +1.71 [+1.64, +1.80] |

| contrast | scenes | positive | median | range | sign-test p | Wilcoxon p |
|---|--:|--:|--:|---|--:|--:|
| proportional budget | 15 | 15 | +2.70 | [+0.02, +3.73] | 0.0001 | 0.0001 |
| coverage throttling | 15 | 5 | -0.13 | [-1.64, +3.53] | 0.3018 | 0.5245 |

Spearman(unthrottled final map size, proportional-budget gain) = +0.52 (p=0.048, n=15).