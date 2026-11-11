# Dilution extension

## replica_office0

| condition | n | PSNR | depth L1 | map | mature cov. | young share | budget scale |
|---|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 33.44 | 0.0326 | 107461 | 0.723 | 0.193 | 1.00 |
| error_budget_prop | 8 | 33.51 | 0.0327 | 107487 | 0.744 | 0.179 | 3.67 |
| error_coverage | 8 | 29.95 | 0.0556 | 65167 | 0.265 | 0.245 | 1.00 |
| error_coverage_backlog | 8 | 27.40 | 0.1720 | 45728 | 0.355 | 0.225 | 1.00 |
| error_off | 8 | 30.63 | 0.0403 | 112703 | 0.170 | 0.302 | 1.00 |
| temporal_coverage | 8 | 28.14 | 0.0651 | 66343 | 0.273 | 0.208 | 1.00 |
| temporal_coverage_backlog | 8 | 26.18 | 0.2385 | 45773 | 0.379 | 0.206 | 1.00 |
| temporal_off | 8 | 28.68 | 0.0489 | 115777 | 0.188 | 0.221 | 1.00 |

| contrast | dPSNR (dB) | 95% CI | p | dDepth L1 | n |
|---|--:|---|--:|--:|--:|
| coverage on error | -0.68 | [-0.70, -0.66] | 0.0078 | +0.0153 | 8 |
| coverage on temporal | -0.54 | [-0.55, -0.52] | 0.0078 | +0.0162 | 8 |
| backlog on error | -2.55 | [-2.56, -2.54] | 0.0078 | +0.1164 | 8 |
| backlog on temporal | -1.96 | [-1.98, -1.93] | 0.0078 | +0.1734 | 8 |
| selector without throttle | -1.95 | [-1.97, -1.93] | 0.0078 | +0.0086 | 8 |
| selector with coverage | -1.81 | [-1.83, -1.79] | 0.0078 | +0.0095 | 8 |
| error budget x4 | +2.81 | [+2.79, +2.82] | 0.0078 | -0.0077 | 8 |
| error budget proportional to map | +2.88 | [+2.86, +2.91] | 0.0078 | -0.0076 | 8 |
| proportional vs coverage (error) | +3.56 | [+3.54, +3.59] | 0.0078 | -0.0229 | 8 |

## replica_office1

| condition | n | PSNR | depth L1 | map | mature cov. | young share | budget scale |
|---|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 35.41 | 0.0106 | 82816 | 0.684 | 0.192 | 1.00 |
| error_budget_prop | 8 | 35.41 | 0.0108 | 82725 | 0.743 | 0.172 | 3.87 |
| error_coverage | 8 | 32.97 | 0.0273 | 39910 | 0.281 | 0.259 | 1.00 |
| error_coverage_backlog | 8 | 30.93 | 0.0744 | 25863 | 0.348 | 0.246 | 1.00 |
| error_off | 8 | 33.84 | 0.0161 | 84227 | 0.177 | 0.232 | 1.00 |
| temporal_coverage | 8 | 31.73 | 0.0329 | 42014 | 0.299 | 0.190 | 1.00 |
| temporal_coverage_backlog | 8 | 29.79 | 0.1093 | 24147 | 0.397 | 0.195 | 1.00 |
| temporal_off | 8 | 31.51 | 0.0202 | 85758 | 0.186 | 0.193 | 1.00 |

| contrast | dPSNR (dB) | 95% CI | p | dDepth L1 | n |
|---|--:|---|--:|--:|--:|
| coverage on error | -0.87 | [-0.90, -0.84] | 0.0078 | +0.0112 | 8 |
| coverage on temporal | +0.21 | [+0.17, +0.26] | 0.0078 | +0.0127 | 8 |
| backlog on error | -2.04 | [-2.07, -2.01] | 0.0078 | +0.0471 | 8 |
| backlog on temporal | -1.93 | [-1.97, -1.89] | 0.0078 | +0.0764 | 8 |
| selector without throttle | -2.33 | [-2.36, -2.30] | 0.0078 | +0.0041 | 8 |
| selector with coverage | -1.25 | [-1.28, -1.21] | 0.0078 | +0.0056 | 8 |
| error budget x4 | +1.56 | [+1.54, +1.59] | 0.0078 | -0.0055 | 8 |
| error budget proportional to map | +1.56 | [+1.54, +1.59] | 0.0078 | -0.0054 | 8 |
| proportional vs coverage (error) | +2.44 | [+2.42, +2.45] | 0.0078 | -0.0165 | 8 |

## tum_fr1_xyz

| condition | n | PSNR | depth L1 | map | mature cov. | young share | budget scale |
|---|--:|--:|--:|--:|--:|--:|--:|
| error_budget_prop | 8 | 24.54 | 0.0307 | 79014 | 1.000 | 0.112 | 3.68 |
| error_off | 8 | 21.54 | 0.0378 | 79065 | 0.289 | 0.207 | 1.00 |

| contrast | dPSNR (dB) | 95% CI | p | dDepth L1 | n |
|---|--:|---|--:|--:|--:|
| error budget proportional to map | +2.99 | [+2.96, +3.03] | 0.0078 | -0.0071 | 8 |

## tum_fr2_xyz

| condition | n | PSNR | depth L1 | map | mature cov. | young share | budget scale |
|---|--:|--:|--:|--:|--:|--:|--:|
| error_budget_prop | 8 | 27.89 | 0.0104 | 70760 | 1.000 | 0.134 | 2.65 |
| error_off | 8 | 24.16 | 0.0203 | 74569 | 0.379 | 0.224 | 1.00 |

| contrast | dPSNR (dB) | 95% CI | p | dDepth L1 | n |
|---|--:|---|--:|--:|--:|
| error budget proportional to map | +3.73 | [+3.64, +3.81] | 0.0078 | -0.0099 | 8 |
