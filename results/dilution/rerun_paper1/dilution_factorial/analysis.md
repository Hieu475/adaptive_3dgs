# Corrected-pose selector × throttle factorial

Mean PSNR uses 150 evaluated frames. CI is paired seed bootstrap; p is two-sided paired Wilcoxon.

## tum_fr1_xyz

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 24.458 | 0.0309 | 79128 | 0.124 | 0.891 | 456 |
| error_cap23k | 8 | 22.519 | 0.1075 | 23000 | 0.102 | 0.654 | 79 |
| error_cap39k | 8 | 21.556 | 0.1029 | 39000 | 0.154 | 0.420 | 189 |
| error_coverage | 8 | 23.234 | 0.0335 | 37131 | 0.156 | 0.574 | 175 |
| error_coverage_backlog | 8 | 23.316 | 0.0334 | 35113 | 0.152 | 0.593 | 161 |
| error_off | 8 | 21.520 | 0.0378 | 79099 | 0.207 | 0.288 | 456 |
| temporal_coverage | 8 | 23.775 | 0.0314 | 37924 | 0.144 | 0.578 | 179 |
| temporal_coverage_backlog | 8 | 23.888 | 0.0318 | 33131 | 0.132 | 0.621 | 147 |
| temporal_off | 8 | 22.277 | 0.0368 | 85493 | 0.241 | 0.264 | 497 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 8 | +1.714 | [+1.637, +1.798] | 0.0078125 |
| coverage on temporal | 8 | +1.498 | [+1.481, +1.519] | 0.0078125 |
| backlog on error | 8 | +0.082 | [+0.046, +0.118] | 0.015625 |
| backlog on temporal | 8 | +0.113 | [+0.096, +0.131] | 0.0078125 |
| selector without throttle | 8 | +0.757 | [+0.696, +0.829] | 0.0078125 |
| selector with coverage | 8 | +0.541 | [+0.507, +0.572] | 0.0078125 |
| error budget x4 | 8 | +2.939 | [+2.885, +3.009] | 0.0078125 |
| error cap 23k | 8 | +0.999 | [+0.958, +1.045] | 0.0078125 |
| error cap 39k | 8 | +0.036 | [-0.027, +0.104] | 0.4609375 |
| selector × coverage interaction | 8 | -0.216 | [-0.302, -0.136] | 0.0078125 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(64, -4.42), (12, -3.97), (52, -3.95), (34, -3.73), (31, -3.68)]; error_coverage: [(91, -4.76), (64, -4.69), (70, -4.44), (34, -4.43), (52, -4.28)]; temporal_coverage: [(91, -5.14), (34, -4.6), (64, -4.57), (70, -4.35), (12, -4.17)]

## tum_fr2_xyz

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 27.822 | 0.0107 | 71277 | 0.138 | 0.970 | 395 |
| error_cap23k | 8 | 26.889 | 0.0217 | 23000 | 0.102 | 0.731 | 73 |
| error_cap39k | 8 | 24.926 | 0.0237 | 39000 | 0.163 | 0.498 | 180 |
| error_coverage | 8 | 27.661 | 0.0118 | 21660 | 0.095 | 0.874 | 64 |
| error_coverage_backlog | 8 | 27.672 | 0.0117 | 21454 | 0.094 | 0.877 | 63 |
| error_off | 8 | 24.130 | 0.0205 | 73115 | 0.227 | 0.379 | 407 |
| temporal_coverage | 8 | 27.754 | 0.0108 | 21469 | 0.092 | 0.874 | 63 |
| temporal_coverage_backlog | 8 | 27.733 | 0.0109 | 21838 | 0.093 | 0.863 | 65 |
| temporal_off | 8 | 25.265 | 0.0134 | 86196 | 0.244 | 0.347 | 494 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 8 | +3.531 | [+3.380, +3.697] | 0.0078125 |
| coverage on temporal | 8 | +2.490 | [+2.395, +2.590] | 0.0078125 |
| backlog on error | 8 | +0.011 | [-0.046, +0.069] | 0.7421875 |
| backlog on temporal | 8 | -0.022 | [-0.043, -0.002] | 0.109375 |
| selector without throttle | 8 | +1.134 | [+1.046, +1.241] | 0.0078125 |
| selector with coverage | 8 | +0.094 | [+0.047, +0.136] | 0.0390625 |
| error budget x4 | 8 | +3.692 | [+3.555, +3.833] | 0.0078125 |
| error cap 23k | 8 | +2.759 | [+2.619, +2.897] | 0.0078125 |
| error cap 39k | 8 | +0.796 | [+0.668, +0.925] | 0.0078125 |
| selector × coverage interaction | 8 | -1.041 | [-1.162, -0.934] | 0.0078125 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(22, -8.22), (53, -7.06), (43, -6.15), (26, -5.79), (46, -5.36)]; error_coverage: [(22, -8.05), (102, -7.07), (53, -6.87), (43, -6.24), (26, -5.69)]; temporal_coverage: [(22, -8.18), (102, -7.04), (53, -6.83), (43, -6.25), (26, -5.91)]
