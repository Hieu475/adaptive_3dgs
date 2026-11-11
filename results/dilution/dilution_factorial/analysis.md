# Corrected-pose selector × throttle factorial

Mean PSNR uses 150 evaluated frames. CI is paired seed bootstrap; p is two-sided paired Wilcoxon.

## tum_fr1_xyz

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 24.190 | 0.0402 | 79339 | 0.125 | 0.886 | 457 |
| error_cap23k | 8 | 22.301 | 0.1023 | 23000 | 0.102 | 0.655 | 78 |
| error_cap39k | 8 | 21.152 | 0.1046 | 39000 | 0.154 | 0.422 | 188 |
| error_coverage | 8 | 22.885 | 0.0443 | 37946 | 0.160 | 0.563 | 180 |
| error_coverage_backlog | 8 | 22.952 | 0.0456 | 35886 | 0.154 | 0.584 | 166 |
| error_off | 8 | 21.116 | 0.0505 | 80634 | 0.208 | 0.286 | 466 |
| temporal_coverage | 8 | 23.479 | 0.0425 | 38678 | 0.139 | 0.572 | 184 |
| temporal_coverage_backlog | 8 | 23.552 | 0.0450 | 32458 | 0.126 | 0.621 | 143 |
| temporal_off | 8 | 22.059 | 0.0484 | 89066 | 0.223 | 0.265 | 521 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 8 | +1.769 | [+1.729, +1.810] | 0.0078125 |
| coverage on temporal | 8 | +1.420 | [+1.405, +1.435] | 0.0078125 |
| backlog on error | 8 | +0.067 | [+0.031, +0.103] | 0.0390625 |
| backlog on temporal | 8 | +0.073 | [+0.058, +0.090] | 0.0078125 |
| selector without throttle | 8 | +0.943 | [+0.908, +0.978] | 0.0078125 |
| selector with coverage | 8 | +0.594 | [+0.571, +0.614] | 0.0078125 |
| error budget x4 | 8 | +3.074 | [+3.039, +3.110] | 0.0078125 |
| error cap 23k | 8 | +1.185 | [+1.150, +1.218] | 0.0078125 |
| error cap 39k | 8 | +0.036 | [-0.055, +0.108] | 0.25 |
| selector × coverage interaction | 8 | -0.349 | [-0.384, -0.316] | 0.0078125 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(64, -3.96), (12, -3.83), (52, -3.69), (31, -3.54), (34, -3.51)]; error_coverage: [(91, -5.0), (64, -4.3), (34, -4.28), (12, -4.08), (52, -4.05)]; temporal_coverage: [(91, -4.94), (64, -4.31), (34, -4.26), (70, -4.09), (12, -4.08)]

## tum_fr1_desk

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_coverage | 3 | 22.643 | 0.0590 | 25209 | 0.162 | 0.522 | 166 |
| error_coverage_backlog | 3 | 22.660 | 0.0585 | 24762 | 0.159 | 0.536 | 161 |
| error_off | 3 | 22.301 | 0.0481 | 55911 | 0.183 | 0.324 | 378 |
| temporal_coverage | 3 | 22.519 | 0.0595 | 27318 | 0.148 | 0.501 | 183 |
| temporal_coverage_backlog | 3 | 21.932 | 0.0686 | 23263 | 0.137 | 0.588 | 149 |
| temporal_off | 3 | 22.417 | 0.0476 | 60256 | 0.174 | 0.313 | 407 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 3 | +0.342 | [+0.326, +0.358] | 0.25 |
| coverage on temporal | 3 | +0.102 | [+0.073, +0.146] | 0.25 |
| backlog on error | 3 | +0.017 | [-0.083, +0.072] | 1.0 |
| backlog on temporal | 3 | -0.587 | [-0.631, -0.556] | 0.25 |
| selector without throttle | 3 | +0.116 | [+0.066, +0.153] | 0.25 |
| selector with coverage | 3 | -0.124 | [-0.219, -0.066] | 0.25 |
| selector × coverage interaction | 3 | -0.240 | [-0.284, -0.196] | 0.25 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(21, -4.93), (98, -3.2), (13, -3.01), (47, -2.46), (109, -2.43)]; error_coverage: [(21, -5.0), (98, -3.77), (134, -3.46), (13, -2.79), (109, -2.78)]; temporal_coverage: [(21, -4.2), (98, -3.76), (134, -3.74), (13, -3.13), (109, -2.88)]

## tum_fr3_sitting_static

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_coverage | 3 | 26.798 | 0.0426 | 14117 | 0.068 | 1.000 | 30 |
| error_coverage_backlog | 3 | 26.796 | 0.0427 | 14161 | 0.068 | 1.000 | 30 |
| error_off | 3 | 26.029 | 0.0446 | 24308 | 0.109 | 0.832 | 98 |
| temporal_coverage | 3 | 26.806 | 0.0427 | 14118 | 0.068 | 1.000 | 30 |
| temporal_coverage_backlog | 3 | 26.793 | 0.0427 | 14161 | 0.068 | 1.000 | 30 |
| temporal_off | 3 | 26.431 | 0.0432 | 24736 | 0.106 | 0.835 | 101 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 3 | +0.769 | [+0.761, +0.784] | 0.25 |
| coverage on temporal | 3 | +0.375 | [+0.320, +0.433] | 0.25 |
| backlog on error | 3 | -0.003 | [-0.009, +0.009] | 0.75 |
| backlog on temporal | 3 | -0.012 | [-0.034, +0.005] | 0.5 |
| selector without throttle | 3 | +0.401 | [+0.368, +0.451] | 0.25 |
| selector with coverage | 3 | +0.007 | [-0.021, +0.032] | 0.75 |
| selector × coverage interaction | 3 | -0.394 | [-0.441, -0.352] | 0.25 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(74, -4.13), (38, -3.56), (127, -3.56), (2, -3.54), (114, -3.34)]; error_coverage: [(114, -5.16), (127, -4.69), (140, -4.64), (74, -4.47), (38, -3.53)]; temporal_coverage: [(114, -5.13), (127, -4.79), (140, -4.65), (74, -4.47), (2, -3.51)]

## tum_fr2_xyz

| Condition | n | PSNR | Depth L1 | Map N | Young budget share | Old selected / old population | Spawn/frame |
|:--|--:|--:|--:|--:|--:|--:|--:|
| error_budget4 | 8 | 27.336 | 0.0142 | 93816 | 0.167 | 0.854 | 546 |
| error_cap23k | 8 | 26.690 | 0.0215 | 23000 | 0.102 | 0.729 | 74 |
| error_coverage | 8 | 27.380 | 0.0148 | 24425 | 0.107 | 0.823 | 83 |
| error_coverage_backlog | 8 | 27.361 | 0.0149 | 24416 | 0.107 | 0.823 | 83 |
| error_off | 8 | 24.046 | 0.0236 | 80291 | 0.225 | 0.358 | 456 |
| temporal_coverage | 8 | 27.640 | 0.0130 | 23765 | 0.100 | 0.834 | 79 |
| temporal_coverage_backlog | 8 | 27.659 | 0.0131 | 23414 | 0.099 | 0.841 | 76 |
| temporal_off | 8 | 24.957 | 0.0165 | 105512 | 0.228 | 0.302 | 624 |

| Contrast | n | Δ PSNR | 95% CI | p |
|:--|--:|--:|:--|--:|
| coverage on error | 8 | +3.334 | [+3.186, +3.504] | 0.0078125 |
| coverage on temporal | 8 | +2.683 | [+2.642, +2.724] | 0.0078125 |
| backlog on error | 8 | -0.019 | [-0.128, +0.071] | 0.9453125 |
| backlog on temporal | 8 | +0.019 | [-0.039, +0.078] | 0.546875 |
| selector without throttle | 8 | +0.910 | [+0.824, +0.995] | 0.0078125 |
| selector with coverage | 8 | +0.260 | [+0.187, +0.330] | 0.0078125 |
| error budget x4 | 8 | +3.289 | [+3.175, +3.406] | 0.0078125 |
| error cap 23k | 8 | +2.644 | [+2.534, +2.757] | 0.0078125 |
| selector × coverage interaction | 8 | -0.650 | [-0.799, -0.505] | 0.0078125 |

Largest PSNR drops for seed 42 (frame, Δ dB): error_off: [(22, -7.96), (53, -7.28), (43, -5.96), (56, -5.67), (46, -5.6)]; error_coverage: [(22, -7.93), (53, -6.81), (102, -6.52), (43, -5.96), (46, -5.66)]; temporal_coverage: [(22, -7.96), (102, -6.96), (53, -6.61), (43, -6.04), (46, -5.61)]
