# TT+VGG vs fixed budget at equal measured time

## tum_fr2_xyz

baseline (free growth): x1: 24.08 dB @ 88 ms, x2: 26.50 dB @ 96 ms, x4: 27.81 dB @ 96 ms

| T (ms) | TT+VGG PSNR | frame ms | map (k) | baseline PSNR at same time | gain |
|--:|--:|--:|--:|--:|--:|
| 70 | 27.30 | 77 | 33 | 24.08 (x1 @ 88 ms) | +3.22 and 11 ms faster |
| 85 | 27.81 | 75 | 35 | 24.08 (x1 @ 88 ms) | +3.73 and 13 ms faster |
| 100 | 27.94 | 72 | 32 | 24.08 (x1 @ 88 ms) | +3.86 and 16 ms faster |
| 120 | 27.95 | 73 | 33 | 24.08 (x1 @ 88 ms) | +3.87 and 16 ms faster |
| 140 | 27.94 | 72 | 32 | 24.08 (x1 @ 88 ms) | +3.86 and 16 ms faster |

## tum_fr1_xyz

baseline (free growth): x1: 21.51 dB @ 100 ms, x2: 22.77 dB @ 108 ms, x4: 24.45 dB @ 119 ms

| T (ms) | TT+VGG PSNR | frame ms | map (k) | baseline PSNR at same time | gain |
|--:|--:|--:|--:|--:|--:|
| 70 | 21.76 | 92 | 72 | 21.51 (x1 @ 100 ms) | +0.25 and 7 ms faster |
| 85 | 21.86 | 92 | 76 | 21.51 (x1 @ 100 ms) | +0.35 and 7 ms faster |
| 100 | 22.67 | 95 | 74 | 21.51 (x1 @ 100 ms) | +1.15 and 5 ms faster |
| 120 | 23.93 | 102 | 77 | 21.92 | +2.01 |
| 140 | 24.47 | 110 | 76 | 23.05 | +1.42 |

## replica_office0

baseline (free growth): x1: 30.62 dB @ 86 ms, x2: 32.26 dB @ 105 ms, x4: 33.44 dB @ 135 ms

| T (ms) | TT+VGG PSNR | frame ms | map (k) | baseline PSNR at same time | gain |
|--:|--:|--:|--:|--:|--:|
| 70 | 30.64 | 83 | 113 | 30.62 (x1 @ 86 ms) | +0.02 and 2 ms faster |
| 85 | 30.79 | 86 | 113 | 30.65 | +0.14 |
| 100 | 31.93 | 99 | 111 | 31.72 | +0.20 |
| 120 | 33.00 | 118 | 108 | 32.77 | +0.23 |
| 140 | 33.43 | 131 | 107 | 33.29 | +0.14 |

## replica_office1

baseline (free growth): x1: 33.85 dB @ 86 ms, x2: 34.86 dB @ 107 ms, x4: 35.42 dB @ 138 ms

| T (ms) | TT+VGG PSNR | frame ms | map (k) | baseline PSNR at same time | gain |
|--:|--:|--:|--:|--:|--:|
| 70 | 33.87 | 84 | 84 | 33.85 (x1 @ 86 ms) | +0.02 and 2 ms faster |
| 85 | 33.96 | 86 | 84 | 33.85 (x1 @ 86 ms) | +0.11 and 0 ms faster |
| 100 | 34.64 | 99 | 83 | 34.49 | +0.15 |
| 120 | 35.24 | 119 | 83 | 35.08 | +0.16 |
| 140 | 35.33 | 133 | 82 | 35.33 | +0.01 |
