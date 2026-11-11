# vgg_fresh

| scene | n | base PSNR (ms) | TT | TT+VGG | TT+cap | fixed cap | prop. budget | VGG |
|---|--:|--|--|--|--|--|--|--|
| tum_fr1_desk | 3 | 23.09 (92) | +1.02 (82) | +1.04 (82) | +0.47 (65) | -0.18 (90) | +1.02 (82) | - |
| tum_fr1_room | 3 | 16.81 (97) | +0.56 (93) | +0.58 (94) | +0.42 (84) | -0.11 (94) | +0.77 (90) | - |
| tum_fr1_xyz | 3 | 22.88 (98) | +0.74 (96) | +0.87 (96) | +1.18 (69) | +0.99 (90) | +2.05 (115) | - |
| tum_fr2_xyz | 3 | 24.63 (90) | +1.19 (87) | +1.63 (85) | +1.80 (62) | +1.58 (77) | +1.73 (95) | - |
| tum_fr3_sitting_static | 3 | 28.35 (63) | +0.34 (55) | +0.41 (56) | +0.47 (51) | +0.50 (50) | +0.35 (55) | - |

| method | scenes | positive | worst | median | mean | best | mean dt (ms) |
|---|--:|--:|--:|--:|--:|--:|--:|
| fixed cap | 5 | 3 | -0.18 | +0.50 | +0.55 | +1.58 | -8 |
| prop. budget | 5 | 5 | +0.35 | +1.02 | +1.18 | +2.05 | -0 |
| TT | 5 | 5 | +0.34 | +0.74 | +0.77 | +1.19 | -5 |
| TT+VGG | 5 | 5 | +0.41 | +0.87 | +0.90 | +1.63 | -5 |
| TT+cap | 5 | 5 | +0.42 | +0.47 | +0.87 | +1.80 | -22 |

TT+VGG minus TT (same controller): positive in 5/5 scenes, worst +0.02, median +0.07, mean +0.14 dB.
TT+VGG better than baseline and within +15% of its frame time: 5/5 scenes.