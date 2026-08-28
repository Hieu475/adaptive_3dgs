# Real TUM RGB-D (fr1/desk) Multi-Seed Benchmark Summary

Evaluated across 3 random seeds ([42, 43, 44]) on real sensor data with noisy depth & camera motion.

| Policy | PSNR (dB) ↑ | Depth L1 (m) ↓ | Opt Time (ms) ↓ |
|:---:|:---:|:---:|:---:|
| **full** | 5.75 ± 0.01 | 1.2150 ± 0.0000 | 40.3 ± 13.8 ms |
| **random** | 5.76 ± 0.01 | 1.2150 ± 0.0000 | 23.3 ± 2.0 ms |
| **error_only** | 5.75 ± 0.01 | 1.2150 ± 0.0000 | 28.7 ± 5.7 ms |
| **error_influence** | 5.75 ± 0.01 | 1.2150 ± 0.0000 | 23.2 ± 0.7 ms |
| **top_k** | 5.75 ± 0.01 | 1.2150 ± 0.0000 | 22.4 ± 0.9 ms |
| **ours** | 5.75 ± 0.01 | 1.2150 ± 0.0000 | 16.3 ± 1.0 ms |

