# Real TUM RGB-D (fr1/desk) Multi-Seed Benchmark Summary

Evaluated across 5 random seeds ([42, 43, 44, 45, 46]) on real sensor data with noisy depth & camera motion.

| Policy | PSNR (dB) ↑ | Depth L1 (m) ↓ | Opt Time (ms) ↓ |
|:---:|:---:|:---:|:---:|
| **full** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 27.9 ± 5.7 ms |
| **random** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 25.4 ± 1.6 ms |
| **error_only** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 24.5 ± 0.9 ms |
| **error_influence** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 24.1 ± 0.8 ms |
| **error_influence_temporal** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 24.0 ± 1.5 ms |
| **top_k** | 12.99 ± 0.09 | 0.3121 ± 0.0038 | 30.2 ± 1.7 ms |
| **ours** | 14.43 ± 0.07 | 0.2464 ± 0.0025 | 24.5 ± 1.2 ms |

