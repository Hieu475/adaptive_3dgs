# Real TUM RGB-D (fr1/desk) Multi-Seed Benchmark Summary

Evaluated across 3 random seeds ([42, 43, 44]) on real sensor data with noisy depth & camera motion.

| Policy | PSNR (dB) ↑ | Depth L1 (m) ↓ | Opt Time (ms) ↓ |
|:---:|:---:|:---:|:---:|
| **full** | 5.89 ± 0.02 | 1.2337 ± 0.0000 | 50.1 ± 3.8 ms |
| **random** | 5.88 ± 0.01 | 1.2337 ± 0.0000 | 17.9 ± 3.0 ms |
| **error_only** | 5.89 ± 0.02 | 1.2337 ± 0.0000 | 20.6 ± 2.8 ms |
| **error_influence** | 5.89 ± 0.02 | 1.2337 ± 0.0000 | 19.2 ± 1.4 ms |
| **top_k** | 5.89 ± 0.02 | 1.2337 ± 0.0000 | 19.5 ± 2.9 ms |
| **ours** | 5.89 ± 0.02 | 1.2337 ± 0.0000 | 8.9 ± 1.8 ms |

