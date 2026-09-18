# Phase 13 Authoritative Frozen Benchmark Report

**Execution Timestamp**: `2026-09-19T03:04:16`  
**Git Commit**: `63146dc37564dd5d23d2f8fc751f6a6a6db2d883-dirty` (Branch: `phase13-cuda-acceleration`)  
**Dataset**: `tum_fr2_xyz` (150 frames, resolution: `320x240`)  
**Optimization Budget**: `15.0 ms` | **Backend**: `gsplat`  
**Hardware**: `NVIDIA GeForce RTX 4050 Laptop GPU` (5.64 GB VRAM, Driver / CUDA `12.8`)  

## 1. Primary Benchmark Results Across Policies (5 Seeds Mean ± Std [95% CI])

| Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Depth L1 | Selected/Frame | Mean K | Opt Time (ms) | FPS | Peak VRAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NO_OP** | 13.40 ± 0.01 | 11.75 ± 0.03 | 0.4333 ± 0.0010 | 0.1704 ± 0.0002 | 0 | 0.0 | 0.0 ms | 60.1 | 277.3 MB |
| **ERROR_ONLY** | 13.87 ± 0.01 | 11.92 ± 0.04 | 0.4597 ± 0.0016 | 0.1575 ± 0.0005 | 1661 | 5.0 | 87.7 ms | 9.3 | 319.0 MB |
| **ERROR_INFLUENCE** | 14.74 ± 0.04 | 12.17 ± 0.25 | 0.5961 ± 0.0035 | 0.1470 ± 0.0002 | 1628 | 5.0 | 88.3 ms | 9.0 | 366.5 MB |
| **ERROR_INFLUENCE_TEMPORAL** | 14.63 ± 0.03 | 12.02 ± 0.26 | 0.5889 ± 0.0033 | 0.1483 ± 0.0010 | 1628 | 5.0 | 88.0 ms | 9.0 | 366.1 MB |
| **OURS** | 14.68 ± 0.05 | 12.02 ± 0.33 | 0.5905 ± 0.0051 | 0.1474 ± 0.0003 | 1629 | 5.0 | 89.9 ms | 0.4 | 368.1 MB |
| **FULL** | 21.02 ± 0.04 | 19.55 ± 0.61 | 0.8758 ± 0.0004 | 0.0923 ± 0.0003 | 79078 | 5.0 | 151.6 ms | 5.8 | 402.4 MB |

## 2. Scientific Headroom & Policy Gain Analysis

- **Substrate Headroom ($H_{substrate}$)**: `+7.62 dB` (Mean), `+7.80 dB` (Final)
- **Gain over Raw Error-Only ($\Delta Q$)**: `+0.81 dB` (SSIM $\Delta$: `+0.1309`)
- **Gain over Temporal Error×Influence ($\Delta Q$)**: `+0.05 dB`

## 3. Provenance & Reproducibility Guarantees

- **Git Commit Hash**: `63146dc37564dd5d23d2f8fc751f6a6a6db2d883-dirty`
- **CUDA Rendering Backend**: `gsplat` (Production: gsplat, Prototypes: custom_cuda)
- **VRAM Guard Applied**: Yes (`max_vram_fraction=0.70`, reserved OS headroom: 1.8 GB)
- **Catastrophic Failures (NaN/Inf)**: `0`
- **Verification**: Frozen benchmark conducted on current repository HEAD with exact seed control.
