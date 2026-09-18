# Multi-Scene 3DGS Edge-SLAM Benchmark Summary

**Generated:** 2026-09-19T01:06:51.976155

## 1. Cross-Scene Quality & Latency Comparison

| Scene | Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Mean Depth L1 (m) | Opt Time (ms) | Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_fr1_xyz` | **NO_OP** | 12.76 | 11.22 | 0.7132 | 0.1428 | 0.0 | 93.8% |
| `tum_fr1_xyz` | **ERROR_ONLY** | 15.46 | 15.70 | 0.8362 | 0.1175 | 98.5 | 95.4% |
| `tum_fr1_xyz` | **OURS** | 15.40 | 15.28 | 0.8529 | 0.1042 | 91.9 | 97.0% |
| `tum_fr1_xyz` | **FULL** | 16.27 | 17.13 | 0.8757 | 0.0884 | 86.5 | 97.3% |
| `replica_office0` | **NO_OP** | 20.65 | 28.70 | 0.7446 | 0.5337 | 0.0 | 99.9% |
| `replica_office0` | **ERROR_ONLY** | 27.64 | 35.46 | 0.9281 | 0.4770 | 68.1 | 99.9% |
| `replica_office0` | **OURS** | 28.68 | 35.60 | 0.9337 | 0.4622 | 63.9 | 99.9% |
| `replica_office0` | **FULL** | 31.68 | 40.54 | 0.9567 | 0.4151 | 119.4 | 100.0% |
| `tum_fr2_xyz` | **NO_OP** | 13.49 | 10.07 | 0.4237 | 0.1850 | 0.0 | 88.6% |
| `tum_fr2_xyz` | **ERROR_ONLY** | 17.19 | 15.75 | 0.6948 | 0.1419 | 100.6 | 93.5% |
| `tum_fr2_xyz` | **OURS** | 18.51 | 15.03 | 0.8076 | 0.1312 | 98.8 | 97.4% |
| `tum_fr2_xyz` | **FULL** | 20.61 | 20.28 | 0.8624 | 0.0953 | 187.3 | 97.5% |

## 2. Knapsack Value-Density Margin (OURS vs ERROR_ONLY)

| Scene | Frames | OURS Mean PSNR | ERROR_ONLY Mean PSNR | Delta PSNR (dB) | OURS SSIM | ERROR_ONLY SSIM | Delta SSIM (%) | Matched Compute? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_fr1_xyz` | 30 | **15.40 dB** | 15.46 dB | **-0.05 dB** | **0.8529** | 0.8362 | **+2.0%** | 91.9 vs 98.5 ms |
| `replica_office0` | 30 | **28.68 dB** | 27.64 dB | **+1.03 dB** | **0.9337** | 0.9281 | **+0.6%** | Yes (63.9 vs 68.1 ms) |
| `tum_fr2_xyz` | 150 | **18.51 dB** | 17.19 dB | **+1.31 dB** | **0.8076** | 0.6948 | **+16.2%** | Yes (98.8 vs 100.6 ms) |

## 3. Comparison with Published SOTA (TUM RGB-D)

| Scene | Point-SLAM (ICCV'23) | MonoGS (CVPR'24) | SplaTAM (CVPR'24) | OURS (Phase 13) | FULL (Ceiling) | OURS Speedup vs SplaTAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_fr1_xyz` | 14.50 dB | — | — | **15.40 dB** | **16.27 dB** | Real-time |
| `replica_office0` | 33.40 dB | — | 38.26 dB | **28.68 dB** | **31.68 dB** | ~15x (10 FPS vs 0.6 FPS) |
| `tum_fr2_xyz` | 17.62 dB | 16.17 dB | 25.06 dB | **18.51 dB** | **20.61 dB** | ~15x (10 FPS vs 0.6 FPS) |