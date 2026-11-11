# Multi-Scene 3DGS Benchmark & SOTA Comparison Report

**Date:** 2026-09-29 14:45:11  
**Resolution:** 320 $\times$ 240 | **Per-Frame Budget:** 15.0 ms | **Device:** cuda  

## 1. Multi-Scene Policy Comparison

| Scene | Policy | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Mean Depth L1 | FPS | Final Gaussians | Opt Time (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_fr2_xyz` | **NO_OP** | **15.95** | 15.47 | 0.509 | 0.1495 | 79.0 | 35,386 | 0.0 |
| `tum_fr2_xyz` | **ERROR_ONLY** | **19.94** | 18.04 | 0.688 | 0.1072 | 9.4 | 30,099 | 96.0 |
| `tum_fr2_xyz` | **OURS** | **20.24** | 18.83 | 0.723 | 0.1081 | 9.5 | 28,188 | 95.1 |
| `tum_fr2_xyz` | **FULL** | **20.66** | 19.56 | 0.738 | 0.1020 | 14.8 | 32,530 | 55.7 |
| `replica_office0` | **NO_OP** | **21.82** | 22.37 | 0.770 | 0.4405 | 58.6 | 69,016 | 0.0 |
| `replica_office0` | **ERROR_ONLY** | **30.40** | 30.23 | 0.949 | 0.3478 | 11.0 | 66,264 | 69.6 |
| `replica_office0` | **OURS** | **24.81** | 28.07 | 0.892 | 0.4638 | 11.5 | 56,944 | 67.8 |
| `replica_office0` | **FULL** | **31.33** | 32.11 | 0.956 | 0.3388 | 10.0 | 66,127 | 84.7 |
| `replica_room0` | **NO_OP** | **14.88** | 9.47 | 0.608 | 0.3936 | 51.7 | 75,911 | 0.0 |
| `replica_room0` | **ERROR_ONLY** | **22.91** | 17.76 | 0.926 | 0.3443 | 10.8 | 71,573 | 70.4 |
| `replica_room0` | **OURS** | **21.30** | 16.35 | 0.895 | 0.3815 | 11.2 | 58,500 | 69.7 |
| `replica_room0` | **FULL** | **24.19** | 20.21 | 0.945 | 0.3202 | 9.8 | 70,158 | 84.5 |
| `tum_fr1_desk` | **NO_OP** | **8.30** | 6.49 | 0.186 | 0.4295 | 60.2 | 59,601 | 0.0 |
| `tum_fr1_desk` | **ERROR_ONLY** | **10.56** | 9.11 | 0.419 | 0.4195 | 8.4 | 59,417 | 98.6 |
| `tum_fr1_desk` | **OURS** | **10.35** | 8.61 | 0.381 | 0.4215 | 8.4 | 59,398 | 98.0 |
| `tum_fr1_desk` | **FULL** | **10.68** | 9.09 | 0.436 | 0.4054 | 11.4 | 58,958 | 73.5 |

## 2. Comparison with Published SOTA (SplaTAM, RTG-SLAM, MonoGS, Point-SLAM)

| Scene | Method | Tracking Mode | PSNR (dB) ↑ | SSIM ↑ | FPS ↑ | Map Size ↓ | Speedup vs SplaTAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tum_fr2_xyz` | **OURS (Adaptive Throttling)** | Oracle (Mapping) | **20.24** | **0.723** | **9.5** | **28,188** | **~15×** |
| `tum_fr2_xyz` | Point-SLAM (ICCV'23) | Estimated | 17.62 | 0.680 | 0.2 | 120,000 | Baseline |
| `tum_fr2_xyz` | MonoGS (CVPR'24) | Estimated | 16.17 | 0.650 | 1.6 | 180,000 | Baseline |
| `tum_fr2_xyz` | Photo-SLAM (CVPR'24) | Estimated | 21.07 | 0.780 | 10.0 | 250,000 | Baseline |
| `tum_fr2_xyz` | RTG-SLAM (2024) | Estimated | 21.40 | 0.785 | 20.0 | 280,000 | Baseline |
| `tum_fr2_xyz` | SplaTAM (CVPR'24) | Estimated | 25.06 | 0.890 | 0.6 | 410,000 | Baseline |
| `replica_office0` | **OURS (Adaptive Throttling)** | Oracle (Mapping) | **24.81** | **0.892** | **11.5** | **56,944** | **~15×** |
| `replica_office0` | Point-SLAM (ICCV'23) | Estimated | 33.40 | 0.960 | 0.2 | 290,000 | Baseline |
| `replica_office0` | SplaTAM (CVPR'24) | Estimated | 38.26 | 0.975 | 0.6 | 635,000 | Baseline |
| `replica_room0` | **OURS (Adaptive Throttling)** | Oracle (Mapping) | **21.30** | **0.895** | **11.2** | **58,500** | **~15×** |
| `replica_room0` | Point-SLAM (ICCV'23) | Estimated | 30.50 | 0.940 | 0.2 | 280,000 | Baseline |
| `replica_room0` | SplaTAM (CVPR'24) | Estimated | 32.86 | 0.958 | 0.6 | 580,000 | Baseline |
| `tum_fr1_desk` | **OURS (Adaptive Throttling)** | Oracle (Mapping) | **10.35** | **0.381** | **8.4** | **59,398** | **~15×** |
| `tum_fr1_desk` | Point-SLAM (ICCV'23) | Estimated | 13.79 | 0.550 | 0.2 | 130,000 | Baseline |
| `tum_fr1_desk` | MonoGS (CVPR'24) | Estimated | 19.67 | 0.720 | 1.6 | 200,000 | Baseline |
| `tum_fr1_desk` | Photo-SLAM (CVPR'24) | Estimated | 20.97 | 0.810 | 10.0 | 260,000 | Baseline |
| `tum_fr1_desk` | SplaTAM (CVPR'24) | Estimated | 21.49 | 0.840 | 0.6 | 430,000 | Baseline |