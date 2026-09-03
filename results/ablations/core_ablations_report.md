# R37 Core Ablation Study Report (A1 to A6)

## Table 4: The 6 Core Scientific Ablations

| Ablation ID | Variant | PSNR ↑ | Depth L1 ↓ | Opt Time (p50) | Jitter | Switches/Frame |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **A1_Binary_vs_Continuous** | | | | | | |
| | Binary (RTG) | 5.63 dB | 1.3625 | 20.2 ms | 10.72 | 413.1 |
| | Continuous (Ours) | 5.63 dB | 1.3625 | 18.4 ms | 6.79 | 427.0 |
|---|---|---|---|---|---|---|
| **A2_Error_vs_Influence** | | | | | | |
| | Error-Only | 5.63 dB | 1.3625 | 23.5 ms | 4.40 | 407.7 |
| | Error × Influence | 5.63 dB | 1.3625 | 24.4 ms | 0.92 | 408.9 |
|---|---|---|---|---|---|---|
| **A3_NoTemporal_vs_TemporalEMA** | | | | | | |
| | No Temporal (Instantaneous) | 5.63 dB | 1.3625 | 17.4 ms | 8.62 | 414.7 |
| | Temporal EMA (Ours) | 5.62 dB | 1.3625 | 18.1 ms | 7.51 | 417.4 |
|---|---|---|---|---|---|---|
| **A4_NoHysteresis_vs_Hysteresis** | | | | | | |
| | No Hysteresis | 5.65 dB | 1.3625 | 18.5 ms | 6.74 | 412.7 |
| | With Hysteresis (Ours) | 5.64 dB | 1.3625 | 19.2 ms | 7.80 | 400.0 |
|---|---|---|---|---|---|---|
| **A5_Fixed_vs_AdaptiveBudget** | | | | | | |
| | Fixed Budget | 5.64 dB | 1.3625 | 17.2 ms | 7.05 | 410.3 |
| | Adaptive Budget (Ours) | 5.65 dB | 1.3625 | 20.3 ms | 6.89 | 402.9 |
|---|---|---|---|---|---|---|
| **A6_Heuristic_vs_Learned** | | | | | | |
| | Heuristic Utility (Ours) | 5.63 dB | 1.3625 | 18.4 ms | 6.79 | 427.0 |
|---|---|---|---|---|---|---|