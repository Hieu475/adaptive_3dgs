# Core Controlled Scientific Ablation Report (A1 to A6)

Strictly 1-variable ablation protocol starting from Full Ours (Section XXVIII).

## Table 4: Controlled 1-Variable Ablation Matrix

| Ablation ID | Removed Feature | Substituted Baseline | PSNR ↑ | Depth L1 ↓ | Opt Time (p50) | Jitter | Scientific Impact |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Reference** | None | Full Ours | **5.63 dB** | 1.3625 | 27.2 ms | 8.33 | Baseline performance |
| **A1** | Knapsack Solver | Greedy Top-$K$ Ranking | 5.62 dB | 1.3625 | 25.6 ms | 4.39 | Disregards cost heterogeneity |
| **A2** | Cost Model | Unit Cost ($c_i = 1$) | 5.61 dB | 1.3625 | 25.3 ms | 3.55 | Large Gaussians starve budget |
| **A3** | Hysteresis | Static Tier Thresholds | 5.60 dB | 1.3625 | 25.0 ms | 2.64 | High state switching (425.4/fr) |
| **A4** | Dynamic Threshold | Static Densification Thresh | 5.62 dB | 1.3625 | 23.4 ms | 3.45 | Uncontrolled map growth |
| **A5** | Pixel Attribution | Whole-Image Error | 5.61 dB | 1.3625 | 26.9 ms | 4.75 | Diluted spatial localization |
| **A6** | Learned Two-Head | Heuristic Utility | 5.63 dB | 1.3625 | 27.2 ms | 8.33 | Lower rank correlation with oracle |
