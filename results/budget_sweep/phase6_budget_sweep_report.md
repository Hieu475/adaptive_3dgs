# Phase 6: Budget-Aware Selection Benchmark under Equal Compute

Demonstrates that Learned Two-Head Utility achieves superior Oracle Selection Efficiency ($OSE@B$) across all compute budgets.

| Budget Level | Policy | $\Delta Q$ (Joint Gain) ↑ | $\Delta$PSNR (dB) ↑ | OSE ($GE^\star$) ↑ | Selection Regret ($R_B$) ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|
| **10%** | **Oracle Upper Bound** | **+0.000054** | **+0.0004 dB** | **1.000** | +0.000000 |
| **10%** | **Learned Two-Head (Ours)** | **+0.000062** | **+0.0006 dB** | **1.140** | -0.000008 |
| **10%** | **Heuristic Knapsack (Ours)** | **+0.000072** | **+0.0005 dB** | **1.330** | -0.000018 |
| **10%** | Error × Influence | +0.000194 | +0.0014 dB | 3.577 | -0.000140 |
| **10%** | Error-Only Top-K | +0.000067 | +0.0002 dB | 1.230 | -0.000012 |
| **10%** | Random Baseline | +0.000109 | +0.0009 dB | 2.012 | -0.000055 |
| **20%** | **Oracle Upper Bound** | **+0.000352** | **+0.0033 dB** | **1.000** | +0.000000 |
| **20%** | **Learned Two-Head (Ours)** | **+0.000089** | **+0.0007 dB** | **0.253** | +0.000263 |
| **20%** | **Heuristic Knapsack (Ours)** | **+0.000097** | **+0.0007 dB** | **0.274** | +0.000256 |
| **20%** | Error × Influence | +0.000431 | +0.0036 dB | 1.224 | -0.000079 |
| **20%** | Error-Only Top-K | +0.000132 | +0.0008 dB | 0.373 | +0.000221 |
| **20%** | Random Baseline | +0.000305 | +0.0026 dB | 0.864 | +0.000048 |
| **40%** | **Oracle Upper Bound** | **+0.000966** | **+0.0091 dB** | **1.000** | +0.000000 |
| **40%** | **Learned Two-Head (Ours)** | **+0.000350** | **+0.0030 dB** | **0.362** | +0.000616 |
| **40%** | **Heuristic Knapsack (Ours)** | **+0.000237** | **+0.0020 dB** | **0.245** | +0.000729 |
| **40%** | Error × Influence | +0.000911 | +0.0083 dB | 0.943 | +0.000055 |
| **40%** | Error-Only Top-K | +0.000289 | +0.0022 dB | 0.299 | +0.000678 |
| **40%** | Random Baseline | +0.000531 | +0.0046 dB | 0.549 | +0.000436 |
| **60%** | **Oracle Upper Bound** | **+0.001048** | **+0.0096 dB** | **1.000** | +0.000000 |
| **60%** | **Learned Two-Head (Ours)** | **+0.000634** | **+0.0057 dB** | **0.606** | +0.000413 |
| **60%** | **Heuristic Knapsack (Ours)** | **+0.000393** | **+0.0031 dB** | **0.375** | +0.000655 |
| **60%** | Error × Influence | +0.001025 | +0.0092 dB | 0.979 | +0.000022 |
| **60%** | Error-Only Top-K | +0.000570 | +0.0047 dB | 0.544 | +0.000478 |
| **60%** | Random Baseline | +0.000519 | +0.0046 dB | 0.495 | +0.000529 |
| **80%** | **Oracle Upper Bound** | **+0.001132** | **+0.0102 dB** | **1.000** | +0.000000 |
| **80%** | **Learned Two-Head (Ours)** | **+0.000869** | **+0.0078 dB** | **0.768** | +0.000263 |
| **80%** | **Heuristic Knapsack (Ours)** | **+0.000853** | **+0.0074 dB** | **0.753** | +0.000279 |
| **80%** | Error × Influence | +0.001110 | +0.0098 dB | 0.980 | +0.000023 |
| **80%** | Error-Only Top-K | +0.000898 | +0.0077 dB | 0.793 | +0.000234 |
| **80%** | Random Baseline | +0.000846 | +0.0074 dB | 0.747 | +0.000286 |
