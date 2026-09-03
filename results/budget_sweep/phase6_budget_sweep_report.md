# Phase 6 & 8: Budget-Aware Selection Benchmark & Gate 3 Rigor

## 1. Gate 3 Headline Result ($B = 60\%$ Capacity)

- **Learned Two-Head Gain ($\Delta Q$):** **$+0.000460$**
- **Heuristic Knapsack Gain ($\Delta Q$):** **$+0.000366$**
- **Absolute Gain Difference:** **$+0.000094$**
- **Relative Gain:** **+25.7%**
- **95% Bootstrap CI on Absolute Gain:** **[$+0.000065$, $+0.000114$]** (Strictly Positive ✅)
- **Wilcoxon Signed-Rank Test:** $p = 0.03125$ (Statistically Significant ✅)
- **Cohen's $d$ Effect Size:** $d = +2.824$ (Large effect size)

## 2. Complete Budget Sweep Table ($B \in [10\%, 80\%]$)

| Budget Level | Policy | $\Delta Q$ (Joint Gain) ↑ | $\Delta$PSNR (dB) ↑ | Latency (ms) ↓ | OSE ↑ | Regret ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **10%** | **Oracle Upper Bound** | **+0.000110** | **+0.0010 dB** | 152.94 ms | **1.000** | +0.000000 |
| **10%** | **Learned Two-Head (Ours)** | **+0.000158** | **+0.0015 dB** | 210.65 ms | **1.439** | -0.000048 |
| **10%** | **Heuristic Knapsack (Ours)** | **+0.000108** | **+0.0010 dB** | 156.72 ms | **0.979** | +0.000002 |
| **10%** | Error × Influence | +0.000203 | +0.0017 dB | 167.17 ms | 1.847 | -0.000093 |
| **10%** | Error-Only Top-K | +0.000121 | +0.0010 dB | 178.02 ms | 1.103 | -0.000011 |
| **10%** | Random Baseline | +0.000037 | +0.0003 dB | 192.92 ms | 0.337 | +0.000073 |
| **20%** | **Oracle Upper Bound** | **+0.000335** | **+0.0031 dB** | 281.15 ms | **1.000** | +0.000000 |
| **20%** | **Learned Two-Head (Ours)** | **+0.000117** | **+0.0010 dB** | 350.57 ms | **0.349** | +0.000218 |
| **20%** | **Heuristic Knapsack (Ours)** | **+0.000073** | **+0.0005 dB** | 326.61 ms | **0.217** | +0.000262 |
| **20%** | Error × Influence | +0.000310 | +0.0028 dB | 335.15 ms | 0.925 | +0.000025 |
| **20%** | Error-Only Top-K | +0.000217 | +0.0019 dB | 313.97 ms | 0.646 | +0.000119 |
| **20%** | Random Baseline | +0.000127 | +0.0011 dB | 339.87 ms | 0.378 | +0.000209 |
| **40%** | **Oracle Upper Bound** | **+0.000478** | **+0.0044 dB** | 571.93 ms | **1.000** | +0.000000 |
| **40%** | **Learned Two-Head (Ours)** | **+0.000361** | **+0.0032 dB** | 648.19 ms | **0.755** | +0.000117 |
| **40%** | **Heuristic Knapsack (Ours)** | **+0.000200** | **+0.0017 dB** | 556.24 ms | **0.418** | +0.000278 |
| **40%** | Error × Influence | +0.000348 | +0.0031 dB | 622.94 ms | 0.728 | +0.000130 |
| **40%** | Error-Only Top-K | +0.000217 | +0.0018 dB | 583.21 ms | 0.455 | +0.000260 |
| **40%** | Random Baseline | +0.000247 | +0.0022 dB | 586.74 ms | 0.517 | +0.000231 |
| **60%** | **Oracle Upper Bound** | **+0.000504** | **+0.0045 dB** | 901.11 ms | **1.000** | +0.000000 |
| **60%** | **Learned Two-Head (Ours)** | **+0.000492** | **+0.0044 dB** | 925.55 ms | **0.977** | +0.000012 |
| **60%** | **Heuristic Knapsack (Ours)** | **+0.000376** | **+0.0032 dB** | 843.54 ms | **0.747** | +0.000128 |
| **60%** | Error × Influence | +0.000372 | +0.0032 dB | 952.51 ms | 0.739 | +0.000132 |
| **60%** | Error-Only Top-K | +0.000323 | +0.0028 dB | 942.59 ms | 0.641 | +0.000181 |
| **60%** | Random Baseline | +0.000345 | +0.0031 dB | 785.91 ms | 0.686 | +0.000158 |
| **80%** | **Oracle Upper Bound** | **+0.000514** | **+0.0045 dB** | 1063.30 ms | **1.000** | +0.000000 |
| **80%** | **Learned Two-Head (Ours)** | **+0.000500** | **+0.0044 dB** | 1114.88 ms | **0.971** | +0.000015 |
| **80%** | **Heuristic Knapsack (Ours)** | **+0.000537** | **+0.0047 dB** | 1008.26 ms | **1.044** | -0.000023 |
| **80%** | Error × Influence | +0.000470 | +0.0041 dB | 1228.95 ms | 0.914 | +0.000044 |
| **80%** | Error-Only Top-K | +0.000481 | +0.0042 dB | 1102.81 ms | 0.935 | +0.000034 |
| **80%** | Random Baseline | +0.000411 | +0.0036 dB | 955.39 ms | 0.799 | +0.000104 |

## 3. Visualizations
- **Figure 5:** Budget-Quality Curve (`results/figures/fig5_quality_at_budget.png`)
- **Figure 7:** Latency vs Quality Pareto Frontier (`results/figures/fig7_pareto_frontier.png`)
