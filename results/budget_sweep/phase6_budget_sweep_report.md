# Phase 6 & 8: Budget-Aware Selection Benchmark & Gate 3 Rigor

## 1. Gate 3 Headline Result ($B = 60\%$ Capacity)

- **Learned Two-Head Gain ($\Delta Q$):** **$+0.000309$**
- **Heuristic Knapsack Gain ($\Delta Q$):** **$+0.000741$**
- **Absolute Gain Difference:** **$-0.000432$**
- **Relative Gain:** **-58.4%**
- **95% Bootstrap CI on Absolute Gain:** **[$-0.000544$, $-0.000321$]** (Cuts 0)
- **Wilcoxon Signed-Rank Test:** $p = 1.00000$ (Not Significant)
- **Cohen's $d$ Effect Size:** $d = -3.050$ (Large effect size)

## 2. Complete Budget Sweep Table ($B \in [10\%, 80\%]$)

| Budget Level | Policy | $\Delta Q$ (Joint Gain) ↑ | $\Delta$PSNR (dB) ↑ | Latency (ms) ↓ | OSE ↑ | Regret ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **10%** | **Oracle Upper Bound** | **+0.000151** | **+0.0014 dB** | 65.55 ms | **1.000** | +0.000000 |
| **10%** | **Learned Two-Head (Ours)** | **+0.000013** | **+0.0001 dB** | 72.33 ms | **0.087** | +0.000138 |
| **10%** | **Heuristic Knapsack (Ours)** | **+0.000023** | **+0.0001 dB** | 83.50 ms | **0.154** | +0.000127 |
| **10%** | Error × Influence | +0.000428 | +0.0041 dB | 115.94 ms | 2.839 | -0.000277 |
| **10%** | Error-Only Top-K | +0.000088 | +0.0007 dB | 65.80 ms | 0.586 | +0.000062 |
| **10%** | Random Baseline | +0.000142 | +0.0012 dB | 106.07 ms | 0.942 | +0.000009 |
| **20%** | **Oracle Upper Bound** | **+0.000296** | **+0.0028 dB** | 84.89 ms | **1.000** | +0.000000 |
| **20%** | **Learned Two-Head (Ours)** | **+0.000057** | **+0.0003 dB** | 119.03 ms | **0.191** | +0.000240 |
| **20%** | **Heuristic Knapsack (Ours)** | **+0.000105** | **+0.0007 dB** | 128.57 ms | **0.355** | +0.000191 |
| **20%** | Error × Influence | +0.000706 | +0.0068 dB | 182.57 ms | 2.382 | -0.000410 |
| **20%** | Error-Only Top-K | +0.000119 | +0.0008 dB | 121.71 ms | 0.403 | +0.000177 |
| **20%** | Random Baseline | +0.000264 | +0.0024 dB | 164.08 ms | 0.892 | +0.000032 |
| **40%** | **Oracle Upper Bound** | **+0.000849** | **+0.0082 dB** | 203.76 ms | **1.000** | +0.000000 |
| **40%** | **Learned Two-Head (Ours)** | **+0.000158** | **+0.0011 dB** | 211.47 ms | **0.187** | +0.000691 |
| **40%** | **Heuristic Knapsack (Ours)** | **+0.000635** | **+0.0061 dB** | 224.86 ms | **0.748** | +0.000214 |
| **40%** | Error × Influence | +0.000943 | +0.0085 dB | 302.65 ms | 1.110 | -0.000094 |
| **40%** | Error-Only Top-K | +0.000293 | +0.0021 dB | 216.51 ms | 0.345 | +0.000557 |
| **40%** | Random Baseline | +0.000399 | +0.0034 dB | 261.56 ms | 0.470 | +0.000450 |
| **60%** | **Oracle Upper Bound** | **+0.001027** | **+0.0095 dB** | 287.98 ms | **1.000** | +0.000000 |
| **60%** | **Learned Two-Head (Ours)** | **+0.000333** | **+0.0026 dB** | 289.74 ms | **0.324** | +0.000694 |
| **60%** | **Heuristic Knapsack (Ours)** | **+0.000860** | **+0.0078 dB** | 281.58 ms | **0.837** | +0.000167 |
| **60%** | Error × Influence | +0.001053 | +0.0093 dB | 347.61 ms | 1.026 | -0.000026 |
| **60%** | Error-Only Top-K | +0.000815 | +0.0071 dB | 308.44 ms | 0.794 | +0.000212 |
| **60%** | Random Baseline | +0.000648 | +0.0054 dB | 315.50 ms | 0.631 | +0.000379 |
| **80%** | **Oracle Upper Bound** | **+0.001098** | **+0.0098 dB** | 340.59 ms | **1.000** | +0.000000 |
| **80%** | **Learned Two-Head (Ours)** | **+0.000614** | **+0.0050 dB** | 333.35 ms | **0.559** | +0.000484 |
| **80%** | **Heuristic Knapsack (Ours)** | **+0.001004** | **+0.0087 dB** | 338.37 ms | **0.914** | +0.000094 |
| **80%** | Error × Influence | +0.001093 | +0.0094 dB | 362.23 ms | 0.995 | +0.000005 |
| **80%** | Error-Only Top-K | +0.001033 | +0.0090 dB | 359.78 ms | 0.940 | +0.000065 |
| **80%** | Random Baseline | +0.000927 | +0.0080 dB | 372.68 ms | 0.844 | +0.000172 |

## 3. Visualizations
- **Figure 5:** Budget-Quality Curve (`results/figures/fig5_quality_at_budget.png`)
- **Figure 7:** Latency vs Quality Pareto Frontier (`results/figures/fig7_pareto_frontier.png`)
