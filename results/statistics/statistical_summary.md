# Protocol v1 Confirmatory Statistical Summary

Multi-seed independent verification across $N=5$ seeds: `[42, 43, 44, 45, 46]`.

## 1. Primary Research Question Verification

| Gate / Research Question | Metric | Mean | 95% Bootstrap CI | Paired Wilcoxon $p$ | Effect Size ($d$) | Scientific Assessment |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| Gate 1 | Headroom H (Joint) | +0.000151 | [+0.000093, +0.000214] | 0.0312 | +1.97 | Statistically Significant ✅ |
| Gate 1 | Negative Utility Rate (%) | 13.33% | [11.00%, 16.33%] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Random | 0.2640 | [0.2119, 0.3074] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - RGB Error | 0.3454 | [0.3454, 0.3454] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Error × Influence | 0.3362 | [0.3362, 0.3362] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Binary | 0.2882 | [0.2882, 0.2882] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Heuristic Knapsack | 0.1777 | [0.1777, 0.1777] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Learned Two-Head (Ours) | 0.3535 | [0.2880, 0.4077] | N/A | N/A | Empirical Evidence Provided |
| Gate 2 | OSE@20% - Oracle (Reference) | 1.0000 | [1.0000, 1.0000] | N/A | N/A | Empirical Evidence Provided |
| Gate 3 | Absolute Gain at B=60% | +0.000094 | [+0.000065, +0.000114] | 0.0312 | +2.82 | Statistically Significant ✅ |
| Gate 4 | Online ΔQ vs Error-Only (dB) | -0.0000 dB | [-0.0007, +0.0006] | 0.5000 | -0.02 | Inconclusive |

## 2. Gate 4 Systems Latency Breakdown (Phase 20)

| Policy | Mean Opt Latency | P95 Latency | P99 Latency | Max Latency | Deadline Violations (> 15 ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `FULL` | 138.32 ms | 165.44 ms | 182.36 ms | 191.68 ms | 0.0% |
| `RANDOM` | 111.11 ms | 126.52 ms | 132.98 ms | 137.03 ms | 100.0% |
| `ERROR_ONLY` | 110.80 ms | 129.89 ms | 142.57 ms | 149.60 ms | 100.0% |
| `OURS` | 67.82 ms | 87.76 ms | 92.74 ms | 93.83 ms | 100.0% |