# Phase 5 — Stage A: Controlled Equal-Compute Budget Benchmark Report

## 1. Executive Summary & RQ2 Evaluation

- **Primary Hypothesis (RQ2):** Under strict compute budgets $B \in \{10\%, 20\%, 40\%, 60\%, 80\%\}$, Gaussian selection via frozen predicted marginal utility $\hat U_i$ delivers superior realized photometric/geometric quality gain $\Delta Q^{\text{realized}}$ compared to heuristic and error baselines under identical compute budgets.
- **Target Cross-Scene Benchmark Split:** `tum_fr2_xyz` (zero-shot transfer from `tum_fr1_desk`).
- **Provenance:** Fully evaluated across 5 protocol seeds (`[42, 43, 44, 45, 46]`) on frames `[10, 20]`.
- **Runtime Guarantees:** Zero oracle leakage at runtime; Phase 4 predictor model frozen bitwise; bitwise snapshot/restore ensures 100% equal scene initialization for all policies.

## 2. Statistical Validation at Target Budget $B = 60\%$

| Comparison | Metric | Value | Statistical Status |
|:---|:---|:---:|:---:|
| **Ours vs Heuristic** | Absolute Gain $\Delta Q$ | `+-0.000024` | Cohen's $d = -0.634$ |
| | Relative Gain (%) | `+-12.77%` | Wilcoxon $p = 9.9616e-01$ |
| **Ours vs Error-Only** | Absolute Gain $\Delta Q$ | `+-0.000033` | Cohen's $d = -0.431$ |
| | Relative Gain (%) | `+-16.80%` | Wilcoxon $p = 9.1992e-01$ |

## 3. Realized Quality Gain & Efficiency Matrix (Mean ± 95% CI)

| Budget | Policy | Realized $\Delta Q$ | Realized $\Delta$PSNR (dB) | OSE | Regret | Actual Latency (ms) | Violation Rate |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 10% | `oracle` | +0.000133 ± 0.000060 | +0.001 dB | 1.000 | +0.000000 | 83.66 ms | 0.0% |
| 10% | `learned_utility` | +0.000044 ± 0.000051 | +0.001 dB | 0.243 | +0.000089 | 109.54 ms | 20.0% |
| 10% | `heuristic` | +0.000046 ± 0.000040 | +0.001 dB | 0.284 | +0.000087 | 102.47 ms | 0.0% |
| 10% | `error_influence` | +0.000070 ± 0.000030 | +0.001 dB | 0.571 | +0.000063 | 84.81 ms | 0.0% |
| 10% | `error_only` | +0.000051 ± 0.000023 | +0.000 dB | 0.441 | +0.000082 | 74.84 ms | 0.0% |
| 10% | `binary` | +0.000051 ± 0.000023 | +0.000 dB | 0.441 | +0.000082 | 74.81 ms | 0.0% |
| 10% | `random` | +0.000026 ± 0.000009 | +0.000 dB | 0.209 | +0.000107 | 91.43 ms | 0.0% |
| 20% | `oracle` | +0.000190 ± 0.000080 | +0.002 dB | 1.000 | +0.000000 | 153.37 ms | 0.0% |
| 20% | `learned_utility` | +0.000082 ± 0.000058 | +0.001 dB | 0.406 | +0.000108 | 174.96 ms | 0.0% |
| 20% | `heuristic` | +0.000085 ± 0.000056 | +0.001 dB | 0.407 | +0.000105 | 171.65 ms | 0.0% |
| 20% | `error_influence` | +0.000151 ± 0.000078 | +0.001 dB | 0.793 | +0.000039 | 159.85 ms | 0.0% |
| 20% | `error_only` | +0.000102 ± 0.000057 | +0.001 dB | 0.524 | +0.000087 | 142.14 ms | 0.0% |
| 20% | `binary` | +0.000102 ± 0.000057 | +0.001 dB | 0.523 | +0.000087 | 149.68 ms | 0.0% |
| 20% | `random` | +0.000050 ± 0.000014 | +0.000 dB | 0.286 | +0.000139 | 172.59 ms | 0.0% |
| 40% | `oracle` | +0.000234 ± 0.000092 | +0.002 dB | 1.000 | +0.000000 | 279.83 ms | 0.0% |
| 40% | `learned_utility` | +0.000133 ± 0.000078 | +0.001 dB | 0.523 | +0.000100 | 299.97 ms | 0.0% |
| 40% | `heuristic` | +0.000133 ± 0.000060 | +0.001 dB | 0.554 | +0.000101 | 296.09 ms | 0.0% |
| 40% | `error_influence` | +0.000203 ± 0.000088 | +0.002 dB | 0.835 | +0.000031 | 300.80 ms | 0.0% |
| 40% | `error_only` | +0.000170 ± 0.000086 | +0.001 dB | 0.713 | +0.000063 | 281.52 ms | 0.0% |
| 40% | `binary` | +0.000170 ± 0.000086 | +0.001 dB | 0.713 | +0.000063 | 282.42 ms | 0.0% |
| 40% | `random` | +0.000100 ± 0.000027 | +0.001 dB | 0.458 | +0.000133 | 315.78 ms | 0.0% |
| 60% | `oracle` | +0.000248 ± 0.000092 | +0.003 dB | 1.000 | +0.000000 | 419.99 ms | 0.0% |
| 60% | `learned_utility` | +0.000161 ± 0.000078 | +0.002 dB | 0.618 | +0.000087 | 371.87 ms | 0.0% |
| 60% | `heuristic` | +0.000185 ± 0.000081 | +0.002 dB | 0.707 | +0.000064 | 427.11 ms | 0.0% |
| 60% | `error_influence` | +0.000239 ± 0.000097 | +0.002 dB | 0.937 | +0.000009 | 434.32 ms | 0.0% |
| 60% | `error_only` | +0.000194 ± 0.000096 | +0.002 dB | 0.766 | +0.000055 | 399.84 ms | 0.0% |
| 60% | `binary` | +0.000178 ± 0.000087 | +0.001 dB | 0.709 | +0.000071 | 323.67 ms | 0.0% |
| 60% | `random` | +0.000148 ± 0.000036 | +0.001 dB | 0.635 | +0.000101 | 445.30 ms | 0.0% |
| 80% | `oracle` | +0.000251 ± 0.000092 | +0.002 dB | 1.000 | +0.000000 | 500.12 ms | 0.0% |
| 80% | `learned_utility` | +0.000179 ± 0.000080 | +0.002 dB | 0.700 | +0.000072 | 449.64 ms | 0.0% |
| 80% | `heuristic` | +0.000229 ± 0.000101 | +0.002 dB | 0.862 | +0.000023 | 545.96 ms | 0.0% |
| 80% | `error_influence` | +0.000256 ± 0.000093 | +0.002 dB | 1.026 | -0.000004 | 554.37 ms | 0.0% |
| 80% | `error_only` | +0.000227 ± 0.000097 | +0.002 dB | 0.894 | +0.000025 | 534.95 ms | 0.0% |
| 80% | `binary` | +0.000178 ± 0.000087 | +0.001 dB | 0.701 | +0.000074 | 336.55 ms | 0.0% |
| 80% | `random` | +0.000195 ± 0.000052 | +0.002 dB | 0.814 | +0.000056 | 567.67 ms | 0.0% |

## 4. Latency Breakdown Audit (ms)

| Budget | Policy | $T_{\text{feat}}$ | $T_{\text{pred}}$ | $T_{\text{select}}$ | $T_{\text{opt}}$ | $T_{\text{total}}$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 10% | `oracle` | 0.09 | 0.63 | 0.07 | 83.66 | 84.44 |
| 10% | `learned_utility` | 0.09 | 0.63 | 0.08 | 109.54 | 110.34 |
| 10% | `heuristic` | 0.09 | 0.63 | 0.09 | 102.47 | 103.29 |
| 10% | `error_influence` | 0.09 | 0.63 | 0.08 | 84.81 | 85.61 |
| 10% | `error_only` | 0.09 | 0.63 | 0.08 | 74.84 | 75.63 |
| 10% | `binary` | 0.09 | 0.63 | 0.17 | 74.81 | 75.69 |
| 10% | `random` | 0.09 | 0.63 | 0.13 | 91.43 | 92.28 |
| 20% | `oracle` | 0.09 | 0.63 | 0.07 | 153.37 | 154.16 |
| 20% | `learned_utility` | 0.09 | 0.63 | 0.08 | 174.96 | 175.76 |
| 20% | `heuristic` | 0.09 | 0.63 | 0.10 | 171.65 | 172.47 |
| 20% | `error_influence` | 0.09 | 0.63 | 0.09 | 159.85 | 160.65 |
| 20% | `error_only` | 0.09 | 0.63 | 0.07 | 142.14 | 142.93 |
| 20% | `binary` | 0.09 | 0.63 | 0.17 | 149.68 | 150.56 |
| 20% | `random` | 0.09 | 0.63 | 0.13 | 172.59 | 173.44 |
| 40% | `oracle` | 0.09 | 0.63 | 0.07 | 279.83 | 280.61 |
| 40% | `learned_utility` | 0.09 | 0.63 | 0.08 | 299.97 | 300.77 |
| 40% | `heuristic` | 0.09 | 0.63 | 0.09 | 296.09 | 296.90 |
| 40% | `error_influence` | 0.09 | 0.63 | 0.08 | 300.80 | 301.60 |
| 40% | `error_only` | 0.09 | 0.63 | 0.09 | 281.52 | 282.33 |
| 40% | `binary` | 0.09 | 0.63 | 0.17 | 282.42 | 283.30 |
| 40% | `random` | 0.09 | 0.63 | 0.13 | 315.78 | 316.63 |
| 60% | `oracle` | 0.09 | 0.63 | 0.07 | 419.99 | 420.77 |
| 60% | `learned_utility` | 0.09 | 0.63 | 0.08 | 371.87 | 372.67 |
| 60% | `heuristic` | 0.09 | 0.63 | 0.09 | 427.11 | 427.91 |
| 60% | `error_influence` | 0.09 | 0.63 | 0.09 | 434.32 | 435.12 |
| 60% | `error_only` | 0.09 | 0.63 | 0.08 | 399.84 | 400.64 |
| 60% | `binary` | 0.09 | 0.63 | 0.17 | 323.67 | 324.56 |
| 60% | `random` | 0.09 | 0.63 | 0.13 | 445.30 | 446.15 |
| 80% | `oracle` | 0.09 | 0.63 | 0.08 | 500.12 | 500.93 |
| 80% | `learned_utility` | 0.09 | 0.63 | 0.08 | 449.64 | 450.44 |
| 80% | `heuristic` | 0.09 | 0.63 | 0.09 | 545.96 | 546.76 |
| 80% | `error_influence` | 0.09 | 0.63 | 0.10 | 554.37 | 555.19 |
| 80% | `error_only` | 0.09 | 0.63 | 0.08 | 534.95 | 535.75 |
| 80% | `binary` | 0.09 | 0.63 | 0.17 | 336.55 | 337.44 |
| 80% | `random` | 0.09 | 0.63 | 0.13 | 567.67 | 568.52 |

## 5. Decision & Verification Verdict

> [!WARNING]
> **This report is DEPRECATED.** It was generated by the legacy `run_phase5_budget_benchmark.py` script
> which used non-unified budget semantics (budget_val for baselines vs budget_pred for learned).
> The authoritative Phase 5 report is at `results/phase5_budget_selection/phase5_report.md`.

- **RQ2 Status: INCONCLUSIVE.** Under equal-compute knapsack selection with unified budget semantics, learned utility does NOT statistically outperform heuristic or error-driven baselines (Wilcoxon $p > 0.9$, Cohen's $d < 0$). The original claim of RQ2 confirmation was based on non-unified budget semantics and is retracted.
- **Cost Control:** Hard budget limits are strictly enforced at *selection time* (scheduled cost). However, actual GPU group optimization latency significantly exceeds scheduled budgets due to fixed rasterization overhead.
- **Zero Leakage:** Complete decoupling between predictor and runtime scheduler verified — this finding remains valid.
