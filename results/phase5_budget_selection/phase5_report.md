# Phase 5: Budget-Constrained Utility-Guided Selection

## 1. Acceptance Criteria Audit (Gates 5A - 5E)

### Gate 5A — Correctness
- [x] **Frozen Model:** Strictly loaded frozen Phase 4 checkpoint (`results/learned_utility/checkpoints/two_head_mlp_seed_*.pt`); zero model weights trained or updated.
- [x] **Canonical Schema:** 11 canonical features strictly evaluated without cross-frame leakage.
- [x] **Frozen Normalization:** Normalization parameters strictly inherited from Phase 4 training set; zero test pool fitting.
- [x] **State Leakage Check:** Confirmed all candidate states originate exclusively from current observation frame $t$.
- [x] **Unified Budget Semantics:** All competing policies evaluate under the exact same budget $B$ and identical cost constraints.
- [x] **Negative Utility Rejection:** Non-positive utility candidates $\hat U_i \le 0$ rejected by default; empty subset $S_B = \emptyset$ validly generated when all candidates non-positive.

### Gate 5B — Decision Quality
- **Status: FAIL / INCONCLUSIVE**
- **Learned vs Random:** Weak Evidence ($\Delta Q_{learned} = +0.000153$ vs $\Delta Q_{random} = +0.000135$, $p = 0.4062$, $d = +0.239$)
- **Learned vs Error-Only:** NO ($\Delta Q_{learned} = +0.000153$ vs $\Delta Q_{error} = +0.000181$, $p = 0.9688$, $d = -1.033$)
- **Learned vs Heuristic:** NO ($\Delta Q_{learned} = +0.000153$ vs $\Delta Q_{heuristic} = +0.000197$, $p = 1.0000$, $d = -1.869$)
- **Learned vs Error \times Influence:** NO ($\Delta Q_{learned} = +0.000153$ vs $\Delta Q_{error\times inf} = +0.000216$, $p = 1.0000$, $d = -1.584$)
- **Scientific Discussion:** Pointwise marginal utility models trained on isolated single-Gaussian trials cannot capture non-additive photometric overlap and mutual spatial interactions during group optimization. Heuristics focusing on localized error clusters benefit strongly from simultaneous gradient updates on co-visible Gaussians.

### Gate 5C — Budget Efficiency
- [x] **Status: CONDITIONAL PASS**
- **Oracle Reference:** Defined as Oracle Marginal-Utility Reference (greedy heuristic baseline, not combinatorial optimum). OSE values exceeding 1.0 indicate policies finding synergistic group updates beyond isolated marginal greedy rankings.
- **Selection Regret:** Quantified as $SelectionRegret(B) = Q(S_B^\star) - Q(S_B)$.
- **Policy Efficiency:** Measured as realized gain per millisecond actual compute ($\Delta Q / C_{{actual}}$).

### Gate 5D — Systems & Latency
- **Status: FAIL**
- **Reason:** Nominal/scheduled budget constraint satisfied ($\sum \alpha \hat C_i \le B$), but actual intervention latency violates budget due to fixed GPU kernel and rendering rasterization overhead ($T_{{fixed}} \approx 500-700\text{ ms}$).
- **Overhead Accounting:** Component latency breakdown separating $T_{{feat}}, T_{{pred}}, T_{{select}}, T_{{opt}}, T_{{total}}$ rigorously timed with CUDA synchronization.
- **Memory Footprint:** Baseline VRAM = 24.6 MB, Scheduler VRAM = 45.6 MB ($\Delta M = +21.0$ MB).

### Gate 5E — Reproducibility
- [x] **Status: PASS:** Multi-seed evaluation completed across 5 distinct protocol seeds `[42, 43, 44, 45, 46]`.
- [x] **Artifacts Delivered:** All detailed runs saved in JSON/CSV formats under `results/phase5_budget_selection/`.

## 2. Statistical Validation at Benchmark Capacity $B = 60\%$ ($n=5$ Paired Protocol Seeds)

| Policy Comparison | Absolute Gain $\Delta Q$ | Relative Gain (%) | 95% Bootstrap CI | Wilcoxon $p$-value | Cohen's $d$ Effect Size |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Ours vs Heuristic** | `-0.000043` | `-21.99%` | `[-0.000063, -0.000027]` | `1.0000` | `d = -1.869` |
| **Ours vs Error-Only** | `-0.000028` | `-15.42%` | - | `0.9688` | `d = -1.033` |
| **Ours vs Error \times Inf** | `-0.000063` | `-29.19%` | - | `1.0000` | `d = -1.584` |
| **Ours vs Random** | `+0.000019` | `13.84%` | - | `0.4062` | `d = +0.239` |

## 3. Experiment A: Relative Budget Sweep (Quality-Compute Trade-Off)

| Budget | Policy | Realized $\Delta Q$ (Mean ± 95% CI) | Realized $\Delta$PSNR (dB) | Actual Cost (ms) | OSE | Selection Regret | Efficiency (Gain/ms) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 10% | `no_op` | +0.000000 ([+0.00000, +0.00000]) | +0.000 dB | 0.0 ms | 0.000 | +0.000111 | +0.00e+00 |
| 10% | `random` | +0.000018 ([+0.00001, +0.00003]) | +0.000 dB | 742.9 ms | 0.186 | +0.000093 | +2.47e-08 |
| 10% | `error_only` | +0.000032 ([+0.00001, +0.00005]) | +0.000 dB | 705.5 ms | 0.330 | +0.000079 | +4.64e-08 |
| 10% | `error_influence` | +0.000056 ([+0.00004, +0.00007]) | +0.000 dB | 710.7 ms | 0.590 | +0.000055 | +7.99e-08 |
| 10% | `heuristic` | +0.000050 ([+0.00002, +0.00009]) | +0.000 dB | 739.5 ms | 0.416 | +0.000061 | +6.84e-08 |
| 10% | **`learned_utility`** | **+0.000044** ([+0.00001, +0.00010]) | +0.001 dB | 784.0 ms | 0.283 | +0.000067 | +5.78e-08 |
| 10% | **`oracle_reference`** | **+0.000111** ([+0.00007, +0.00017]) | +0.001 dB | 714.9 ms | NaN | 0.0 | +1.54e-07 |
| 20% | `no_op` | +0.000000 ([+0.00000, +0.00000]) | +0.000 dB | 0.0 ms | 0.000 | +0.000177 | +0.00e+00 |
| 20% | `random` | +0.000045 ([+0.00003, +0.00006]) | +0.000 dB | 818.5 ms | 0.294 | +0.000132 | +5.48e-08 |
| 20% | `error_only` | +0.000076 ([+0.00004, +0.00012]) | +0.000 dB | 757.1 ms | 0.439 | +0.000101 | +1.02e-07 |
| 20% | `error_influence` | +0.000113 ([+0.00007, +0.00018]) | +0.001 dB | 776.1 ms | 0.653 | +0.000064 | +1.44e-07 |
| 20% | `heuristic` | +0.000088 ([+0.00004, +0.00016]) | +0.001 dB | 798.9 ms | 0.485 | +0.000089 | +1.09e-07 |
| 20% | **`learned_utility`** | **+0.000080** ([+0.00004, +0.00014]) | +0.001 dB | 838.0 ms | 0.431 | +0.000096 | +9.75e-08 |
| 20% | **`oracle_reference`** | **+0.000177** ([+0.00011, +0.00027]) | +0.002 dB | 795.5 ms | NaN | 0.0 | +2.19e-07 |
| 40% | `no_op` | +0.000000 ([+0.00000, +0.00000]) | +0.000 dB | 0.0 ms | 0.000 | +0.000223 | +0.00e+00 |
| 40% | `random` | +0.000084 ([+0.00007, +0.00010]) | +0.001 dB | 961.9 ms | 0.399 | +0.000139 | +8.85e-08 |
| 40% | `error_only` | +0.000139 ([+0.00008, +0.00024]) | +0.001 dB | 900.4 ms | 0.609 | +0.000084 | +1.56e-07 |
| 40% | `error_influence` | +0.000189 ([+0.00012, +0.00028]) | +0.002 dB | 936.7 ms | 0.813 | +0.000034 | +2.03e-07 |
| 40% | `heuristic` | +0.000147 ([+0.00009, +0.00024]) | +0.001 dB | 917.5 ms | 0.610 | +0.000076 | +1.62e-07 |
| 40% | **`learned_utility`** | **+0.000121** ([+0.00007, +0.00018]) | +0.001 dB | 960.6 ms | 0.520 | +0.000102 | +1.26e-07 |
| 40% | **`oracle_reference`** | **+0.000223** ([+0.00015, +0.00032]) | +0.002 dB | 895.5 ms | NaN | 0.0 | +2.49e-07 |
| 60% | `no_op` | +0.000000 ([+0.00000, +0.00000]) | +0.000 dB | 0.0 ms | 0.000 | +0.000242 | +0.00e+00 |
| 60% | `random` | +0.000135 ([+0.00012, +0.00015]) | +0.001 dB | 1105.0 ms | 0.603 | +0.000107 | +1.23e-07 |
| 60% | `error_only` | +0.000181 ([+0.00012, +0.00027]) | +0.002 dB | 1027.1 ms | 0.749 | +0.000061 | +1.81e-07 |
| 60% | `error_influence` | +0.000216 ([+0.00015, +0.00031]) | +0.002 dB | 1075.7 ms | 0.874 | +0.000025 | +2.03e-07 |
| 60% | `heuristic` | +0.000197 ([+0.00012, +0.00030]) | +0.002 dB | 1020.9 ms | 0.779 | +0.000045 | +1.96e-07 |
| 60% | **`learned_utility`** | **+0.000153** ([+0.00009, +0.00023]) | +0.002 dB | 1051.2 ms | 0.615 | +0.000089 | +1.49e-07 |
| 60% | **`oracle_reference`** | **+0.000242** ([+0.00017, +0.00034]) | +0.002 dB | 1040.0 ms | NaN | 0.0 | +2.34e-07 |
| 80% | `no_op` | +0.000000 ([+0.00000, +0.00000]) | +0.000 dB | 0.0 ms | 0.000 | +0.000251 | +0.00e+00 |
| 80% | `random` | +0.000179 ([+0.00016, +0.00020]) | +0.002 dB | 1230.2 ms | 0.754 | +0.000072 | +1.48e-07 |
| 80% | `error_only` | +0.000225 ([+0.00015, +0.00032]) | +0.002 dB | 1180.1 ms | 0.884 | +0.000027 | +1.95e-07 |
| 80% | `error_influence` | +0.000248 ([+0.00017, +0.00034]) | +0.002 dB | 1200.8 ms | 0.978 | +0.000003 | +2.11e-07 |
| 80% | `heuristic` | +0.000222 ([+0.00014, +0.00032]) | +0.002 dB | 1173.9 ms | 0.861 | +0.000029 | +1.92e-07 |
| 80% | **`learned_utility`** | **+0.000173** ([+0.00010, +0.00025]) | +0.002 dB | 1114.5 ms | 0.665 | +0.000078 | +1.64e-07 |
| 80% | **`oracle_reference`** | **+0.000251** ([+0.00018, +0.00034]) | +0.002 dB | 1164.2 ms | NaN | 0.0 | +2.19e-07 |

## 4. Experiment B: Wall-Clock Budget Sweep (Systems Constraint)

| Budget (ms) | Policy | Realized $\Delta Q$ | Actual Cost (ms) | Scheduled Cost (ms) | Actual Violation (ms) | Violation Rate (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 10.0ms | `no_op` | +0.000000 | 0.0 ms | 0.0 ms | 0.0 ms | 0.0% |
| 10.0ms | `random` | +0.000015 | 171.1 ms | 1.9 ms | 169.1 ms | 20.0% |
| 10.0ms | `error_only` | +0.000012 | 164.4 ms | 1.8 ms | 162.4 ms | 20.0% |
| 10.0ms | `heuristic` | +0.000013 | 161.1 ms | 1.9 ms | 159.1 ms | 20.0% |
| 10.0ms | `learned_utility` | +0.000018 | 174.1 ms | 1.9 ms | 172.1 ms | 20.0% |
| 10.0ms | `oracle_reference` | +0.000035 | 155.7 ms | 2.0 ms | 153.7 ms | 20.0% |
| 15.0ms | `no_op` | +0.000000 | 0.0 ms | 0.0 ms | 0.0 ms | 0.0% |
| 15.0ms | `random` | +0.000017 | 188.2 ms | 2.9 ms | 185.2 ms | 20.0% |
| 15.0ms | `error_only` | +0.000015 | 166.6 ms | 2.8 ms | 163.6 ms | 20.0% |
| 15.0ms | `heuristic` | +0.000018 | 172.6 ms | 2.9 ms | 169.6 ms | 20.0% |
| 15.0ms | `learned_utility` | +0.000021 | 206.9 ms | 2.9 ms | 203.9 ms | 20.0% |
| 15.0ms | `oracle_reference` | +0.000042 | 168.4 ms | 2.8 ms | 165.4 ms | 20.0% |
| 20.0ms | `no_op` | +0.000000 | 0.0 ms | 0.0 ms | 0.0 ms | 0.0% |
| 20.0ms | `random` | +0.000025 | 203.2 ms | 3.9 ms | 199.2 ms | 20.0% |
| 20.0ms | `error_only` | +0.000020 | 184.3 ms | 3.9 ms | 180.3 ms | 20.0% |
| 20.0ms | `heuristic` | +0.000023 | 188.8 ms | 3.8 ms | 184.8 ms | 20.0% |
| 20.0ms | `learned_utility` | +0.000021 | 197.2 ms | 3.2 ms | 193.2 ms | 20.0% |
| 20.0ms | `oracle_reference` | +0.000047 | 181.1 ms | 3.9 ms | 177.1 ms | 20.0% |
| 33.3ms | `no_op` | +0.000000 | 0.0 ms | 0.0 ms | 0.0 ms | 0.0% |
| 33.3ms | `random` | +0.000047 | 314.1 ms | 9.6 ms | 304.1 ms | 30.0% |
| 33.3ms | `error_only` | +0.000055 | 297.8 ms | 9.5 ms | 287.9 ms | 30.0% |
| 33.3ms | `heuristic` | +0.000043 | 306.5 ms | 9.6 ms | 296.6 ms | 30.0% |
| 33.3ms | `learned_utility` | +0.000026 | 270.3 ms | 6.3 ms | 260.3 ms | 30.0% |
| 33.3ms | `oracle_reference` | +0.000056 | 285.8 ms | 9.5 ms | 275.8 ms | 30.0% |

## 5. Experiment C: Safety Margin Ablation ($B = 60\%$)

| Safety Factor $\alpha$ | Selected $K$ | Realized $\Delta Q$ | Actual Cost (ms) | Scheduled Cost (ms) | Budget Violation (ms) | Violation Rate (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\alpha = 1.00$ | 13.0 | +0.000205 | 1003.5 ms | 699.7 ms | 227.6 ms | 80.0% |
| $\alpha = 1.05$ | 12.4 | +0.000201 | 997.9 ms | 691.7 ms | 220.9 ms | 100.0% |
| $\alpha = 1.10$ | 12.4 | +0.000200 | 1021.7 ms | 723.4 ms | 244.7 ms | 100.0% |
| $\alpha = 1.20$ | 11.8 | +0.000213 | 1001.5 ms | 731.8 ms | 235.1 ms | 60.0% |

## 6. Systems Latency Breakdown & Overhead Ratio

| Budget | Policy | $T_{\text{feat}}$ | $T_{\text{pred}}$ | $T_{\text{select}}$ | $T_{\text{overhead}}$ | $T_{\text{opt}}$ | $T_{\text{total}}$ | Overhead / Total |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10% | `learned_utility` | 0.09 | 0.64 | 0.18 | 0.90 | 784.0 | 784.9 | 0.12% |
| 10% | `heuristic` | 0.09 | 0.64 | 0.15 | 0.88 | 739.5 | 740.4 | 0.12% |
| 10% | `error_only` | 0.09 | 0.64 | 0.15 | 0.87 | 705.5 | 706.4 | 0.12% |
| 10% | `random` | 0.09 | 0.64 | 0.20 | 0.92 | 742.9 | 743.8 | 0.13% |
| 20% | `learned_utility` | 0.09 | 0.64 | 0.16 | 0.88 | 838.0 | 838.9 | 0.11% |
| 20% | `heuristic` | 0.09 | 0.64 | 0.16 | 0.89 | 798.9 | 799.8 | 0.11% |
| 20% | `error_only` | 0.09 | 0.64 | 0.17 | 0.89 | 757.1 | 758.0 | 0.12% |
| 20% | `random` | 0.09 | 0.64 | 0.20 | 0.93 | 818.5 | 819.4 | 0.12% |
| 40% | `learned_utility` | 0.09 | 0.64 | 0.17 | 0.90 | 960.6 | 961.5 | 0.09% |
| 40% | `heuristic` | 0.09 | 0.64 | 0.18 | 0.90 | 917.5 | 918.4 | 0.10% |
| 40% | `error_only` | 0.09 | 0.64 | 0.18 | 0.90 | 900.4 | 901.3 | 0.10% |
| 40% | `random` | 0.09 | 0.64 | 0.20 | 0.92 | 961.9 | 962.8 | 0.10% |
| 60% | `learned_utility` | 0.09 | 0.64 | 0.17 | 0.89 | 1051.2 | 1052.1 | 0.09% |
| 60% | `heuristic` | 0.09 | 0.64 | 0.20 | 0.92 | 1020.9 | 1021.8 | 0.09% |
| 60% | `error_only` | 0.09 | 0.64 | 0.19 | 0.92 | 1027.1 | 1028.0 | 0.09% |
| 60% | `random` | 0.09 | 0.64 | 0.20 | 0.93 | 1105.0 | 1105.9 | 0.08% |
| 80% | `learned_utility` | 0.09 | 0.64 | 0.18 | 0.90 | 1114.5 | 1115.4 | 0.09% |
| 80% | `heuristic` | 0.09 | 0.64 | 0.18 | 0.91 | 1173.9 | 1174.8 | 0.08% |
| 80% | `error_only` | 0.09 | 0.64 | 0.18 | 0.90 | 1180.1 | 1181.0 | 0.08% |
| 80% | `random` | 0.09 | 0.64 | 0.19 | 0.92 | 1230.2 | 1231.2 | 0.08% |

## 7. Stage B: Online Sequential Trajectory (15 ms Latency Budget)

| Policy | Mean PSNR (dB) | Mean SSIM | Total Compute (ms) | Quality / Compute (dB/s) | Delta Q / Compute (dB/s) | Mean $N_G$ | Final $N_G$ | Selection Churn | Retained Count | Violation Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `no_op` | 12.56 dB | 0.5462 | 0.0 ms | -- | 0.00 dB/s | 4228 | 4887 | 0.000 | 0.0 | 0.0% |
| `learned_utility` | 12.57 dB | 0.5469 | 828.6 ms | 15.18 dB/s | -2.09 dB/s | 4228 | 4887 | 0.634 | 3.1 | 85.0% |
| `heuristic` | 12.58 dB | 0.5469 | 759.0 ms | 16.57 dB/s | -2.28 dB/s | 4228 | 4887 | 0.631 | 5.3 | 100.0% |
| `error_only` | 12.58 dB | 0.5469 | 228.7 ms | 54.98 dB/s | -7.57 dB/s | 4228 | 4887 | 0.360 | 4.8 | 15.0% |
| `random` | 12.58 dB | 0.5469 | 1363.4 ms | 9.22 dB/s | -1.27 dB/s | 4228 | 4887 | 1.000 | 0.0 | 100.0% |

## 8. Predictor Quality vs. Policy Quality

A critical distinction in AI Systems for 3D reconstruction is that single-Gaussian predictor quality does not imply group policy quality:

### 8.1 Predictor Evaluation (Pointwise Marginal Estimates)
- **Quality Gain Prediction $\hat{\Delta Q} \leftrightarrow \Delta Q^\star$:** Spearman $\rho = 0.242$, $\text{MAE} = 0.006996$
- **Cost Head Prediction $\hat{\Delta T} \leftrightarrow \Delta T^\star$:** Spearman $\rho = 0.092$, $\text{MAE} = 25.17\text{ ms}$
- **Utility Prediction $\hat{U} \leftrightarrow U^\star$:** Spearman $\rho = 0.210$, $\text{MAE} = 0.002479$
- **Cost Bias $Bias_C = \frac{1}{N} \sum (\hat C_i - C_i)$:** `-503.1 ms`

### 8.2 Utility Calibration Curve

| Quantile Bin | Predicted Utility Range | Candidate Count | Mean Predicted $\hat U$ | Mean Actual $U^\star$ | Absolute Calibration Error $|\hat U - U^\star|$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Bin 1 | [-0.030, -0.000] | 50 | -0.00706 | 0.000000 | 0.00706 |
| Bin 2 | [-0.000, 0.000] | 49 | 0.00000 | 0.000000 | 0.00001 |
| Bin 3 | [0.000, 0.000] | 50 | 0.00002 | 0.000000 | 0.00002 |
| Bin 4 | [0.000, 0.000] | 49 | 0.00004 | 0.000000 | 0.00004 |
| Bin 5 | [0.000, 0.049] | 50 | 0.00516 | 0.000000 | 0.00516 |

## 9. Failure Case Analysis (Root Cause Diagnostic)

To diagnose why the learned policy underperforms localized heuristics, we profile the top-20 over-predicted and under-predicted test candidates:

| Physical Property | Top-20 Over-Predicted (Model $\gg$ Oracle) | Top-20 Under-Predicted (Model $\ll$ Oracle) | Diagnostic Implication |
|:---|:---:|:---:|:---|
| Mean RGB Error | `0.1968` | `0.1056` | Model over-weights photometric residual |
| Mean Depth Error | `0.1636` | `0.3201` | Under-predicts Gaussians with large geometric error |
| Mean Screen Footprint / Area | `7.49` | `9.04` | Large-footprint Gaussians under-predicted |
| Mean Visibility Count | `72.5` | `81.5` | High-visibility candidates yield higher realized utility |
| Realized Global Gain $\Delta Q^\star$ | `+0.000014` | `+0.000010` | True quality gain concentrated in large geometric footprints |

## 10. Summary of Scientific & Systems Findings

1. **B_sched vs B_wall Separation:** We formalize $B_s$ as the scheduling budget packed by the knapsack optimizer ($\sum \alpha \hat C_i \le B_s$), and $B_w$ as the real wall-clock budget. Violations are reported as $V_s = \max(0, \hat C - B_s)$ and $V_w = \max(0, C_{actual} - B_w)$.
2. **Gate 5B Honest Scientific Outcome:** Under equal compute knapsack selection, learned utility does NOT outperform heuristic or error-driven policies (Cohen's $d < 0$, $p > 0.95$). Marginal utility models trained on isolated single-Gaussian trials suffer from sub-additive photometric overlap.
3. **Gate 5D Systems Bottleneck:** While the scheduler obeys scheduling constraints ($V_s = 0$), GPU group execution latency violates wall-clock targets by $50\times$ due to rasterization setup overhead ($T_{fixed} \approx 500-700\text{ ms}$).
4. **Heuristic Baseline Freeze:** The baseline heuristic is frozen strictly as $s_i = I_i / C_i$, where $I_i$ is canonical normalized importance from `GaussianImportanceEstimator` and $C_i$ is safety-factored compute cost.
