# Gate 1: Ground-Truth Oracle Validation Report

Generated on 2026-09-03 09:40:38

## 1. Executive Summary & Gate 1 Verification

**Gate 1 Status:** ✅ PASSED (Statistically Valid Oracle Ground Truth)

- **Rank Correlation with Geometry Stratification:** $\rho = +0.2090$
- **Rank Correlation with Importance Stratification:** $\rho = -0.2391$
- **Rank Correlation with Random Visible:** $\rho = +0.0947$
- **Oracle Repeat Stability ($CV$):** Mean $CV = 0.051$ (100.0% stable trials)

## 2. Multi-Population Ranking & Correlation Table

| Sampling Population | Spearman $\rho(U, U_{oracle})$ | Spearman $\rho(I, \Delta Q)$ | Overlap@10% | Overlap@20% | Gain Ratio@20% | Regret@20% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **geometry_stratified** | **+0.2090** | +0.2677 | 0.0% | 25.0% | 0.7570 | 0.2430 |
| **importance_stratified** | **-0.2391** | -0.2458 | 0.0% | 25.0% | 0.4087 | 0.5913 |
| **random_visible** | **+0.0947** | +0.1737 | 50.0% | 25.0% | 0.7725 | 0.2275 |
| **uniform_visible** | **+0.0226** | +0.0549 | 0.0% | 25.0% | 0.4192 | 0.5808 |

## 3. Repeat Measurement Stability Analysis ($n=3–5$ Repeated Trials)

Evaluated across 8 candidate Gaussians over 3 identical initial trials:

| Gaussian ID | Mean Utility $\mu_U$ | Std $\sigma_U$ | $CV = \sigma / (|\mu| + \epsilon)$ | Mean Time (ms) | Stable ($CV \le 0.35$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 295 | -0.0000 | 0.0000 | 0.033 | 24.82 ms | Yes |
| 299 | -0.0002 | 0.0000 | 0.036 | 27.86 ms | Yes |
| 441 | 0.0003 | 0.0000 | 0.113 | 26.91 ms | Yes |
| 108 | 0.0002 | 0.0000 | 0.009 | 26.92 ms | Yes |
| 242 | 0.0006 | 0.0000 | 0.024 | 23.93 ms | Yes |
| 240 | 0.0002 | 0.0000 | 0.031 | 30.05 ms | Yes |
| 259 | 0.0002 | 0.0000 | 0.105 | 30.81 ms | Yes |
| 81 | 0.0002 | 0.0000 | 0.056 | 28.02 ms | Yes |

**Mean Population $CV$:** 0.0510 (Threshold $\le 0.35$)

## 4. Group Size Scaling & Non-Additivity ($g \in \{1, 4, 16\}$)

| Group Size ($g$) | Number of Groups | Spearman $\rho(U, U_{oracle})$ | Gain Ratio@20% | Mean Group Time (ms) |
|:---:|:---:|:---:|:---:|:---:|
| group_size_4 | 20 | -0.0674 | 0.7730 | — |
| group_size_16 | 20 | +0.1734 | 1.3344 | — |

## 5. Geometry-Stratified Breakdown Analysis

| Geometric Stratum | Count | Mean $\Delta$PSNR (dB) | Mean $\Delta$Depth Gain (m) | Mean $\Delta$Loss | Mean Utility |
|:---|:---:|:---:|:---:|:---:|:---:|
| **flat** | 5 | 0.2193 dB | 0.0023 m | 0.0025 | 0.0005 |
| **edge** | 5 | 0.0311 dB | 0.0079 m | 0.0025 | 0.0002 |
| **texture** | 5 | 0.1721 dB | 0.0019 m | 0.0035 | 0.0005 |
| **depth_discontinuity** | 5 | 0.1509 dB | 0.0047 m | 0.0044 | 0.0006 |
