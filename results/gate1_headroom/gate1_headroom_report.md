# Gate 1 Confirmatory Statistical Report

**Protocol:** v1.0.0 | **Seeds:** [42, 43, 44, 45, 46] ($n=5$) | **Dataset:** TUM RGB-D (`freiburg1_desk`)

## 1. Optimization Headroom ($H$) with 95% Bootstrap CI

- **Headroom Definition:** $H = \Delta Q(S^\star_K) - \Delta Q(S_{\text{random}})$ at $K = 12$ (Top 20% budget).
- **Mean Headroom:** **$+0.000148$** ($\sigma = 0.000084$)
- **95% Bootstrap CI:** **[$+0.000084$, $+0.000214$]** (Strictly Positive $> 0$ ✅)
- **Paired Wilcoxon Signed-Rank Test:** $p = 0.03125$ (Statistically Significant ✅)
- **Cohen's $d$ Effect Size:** $d = +1.768$ (Large effect size)

| Policy | Realized $\Delta Q$ (Mean $\pm$ Std) | Oracle Selection Efficiency ($OSE$) | Cohen's $d$ vs Error-Only | Wilcoxon $p$ vs Error |
|:---|:---:|:---:|:---:|:---:|
| **Oracle Reference ($S^\star$)** | $+0.000253 \pm 0.000120$ | **1.000** | -- | -- |
| **Heuristic Knapsack** | $+0.000150 \pm 0.000014$ | **0.595** | **+0.816** | **0.09375** |
| **Error-Only Top-$K$** | $+0.000081 \pm 0.000074$ | 0.321 | 0.000 (Ref) | -- |
| **Random Baseline** | $+0.000105 \pm 0.000050$ | 0.413 | -- | -- |

## 2. Stratified Negative Utility Breakdown (Phase 2.3)

Ground-truth marginal utility preserves degradation signals without artificial clamping ($U_i^\star < 0$).

| Stratum | Total Samples ($N$) | $\% U^\star < 0$ | Mean $U^\star$ | Median $U^\star$ | Physical Rationale |
|:---|:---:|:---:|:---:|:---:|:---|
| **Flat** | 75 | **14.7%** | +0.000241 | +0.000113 | Converged planar surfaces: gradients perturb smooth normals producing negative utility. |
| **Texture** | 75 | **8.0%** | +0.000137 | +0.000040 | High-frequency appearance: updates converge quickly but can cause mild color shift. |
| **Edge** | 75 | **14.7%** | +0.000230 | +0.000061 | Boundary gradients: updates blur sharp silhouettes or shift foreground/background depth. |
| **Depth Discontinuity** | 75 | **30.7%** | +0.000152 | +0.000023 | Occlusion boundaries: severe depth conflict leads to geometric degradation. |

## 3. Group Non-Additivity & Interaction Error Curve (Phase 4.1)

Interaction error $I(S) = \frac{|\Delta Q(S) - \sum_{i \in S} \Delta Q_i|}{|\Delta Q(S)| + \epsilon}$ and additivity ratio $R_{add}(S) = \frac{\Delta Q(S)}{\sum_{i \in S} \Delta Q_i}$:

| Group Size ($|S|$) | Mean Interaction Error $I(S)$ | Median $I(S)$ | Additivity Ratio $R_{add}(S)$ |
|:---:|:---:|:---:|:---:|
| **1** | 0.0000 | 0.0000 | **1.0000** |
| **2** | 3.9069 | 2.4657 | **0.4762** |
| **4** | 9.5657 | 9.6077 | **-2.6478** |
| **8** | 45.1536 | 44.3902 | **0.0209** |
| **16** | 175.8208 | 175.8208 | **0.0054** |
| **32** | 361.0413 | 361.0413 | **0.0024** |

## 4. Diminishing Returns Verification (Phase 4.2)

- **Condition:** $\Delta_i(A) \ge \Delta_i(B)$ for $A \subset B$ ($|A|=2, |B|=6$).
- **Marginal Gain in Small Context $\mathbb{E}[\Delta_i(A)]$:** **+0.000007**
- **Marginal Gain in Large Context $\mathbb{E}[\Delta_i(B)]$:** **+0.000007**
- **Empirical Diminishing Consistency:** **100.0%** of trials satisfied $\Delta_i(A) \ge \Delta_i(B)$.
- **Scientific Finding:** Alpha-compositing induces an interaction structure consistent with diminishing-return behavior under the evaluated intervention protocol, mathematically motivating budget knapsack selection.
