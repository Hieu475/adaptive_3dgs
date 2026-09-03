# Gate 1 Confirmatory Statistical Report

**Protocol:** v1.0.0 | **Seeds:** [42, 43, 44, 45, 46] ($n=5$) | **Dataset:** TUM RGB-D (`freiburg1_desk`)

## 1. Optimization Headroom ($H$) with 95% Bootstrap CI

- **Headroom Definition:** $H = \Delta Q(S^\star_K) - \Delta Q(S_{\text{random}})$ at $K = 12$ (Top 20% budget).
- **Mean Headroom:** **$+0.000185$** ($\sigma = 0.000158$)
- **95% Bootstrap CI:** **[$+0.000073$, $+0.000327$]** (Strictly Positive $> 0$ ✅)
- **Paired Wilcoxon Signed-Rank Test:** $p = 0.03125$ (Statistically Significant ✅)
- **Cohen's $d$ Effect Size:** $d = +1.169$ (Large effect size)

| Policy | Realized $\Delta Q$ (Mean $\pm$ Std) | Oracle Selection Efficiency ($OSE$) | Cohen's $d$ vs Error-Only | Wilcoxon $p$ vs Error |
|:---|:---:|:---:|:---:|:---:|
| **Oracle Reference ($S^\star$)** | $+0.000429 \pm 0.000155$ | **1.000** | -- | -- |
| **Heuristic Knapsack** | $+0.000343 \pm 0.000143$ | **0.799** | **+1.055** | **0.03125** |
| **Error-Only Top-$K$** | $+0.000247 \pm 0.000166$ | 0.576 | 0.000 (Ref) | -- |
| **Random Baseline** | $+0.000244 \pm 0.000048$ | 0.569 | -- | -- |

## 2. Stratified Negative Utility Breakdown (Phase 2.3)

Ground-truth marginal utility preserves degradation signals without artificial clamping ($U_i^\star < 0$).

| Stratum | Total Samples ($N$) | $\% U^\star < 0$ | Mean $U^\star$ | Median $U^\star$ | Physical Rationale |
|:---|:---:|:---:|:---:|:---:|:---|
| **Flat** | 75 | **10.7%** | +0.000227 | +0.000177 | Converged planar surfaces: gradients perturb smooth normals producing negative utility. |
| **Texture** | 75 | **8.0%** | +0.000291 | +0.000178 | High-frequency appearance: updates converge quickly but can cause mild color shift. |
| **Edge** | 75 | **12.0%** | +0.000286 | +0.000140 | Boundary gradients: updates blur sharp silhouettes or shift foreground/background depth. |
| **Depth Discontinuity** | 75 | **9.3%** | +0.000453 | +0.000090 | Occlusion boundaries: severe depth conflict leads to geometric degradation. |

## 3. Group Non-Additivity & Interaction Error Curve (Phase 4.1)

Interaction error $I(S) = \frac{|\Delta Q(S) - \sum_{i \in S} \Delta Q_i|}{|\Delta Q(S)| + \epsilon}$ and additivity ratio $R_{add}(S) = \frac{\Delta Q(S)}{\sum_{i \in S} \Delta Q_i}$:

| Group Size ($|S|$) | Mean Interaction Error $I(S)$ | Median $I(S)$ | Additivity Ratio $R_{add}(S)$ |
|:---:|:---:|:---:|:---:|
| **1** | 0.0000 | 0.0000 | **1.0000** |
| **2** | 1.1613 | 1.1753 | **0.5552** |
| **4** | 19.7341 | 19.6211 | **0.0704** |
| **8** | 63.0358 | 54.1660 | **0.0162** |
| **16** | 129.6326 | 129.6326 | **0.0082** |
| **32** | 365.5030 | 365.5030 | **0.0027** |

## 4. Diminishing Returns Verification (Phase 4.2)

- **Condition:** $\Delta_i(A) \ge \Delta_i(B)$ for $A \subset B$ ($|A|=2, |B|=6$).
- **Marginal Gain in Small Context $\mathbb{E}[\Delta_i(A)]$:** **+0.000017**
- **Marginal Gain in Large Context $\mathbb{E}[\Delta_i(B)]$:** **+0.000017**
- **Empirical Diminishing Consistency:** **90.0%** of trials satisfied $\Delta_i(A) \ge \Delta_i(B)$.
- **Scientific Finding:** Alpha-compositing induces an interaction structure consistent with diminishing-return behavior under the evaluated intervention protocol, mathematically motivating budget knapsack selection.
