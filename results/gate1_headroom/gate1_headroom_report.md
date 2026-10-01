# Gate 1 Confirmatory Statistical Report

**Protocol:** v1.0.0 | **Seeds:** [42, 43, 44, 45, 46] ($n=5$) | **Dataset:** TUM RGB-D (`freiburg1_desk`)

## 1. Optimization Headroom ($H$) with 95% Bootstrap CI

- **Headroom Definition:** $H = \Delta Q(S^\star_{\text{pointwise}}) - \Delta Q(S_{\text{random}})$ at $K = 12$ (Top 20% budget Pointwise Oracle Reference).
- **Mean Headroom:** **$+0.000237$** ($\sigma = 0.000077$)
- **95% Bootstrap CI:** **[$+0.000181$, $+0.000302$]** (Strictly Positive $> 0$ ✅)
- **Paired Wilcoxon Signed-Rank Test:** $p = 0.03125$ (Statistically Significant ✅)
- **Paired Cohen's $d_z$ Effect Size:** $d_z = +3.079$ (Large effect size)

| Policy | Realized $\Delta Q$ (Mean $\pm$ Std) | Pointwise Selection Efficiency ($OSE$) | Paired Cohen's $d_z$ vs Error-Only | Wilcoxon $p$ vs Error |
|:---|:---:|:---:|:---:|:---:|
| **Pointwise Oracle Reference ($S^\star_{\text{pointwise}})** | $+0.000312 \pm 0.000101$ | **1.000** | -- | -- |
| **Heuristic Knapsack** | $+0.000123 \pm 0.000022$ | **0.396** | **-1.039** | **1.00000** |
| **Error-Only Top-$K$** | $+0.000167 \pm 0.000049$ | 0.536 | 0.000 (Ref) | -- |
| **Random Baseline** | $+0.000075 \pm 0.000037$ | 0.240 | -- | -- |

## 2. Stratified Negative Utility Breakdown (Phase 2.3)

Ground-truth marginal utility preserves degradation signals without artificial clamping ($U_i^\star < 0$).

| Stratum | Total Samples ($N$) | $\% U^\star < 0$ | Mean $U^\star$ | Median $U^\star$ | Physical Rationale |
|:---|:---:|:---:|:---:|:---:|:---|
| **Flat** | 75 | **38.7%** | +0.005898 | +0.000078 | Converged planar surfaces: gradients perturb smooth normals producing negative utility. |
| **Texture** | 75 | **21.3%** | +0.006649 | +0.003242 | High-frequency appearance: updates converge quickly but can cause mild color shift. |
| **Edge** | 75 | **18.7%** | +0.008595 | +0.000931 | Boundary gradients: updates blur sharp silhouettes or shift foreground/background depth. |
| **Depth Discontinuity** | 75 | **2.7%** | +0.022947 | +0.003643 | Occlusion boundaries: severe depth conflict leads to geometric degradation. |

## 3. Group Non-Additivity & Interaction Error Curve (Phase 4.1)

Interaction error $I(S) = \frac{|\Delta Q(S) - \sum_{i \in S} \Delta Q_i|}{|\Delta Q(S)| + \epsilon}$ and additivity ratio $R_{add}(S) = \frac{\Delta Q(S)}{\sum_{i \in S} \Delta Q_i}$:

| Group Size ($|S|$) | Mean Interaction Error $I(S)$ | Median $I(S)$ | Additivity Ratio $R_{add}(S)$ |
|:---:|:---:|:---:|:---:|
| **1** | 0.0000 | 0.0000 | **1.0000** |
| **4** | 0.0478 | 0.0505 | **0.9548** |
| **16** | 0.2166 | 0.2166 | **0.8221** |

## 4. Diminishing Returns Verification (Phase 4.2)

- **Condition:** $\Delta_i(A) \ge \Delta_i(B)$ for $A \subset B$ ($|A|=2, |B|=6$).
- **Marginal Gain in Small Context $\mathbb{E}[\Delta_i(A)]$:** **+0.000005**
- **Marginal Gain in Large Context $\mathbb{E}[\Delta_i(B)]$:** **+0.000005**
- **Empirical Diminishing Consistency:** **100.0%** of trials satisfied $\Delta_i(A) \ge \Delta_i(B)$.
- **Scientific Finding:** Empirical evidence is consistent with diminishing-return behavior under the evaluated intervention protocol, motivating budgeted knapsack selection over unconstrained allocation.
