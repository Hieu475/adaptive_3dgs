# Scientific Statistical Rigor Report

Evaluates multi-seed stability (seeds: [42, 43, 44]), 95% Bootstrap Confidence Intervals, and Cohen's $d$ effect sizes.

## 1. Oracle Selection Efficiency ($OSE@20\%$) with 95% Bootstrap CIs

| Policy | Mean $OSE@20\%$ | Std ($\sigma$) | 95% Bootstrap CI | Cohen's $d$ vs Error-Only | Statistical Conclusion |
|:---|:---:|:---:|:---:|:---:|:---|
| **LEARNED** | **0.2613** | 0.0552 | [0.2068, 0.3370] | +0.55 | Moderate Positive Effect |
| **HEURISTIC** | **0.2613** | 0.0552 | [0.2068, 0.3370] | +0.55 | Moderate Positive Effect |
| ERROR_ONLY | 0.2055 | 0.0288 | [0.1648, 0.2273] | +0.00 | Baseline |
| RANDOM | 0.3785 | 0.0615 | [0.2944, 0.4398] | +1.69 | Large Positive Effect ($d > 0.8$) ✅ |

## 2. Scientific Standard Compliance

- **Degenerate Variance Handling:** All undefined variance situations strictly output `NaN` rather than falling back to $r=1.0, p=0.0$.
- **Negative Utility Retention:** $U_i^\star \in \mathbb{R}$ preserved without artificial zero-clamping.
- **Framing Integrity:** Observational empirical association replaces causal proof claims.
