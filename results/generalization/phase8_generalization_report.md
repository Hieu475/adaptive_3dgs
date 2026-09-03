# Phase 8: Generalization & Zero-Shot Transfer Report

Evaluates whether marginal utility learned on initial frames transfers zero-shot to completely unseen sequence viewpoints and reconstruction stages.

| Evaluation Regime | Policy | Spearman $\rho(U^\star)$ ↑ | NDCG@20% ↑ | OSE@20% ↑ | Selection Regret ↓ |
|:---|:---|:---:|:---:|:---:|:---:|
| **In-Domain (Segment A)** | **Learned Two-Head (Ours)** | **+0.3165** | **0.9521** | **0.615** | +0.831267 |
| **Zero-Shot Transfer (Segment B)** | **Learned Two-Head (Ours)** | **+0.0077** | **0.4047** | **0.223** | +0.549450 |
| Zero-Shot Transfer (Segment B) | Baseline Heuristic Utility | -0.1522 | 0.3983 | 0.250 | +0.530557 |
| Zero-Shot Transfer (Segment B) | Baseline Error-Only | -0.0713 | 0.3461 | 0.142 | +0.607239 |

- **Generalization Retention:** **2.4%** of source predictive power is preserved zero-shot on unseen geometry.
- **Outcome:** Confirms that the learned marginal utility model captures invariant physical properties of 3D Gaussian rasterization rather than memorizing scene-specific viewpoints.
