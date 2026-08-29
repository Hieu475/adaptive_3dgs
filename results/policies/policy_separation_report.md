# Policy Separation & Jaccard Overlap Report

Evaluated with $N=392$ Gaussians at $K=20\%$ active selection budget.

| Policy | **Random** | **Error-Only** | **Error × Influence** | **Top-K Importance** | **Ours (Knapsack U/C)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random** | 1.000 | 0.091 | 0.091 | 0.091 | 0.061 |
| **Error-Only** | 0.091 | 1.000 | 1.000 | 0.000 | 0.592 |
| **Error × Influence** | 0.091 | 1.000 | 1.000 | 0.000 | 0.592 |
| **Top-K Importance** | 0.091 | 0.000 | 0.000 | 1.000 | 0.130 |
| **Ours (Knapsack U/C)** | 0.061 | 0.592 | 0.592 | 0.130 | 1.000 |

### Distinction Diagnostics
- **Cost Variation ($CV = \sigma_C / \mu_C$):** 1.0484
- **Spearman $\rho(U, E)$:** -0.2423
- **Spearman $\rho(U/C, E)$:** +0.2469
- **Jaccard(Ours, Top-K):** 0.130
