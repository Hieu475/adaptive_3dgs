# Phase 12: Controlled Model Ablation & Hypothesis Test Results

Formal comparison of utility estimation formulations across 5 protocol seeds (42–46) on unseen test scenes:
- **M0**: Error Heuristic ($e_{rgb} + e_{depth}$)
- **M1**: Gradient Sensitivity Heuristic (Grad-Norm $\|
abla L\|$)
- **M2**: Current Baseline TwoHeadMLP (11D Local Observable State)
- **M3**: Two-Stage MLP + Positive Head (11D Local + $P(U^* > 0)$)
- **M4**: Two-Stage MLP + Positive Head + Global Context (11D Local + 12D Global Frame State)

## 1. Primary Model Ablation Table

| Model | Spearman $\rho$ | AUROC | NDCG@20 | OSE (Knapsack) | Realized $\Delta Q$ | $\Delta Q$ / ms | Neg. Selected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M0: Error Heuristic | 0.3552 ± 0.0000 | 0.6277 | 0.2434 | 0.2273 | 3.2566e-04 | 2.4632e-05 | 1.0 |
| M1: Grad-Norm Sensitivity | 0.5143 ± 0.0000 | 0.6995 | 0.4068 | 0.4857 | 6.9572e-04 | 5.2562e-05 | 2.0 |
| M2: Current TwoHeadMLP | 0.2500 ± 0.1949 | 0.5879 | 0.3768 | 0.2697 | 3.8638e-04 | 2.8889e-05 | 0.6 |
| M3: MLP + Positive Head | 0.2153 ± 0.1948 | 0.6502 | 0.3656 | 0.3652 | 5.2320e-04 | 3.8947e-05 | 2.2 |
| **M4: MLP + Positive + Global** | 0.0666 ± 0.0714 | 0.5877 | 0.3167 | 0.2768 | 3.9654e-04 | 2.9792e-05 | 4.8 |
| *Policy: Oracle-Positive + M4* | 0.0666 ± 0.0714 | 1.0000 | 0.3550 | 0.3444 | 4.9328e-04 | 3.6884e-05 | 0.0 |
| *Policy: Prob-Thresholded M4* | 0.0666 ± 0.0714 | 0.5877 | 0.3167 | 0.2768 | 3.9654e-04 | 2.9792e-05 | 4.8 |

## 2. Paired Hypothesis Tests & Effect Sizes

| Comparison | Metric | Mean Difference | Wilcoxon $p$-value | Paired $t$-test $p$ | Cohen's $d$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| M4 vs M1 (Grad-Norm) | `spearman_rho` | -0.4477 | 0.0625 | 0.0001 | -6.27 |
| M4 vs M1 (Grad-Norm) | `auroc` | -0.1118 | 0.0625 | 0.0254 | -1.56 |
| M4 vs M1 (Grad-Norm) | `ndcg_20` | -0.0901 | 0.1250 | 0.1435 | -0.81 |
| M4 vs M1 (Grad-Norm) | `ose` | -0.2089 | 0.0625 | 0.0137 | -1.88 |
| M4 vs M2 (Current TwoHeadMLP) | `spearman_rho` | -0.1834 | 0.1250 | 0.1249 | -0.87 |
| M4 vs M2 (Current TwoHeadMLP) | `auroc` | -0.0002 | 1.0000 | 0.9977 | -0.00 |
| M4 vs M2 (Current TwoHeadMLP) | `ndcg_20` | -0.0601 | 0.6250 | 0.4091 | -0.41 |
| M4 vs M2 (Current TwoHeadMLP) | `ose` | +0.0071 | 1.0000 | 0.9579 | +0.03 |
| M3 vs M2 (Positive Head Effect) | `spearman_rho` | -0.0347 | 1.0000 | 0.7662 | -0.14 |
| M3 vs M2 (Positive Head Effect) | `auroc` | +0.0624 | 0.4375 | 0.3060 | +0.52 |
| M3 vs M2 (Positive Head Effect) | `ndcg_20` | -0.0112 | 1.0000 | 0.8367 | -0.10 |
| M3 vs M2 (Positive Head Effect) | `ose` | +0.0955 | 0.8125 | 0.5017 | +0.33 |
| M1 vs M2 (Heuristic vs Current ML) | `spearman_rho` | +0.2643 | 0.0625 | 0.0387 | +1.36 |
| M1 vs M2 (Heuristic vs Current ML) | `auroc` | +0.1117 | 0.0625 | 0.0587 | +1.17 |
| M1 vs M2 (Heuristic vs Current ML) | `ndcg_20` | +0.0300 | 0.6250 | 0.3701 | +0.45 |
| M1 vs M2 (Heuristic vs Current ML) | `ose` | +0.2159 | 0.0625 | 0.1226 | +0.87 |
| M4 vs M0 (Full Model vs Error Heuristic) | `spearman_rho` | -0.2886 | 0.0625 | 0.0008 | -4.04 |
| M4 vs M0 (Full Model vs Error Heuristic) | `auroc` | -0.0400 | 0.3125 | 0.2811 | -0.56 |
| M4 vs M0 (Full Model vs Error Heuristic) | `ndcg_20` | +0.0734 | 0.1875 | 0.2131 | +0.66 |
| M4 vs M0 (Full Model vs Error Heuristic) | `ose` | +0.0495 | 0.4375 | 0.3761 | +0.44 |

## 3. Scientific Checkpoint Evaluation (Point 9 Decision)

> [!WARNING]
> **Outcome: Diagnostic Checkpoint Threshold (M4 <= M1)**
> - Model ranking ($\rho = 0.0666$, OSE = 0.2768) remains below or comparable to Grad-Norm sensitivity ($\rho = 0.5143$, OSE = 0.4857).
> - **Scientific Conclusion**: Local and frame-level observable states are only weakly predictive of counterfactual marginal return.
> - As advised in Point 9, rather than indefinitely inflating MLP capacity, the scientific conclusion is documented transparently.
