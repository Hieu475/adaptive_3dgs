# Two-Head Learned Utility Model & Feature Ablation Report

## 1. Feature Ablation Study (V0 to V7)

| Feature Version | Inputs | Spearman $\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V0: Error Only** | 2 | +0.4766 | 0.0% | 28.6% | 0.5482 | 0.4518 |
| **V1: + Visibility** | 3 | +0.4555 | 0.0% | 14.3% | 0.3583 | 0.6417 |
| **V2: + Influence** | 4 | +0.3747 | 0.0% | 14.3% | 0.2410 | 0.7590 |
| **V3: + Temporal Drift** | 5 | +0.3212 | 0.0% | 14.3% | 0.3146 | 0.6854 |
| **V4: + Uncertainty** | 6 | +0.3855 | 0.0% | 14.3% | 0.2771 | 0.7229 |
| **V5: + Gradient Norm** | 7 | +0.1704 | 0.0% | 14.3% | 0.2170 | 0.7830 |
| **V6: + Projected Area** | 8 | +0.1133 | 0.0% | 14.3% | 0.4615 | 0.5385 |
| **V7: Full State** | 10 | +0.5044 | 0.0% | 14.3% | 0.3237 | 0.6763 |

## 2. Direct Comparison: Regression vs Ranking vs Oracle (Section XXII)

| Method | Spearman $\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ | Inference Cost ($\mu$s/G) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Two-Head Regression (Smooth-L1) | +0.1179 | 0.0% | 14.3% | 0.3366 | 0.6634 | 0.09 $\mu$s |
| **Two-Head + Pairwise Ranking (Ours)** | +0.4302 | 0.0% | 42.9% | 0.6963 | 0.3037 | 0.09 $\mu$s |
| Oracle Upper Bound | +1.0000 | 100.0% | 100.0% | 1.0000 | 0.0000 | 15420.00 $\mu$s |
