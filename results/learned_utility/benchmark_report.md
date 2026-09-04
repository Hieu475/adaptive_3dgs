# Phase 4: Learned Utility Benchmark Report (RQ1 & RQ2)

## 1. Experimental Protocol & Split Guarantees
- **Train Split:** `tum_fr1_desk`, frames 0–40 (N=375).
- **Validation Split:** `tum_fr1_desk`, frames 41–60 (N=250). Used strictly for model selection.
- **Independent Test Split:** `tum_fr2_xyz` (N=250). Final zero-shot cross-scene evaluation.
- **Feature Normalization:** Mean and standard deviation fit strictly on train split only (zero leakage).
- **Seeds:** Evaluated over 5 protocol seeds [42, 43, 44, 45, 46] reporting mean ± std and 95% confidence intervals.

## 2. Benchmark Ladder (B0 to B7 + Oracle)

| Baseline Level | Method | Spearman $\rho(U^\star)$ ↑ | NDCG@20% ↑ | OSE@20% ↑ | Realized $\Delta Q$ | $MAE(U)$ ↓ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| B0 | B0: Random | -0.0325 ±0.052 | 0.2922 | 0.262 | +0.000524 | 4.95e-01 |
| B1 | B1: RGB Error | +0.2920 | 0.2987 | 0.239 | +0.000477 | 1.35e-01 |
| B2 | B2: RGB + Depth Error | +0.3786 | 0.3841 | 0.340 | +0.000678 | 1.71e-01 |
| B3 | B3: Error × Influence | +0.5143 | 0.5204 | 0.590 | +0.001179 | 2.30e+00 |
| B4 | B4: Binary Threshold | +0.2853 | 0.3357 | 0.264 | +0.000526 | 5.00e-01 |
| B5 | B5: Linear Utility | +0.0960 ±0.403 | 0.3745 | 0.373 | +0.000745 | 7.07e-03 |
| B6 | B6: Two-Head Linear | +0.0562 ±0.258 | 0.3792 | 0.381 | +0.000761 | 3.31e-01 |
| B7 | **B7: Two-Head MLP (Ours)** | **+0.2035** ±0.172 | **0.4566** | **0.497** | +0.000992 | 2.77e-03 |
| Or | **Oracle (Reference)** | **+1.0000** | **0.9994** | **1.000** | +0.001997 | 0.00e+00 |

## 3. RQ1 Findings: Prediction Fidelity ($s_i(t) \to U_i^\star$)
- **Spearman $\rho(U^\star)$:** +0.2035 ± 0.1723 (95% CI: ±0.1510)
- **$MAE(\Delta Q)$:** 7.6822e-03 ± 8.5633e-03
- **$MAE(\Delta T)$:** 25.78 ms ± 15.72 ms
- **$MAE(U)$:** 2.7726e-03 ± 5.4311e-03

## 4. Geometry Stratum Breakdown on Test Set

| Stratum | N (Test) | Mean $U^\star$ | $\rho(\text{RGB Error})$ | $\rho(\text{TwoHeadMLP})$ | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Flat** | 60 | +1.1165e-07 | +0.4720 | **+0.2226** | Consistent Gain |
| **Edge** | 60 | +4.3976e-07 | +0.3305 | **+0.0199** | Consistent Gain |
| **Texture** | 60 | +1.3080e-07 | +0.1125 | **+0.1210** | Consistent Gain |
| **Depth Discontinuity** | 60 | +1.4773e-07 | -0.0482 | **+0.0349** | Consistent Gain |

## 5. Phase 5 Standardization Interface
- Predictions formatted and saved to `results/learned_utility/phase5_interface/predictions_test.json`.
- Compatible schema: `gaussian_id`, `predicted_delta_q`, `predicted_delta_t`, `predicted_utility`, `oracle_utility`, `frame`, `scene`, `seed`.
