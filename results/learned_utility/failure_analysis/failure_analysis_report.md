# Phase 4: Systematic Failure Mode Analysis of Learned Utility Estimator

## 1. Overview
- **Dataset Evaluated:** Independent cross-scene test set (`tum_fr2_xyz`, N=250).
- **Over-Predicted Gaussians ($\hat U_i \gg U_i^\star$):** 25 samples (top 10% positive residual).
- **Under-Predicted Gaussians ($\hat U_i \ll U_i^\star$):** 25 samples (bottom 10% negative residual).

## 2. Geometry Stratum Breakdown

| Stratum | Total Samples | Over-Predicted Rate | Under-Predicted Rate | Mean Residual | $MAE(U)$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Depth Discontinuity** | 60 | 5.0% | 8.3% | -3.19e-03 | 1.35e-02 |
| **Edge** | 60 | 10.0% | 21.7% | -5.05e-03 | 1.55e-02 |
| **Flat** | 60 | 1.7% | 5.0% | -4.71e-03 | 1.04e-02 |
| **General Visible** | 10 | 0.0% | 20.0% | -7.26e-03 | 1.16e-02 |
| **Texture** | 60 | 25.0% | 3.3% | +1.02e-02 | 1.54e-02 |

## 3. Systematic Feature Drivers of Misprediction

### 3.1 Over-Prediction Drivers (Features with Largest Positive Z-Shift in Over-Predicted Set)
- **`rgb_error`**: z-shift = `+1.627` (elevated in over-predicted Gaussians)
- **`gradient_norm`**: z-shift = `+0.333` (elevated in over-predicted Gaussians)
- **`influence_mass`**: z-shift = `-0.229` (elevated in over-predicted Gaussians)

### 3.2 Under-Prediction Drivers (Features with Largest Z-Shift in Under-Predicted Set)
- **`uncertainty_var`**: z-shift = `-1.463` (depressed in under-predicted Gaussians)
- **`depth_error`**: z-shift = `+1.209` (depressed in under-predicted Gaussians)
- **`gradient_norm`**: z-shift = `+0.559` (depressed in under-predicted Gaussians)

> **Scientific Implication for Phase 5:** Scheduler must monitor unyielding residuals on flat and boundary Gaussians where footprint or residual EMA artificially inflate perceived utility without delivering actual quality gains.