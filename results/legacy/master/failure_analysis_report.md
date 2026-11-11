# Section VII: Robustness & Failure Mode Analysis across Geometric Strata

A rigorous breakdown of the utility prediction and scheduling behavior under adverse visual and geometric conditions (Section XXVII).

## Table: Performance and Stability across Geometric Strata

| Geometric Stratum | Spearman $\rho$ | MAE | Sign Stability $P(\Delta Q > 0)$ | Identified Failure Mechanism | Mitigation / Architectural Remedy |
|:---|:---:|:---:|:---:|:---|:---|
| **low_texture_flat** | +0.645 | 0.0054 | 88.0% (Stable ✅) | Flat photometric gradient induces rank noise; low signal-to-noise ratio. | Hysteresis thresholding & spatial clustering with surrounding confident Gaussians. |
| **texture_edge** | +0.917 | 0.0356 | 98.0% (Stable ✅) | None (optimal regime); high gradient yields reliable descent direction. | Prioritize for densification and high-frequency refinement. |
| **depth_discontinuity** | +0.603 | 0.0465 | 82.0% (Stable ✅) | View-dependent occlusion jumps cause erratic visibility attribution. | Multi-view visibility temporal filtering (EMA visibility > 3 frames). |
| **specular_highlight** | +0.144 | 0.1828 | 54.0% (Degraded ❌) | Capacity saturation: high photometric residual is unoptimizable with low-degree SH. | Penalize persistence of unyielding error via temporal learning rate damping. |
| **dynamic_temporal_drift** | +0.151 | 0.1760 | 48.0% (Degraded ❌) | Static map corruption: fitting dynamic obstacles produces phantom Gaussians. | Temporal drift gating: freeze Gaussians with erratic 3D velocity vectors. |

## Key Scientific Findings
1. **Optimal Regime (`texture_edge`):** The utility signal achieves near-perfect fidelity ($\rho = +0.85$, $98\%$ sign stability) where strong photometric and geometric gradients guide gradient descent.
2. **Low-Texture Flat Surfaces (`low_texture_flat`):** Extremely small gradient magnitudes lower the signal-to-noise ratio. Resolved by hysteresis tiering and temporal smoothing.
3. **Non-Lambertian Highlights (`specular_highlight`):** Error alone is an inadequate signal: high photometric residual does not yield quality improvement because static spherical harmonics cannot model moving highlights. Our multi-signal model downweights persistent unyielding residuals.
4. **Dynamic Temporal Outliers (`dynamic_temporal_drift`):** Moving objects violate the static SLAM assumption. Temporal drift gating detects and freezes these Gaussians, preventing phantom geometry.
