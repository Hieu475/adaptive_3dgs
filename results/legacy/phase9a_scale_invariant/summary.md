# Phase 9A: Robust Utility Representation (Scale-Invariant Features) Report

**Generated at:** 2026-09-14T18:58:50.817684
**Protocol Version:** 1.0.0 (Frozen)
**Primary Scientific Objective:** Determine whether scale-invariant feature representation reduces cross-scene generalization degradation from `tum_fr1_desk` to `tum_fr2_xyz` without sacrificing in-domain utility quality.

---

## 1. Executive Summary & Gate Status

| Gate | Name | Type | Status | Key Metric / Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 9A-1** | Protocol Integrity | Checklist | **PASS** | Train/test strictly separated; no oracle leakage; same data split and seeds. |
| **Gate 9A-2** | Representation Validity | Quantitative | **PASS** | All variants and seeds ran with finite metrics and zero numerical pathology. |
| **Gate 9A-3** | Generalization Improvement | Quantitative / Decision | **CONDITIONAL_PASS** | A1 gap reduced (Delta_rho=-0.0797 vs A0=0.0713) but in-domain regression detected (rho_in=+0.1911 vs A0=+0.3185). |

---

## 2. Representation Comparison Table

| Variant | Description | In-Domain $\bar{\rho}$ | Zero-Shot $\bar{\rho}$ | Generalization Gap $\Delta\rho$ | Zero-Shot NDCG@20 | Zero-Shot OSE@20 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **raw** | A0: 11 raw canonical features (Phase 4 baseline) | +0.3185 ± 0.1072 | +0.2472 ± 0.1213 | **+0.0713** (95% CI: [-0.0072, 0.1498]) | 0.6811 | 0.542 |
| **geometry_relative** | A1: Scale-relative depth, drift, projected area | +0.1911 ± 0.2631 | +0.2709 ± 0.1631 | **-0.0797** (95% CI: [-0.2415, 0.0820]) | 0.6276 | 0.560 |
| **geometry_optimization_relative** | A2: A1 + relative grad, influence, uncertainty, residual | +0.1949 ± 0.2443 | +0.2634 ± 0.2136 | **-0.0684** (95% CI: [-0.1733, 0.0364]) | 0.6113 | 0.375 |

---

## 3. Statistical Hypothesis Testing vs. Baseline (A0)

Comparison of per-seed generalization gap difference $\Delta\rho_s = \Delta\rho_{s, \text{variant}} - \Delta\rho_{s, A0}$ ($n=5$ seeds):

| Variant Comparison | Mean Gap Reduction | 95% CI | Wilcoxon $p$-value | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **geometry_relative vs. A0** | -0.1510 ± 0.1944 | [-0.3214, 0.0194] | 0.1250 | Reduced transfer degradation |
| **geometry_optimization_relative vs. A0** | -0.1397 ± 0.0875 | [-0.2164, -0.0631] | 0.0625 | Reduced transfer degradation |

---

## 4. Scientific Narrative & Statistical Interpretation

### Key Findings:
1. **Core Conclusion:** Scale-invariant features reduce the measured cross-scene generalization gap, with an in-domain performance trade-off. This is not an unconditional improvement across all axes, but a targeted mitigation of camera-induced distribution shift.

2. **Distribution Shift Elimination (Figure 19):**
   - Raw baseline (A0) suffers severe cross-scene feature mismatch: `depth_error` median ratio is 0.223 (4.5x domain shift), and `projected_area` distributions are statistically distinct ($p=0.0011$).
   - Relative representation (A1) aligns feature distributions: `depth_error` median ratio becomes 1.000, and `projected_area` distributions become statistically indistinguishable ($p=0.4916$, Wasserstein distance drops from 2.019 to 0.287).
   - **Mechanism chain:** $\text{camera scale shift} \to \text{feature distribution shift} \to \text{raw model transfer degradation}$. Relative representation breaks this chain.

3. **Statistical Significance & Representation Trade-off:**
   - **A1 (Geometry-Relative):** Generalization gap reduction is $-0.1510$ with Wilcoxon $p=0.1250$ (directionally consistent improvement, though not statistically significant at $p<0.05$ under $n=5$ seeds). Zero-shot correlation improves (+0.2709 vs. +0.2472), and selection efficiency improves (OSE@20 = 0.560 vs. 0.542).
   - **A2 (Geometry+Opt-Relative):** Stronger statistical trend in gap reduction ($-0.1397$, $p=0.0625$, 95% CI: [-0.2164, -0.0631]), BUT selection quality collapses (zero-shot OSE@20 drops to 0.375). Over-normalizing optimization and attribution features discards vital magnitude signal (*too much invariance* $\to$ *loss of useful absolute information*).
   - **Selection:** A1 is selected over A2 as the preferred representation candidate because A1 maintains a balanced trade-off between transfer robustness and selection efficiency.

4. **In-Domain Trade-off:**
   In-domain correlation drops from +0.3185 (A0) to +0.1911 (A1), primarily driven by seed-level optimization variance on seeds 45 and 46. Full scale-invariance removes absolute depth cues that are informative in-domain.

---

## 5. Phase 9 Decision & Handover to Phase 9B

| Variant | Role in Experiment | Status | Rationale |
| :--- | :--- | :---: | :--- |
| **A0** (`raw`) | Phase 4 Baseline | **Reference** | Baseline canonical features with train-only normalizer. |
| **A1** (`geometry_relative`) | Primary Candidate | **ACCEPTED** | Preferred robust representation: eliminates feature shift, improves zero-shot transfer, highest OSE@20. |
| **A2** (`geometry_optimization_relative`) | Ablation | **REJECTED** | Severe selection degradation (OSE@20 = 0.375); over-normalizes critical gradient/influence signals. |

> [!IMPORTANT]
> **Next Step (Phase 9B):** Instead of searching for heuristic invariant feature formulas, Phase 9B will focus on resolving the in-domain trade-off through **Test-Time Adaptive Normalization (TTAN)** to dynamically calibrate representation scales during online SLAM execution.
