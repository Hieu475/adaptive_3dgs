# Phase 3 — Ground-Truth Marginal Utility Oracle Validation Report

**Protocol Version**: `1.0.0`  
**Date Generated**: `2026-09-04 15:13:03`  
**Execution Device**: `cuda`  
**Evaluated Seeds**: `[42, 43, 44, 45, 46]`  
**Primary Scientific Estimand**: Global $\Delta Q_i^{\text{global}}$ ($w_{\text{rgb}}=0.70, w_{\text{depth}}=0.30$)  
**Phase 3 Acceptance Status**: `PASS`

---

## 1. Executive Summary & Acceptance Criteria Verification

| Criterion | Requirement | Observed Empirical Value | Status |
| :--- | :--- | :--- | :---: |
| **Primary Scientific Estimand** | Global $\Delta Q_i^{\text{global}}$ via SelectiveAdam | **Locked as primary label across dataset** | **PASS** |
| **Non-Trivial Variance** | $\text{Var}(U^\star) > 0$ | **3.40e-13** (std = 5.84e-07) | **PASS** |
| **Negative Utility Preservation** | $U^\star < 0$ unclamped & preserved | **144/704 (20.5%)** | **PASS** |
| **State Snapshot/Restore** | Bitwise cryptographic equality | **SHA-256 state hash identical** | **PASS** |
| **Leakage Audit** | Zero post-intervention tokens in $s_i(t)$ | **100% Pre-intervention features** | **PASS** |
| **Feature Provenance** | Observed update frequency & visibility count | **Deterministic from StateStore & attribution** | **PASS** |
| **Duplicate Temporal Drift** | Removed from feature inputs | **temporal_drift eliminated from input features** | **PASS** |
| **Split Separation** | Train / Val / Cross-scene partitioning | **Train=330, Val=190, Test=184** | **PASS** |
| **Multi-Seed Provenance** | Seeds evaluated across protocol | **seeds=[42, 43, 44, 45, 46] with per-row seed tag** | **PASS** |
| **Repeatability** | Multi-trial stability on candidates | **Mean CV = 0.0124** (Pos CV = 0.0132) | **PASS** |
| **Group Interaction Isolation** | Separate artifact for interaction $\Delta Q(S)$ | **Exported to `group_interaction_analysis.json`** | **PASS** |
| **Non-Additivity & Interaction** | Empirical evaluation of interaction $\Delta Q(S)$ | **Substantial non-additivity; mixed diminishing evidence (37.5%)** | **PASS** |

---

## 2. Dataset Distribution & Filtering (Phase 3.6 & 3.7)

- **Total Interventions Recorded ($N_{\text{total}}$)**: `875` across 5 protocol seeds
- **Valid Interventions ($N_{\text{valid}}$, influence $\ge 25$ pixels)**: `704`
- **Filtered Interventions ($N_{\text{filtered}}$, influence $< 25$ pixels)**: `171` (19.5%)
- **Positive Utility Count ($U^\star > 0$)**: `560` (79.5%)
- **Negative Utility Count ($U^\star < 0$)**: `144` (20.5%)

### Decoupled Quality & Cost Statistics
- **Utility ($U^\star$) Mean**: `2.87e-07` | **Median**: `8.02e-08` | **Std**: `5.84e-07`
- **Utility Range**: `[-1.08e-06, 4.62e-06]`
- **Mean Intervention Cost ($\Delta T$)**: `62.88 ms` per Gaussian trial

---

## 3. Multi-Seed Provenance Breakdown (Protocol Seeds)

| Seed | Total Recorded | Valid Interventions | Mean Utility ($U^\star$) | Std Utility | Negative Utility Count ($U^\star < 0$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `175` | `145` | `2.95e-07` | `5.03e-07` | `26 (17.9%)` |
| **43** | `175` | `141` | `2.56e-07` | `5.66e-07` | `29 (20.6%)` |
| **44** | `175` | `145` | `2.78e-07` | `6.38e-07` | `29 (20.0%)` |
| **45** | `175` | `141` | `3.35e-07` | `6.68e-07` | `25 (17.7%)` |
| **46** | `175` | `132` | `2.68e-07` | `5.17e-07` | `35 (26.5%)` |

---

## 4. Geometry Stratification & Negative Utility Risk (Phase 3.8)

$$P(U^\star < 0 \mid \text{geometry})$$

| Geometry Stratum | Interventions ($N$) | Mean Utility ($U^\star$) | Std Utility | Negative Utility Fraction $P(U^\star < 0)$ |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Surfaces** | `173` | `3.07e-07` | `6.29e-07` | `22.0%` |
| **Object Edges** | `183` | `3.01e-07` | `5.38e-07` | `17.5%` |
| **High Texture** | `181` | `2.59e-07` | `5.45e-07` | `22.1%` |
| **Depth Discontinuity** | `143` | `2.85e-07` | `6.46e-07` | `18.9%` |

---

## 5. Multi-Trial Repeatability Analysis (Phase 3.9)

Tested across $N = 25$ Gaussians with $3$ independent trials per Gaussian from identical baseline state:
- **Overall Mean CV**: `0.0124`
- **Overall Median CV**: `0.0080`
- **Positive Utility CV**: `0.0132` ($N = 23$)
- **Negative Utility CV**: `0.0032` ($N = 2$)
- **Mean Sign Stability**: `92.0%`

---

## 6. Group Interaction & Non-Additivity (Phase 3.10)

Evaluated empirical additivity ratio $R_{\text{add}}(S) = \frac{\Delta Q(S)}{\sum_{i \in S} \Delta Q_i + \epsilon}$ and Interaction Error $I(S)$:

| Group Size ($|S|$) | Mean Additivity Ratio $R_{\text{add}}$ | Mean Interaction Error $I(S)$ | Tested Groups |
| :---: | :---: | :---: | :---: |
| **1** | `1.0000` | `0.0000` | `32` |
| **4** | `0.4538` | `0.5240` | `4` |
| **16** | `0.4576` | `0.8808` | `2` |

*Conclusion*: Single-Gaussian utility cannot be summed linearly to predict group update gain; group interactions demonstrate substantial non-additivity. The full interaction data is preserved in `results/oracle_dataset/group_interaction_analysis.json`.

---

## 7. Empirical Diminishing Marginal Returns (Phase 3.11)

Tested whether $\Delta_i(A) \ge \Delta_i(B)$ for nested subsets $A \subset B$ ($|A| = 2, |B| = 6$):
- **Mean Marginal Gain $\Delta_i(A)$**: `0.000024`
- **Mean Marginal Gain $\Delta_i(B)$**: `0.000033`
- **Empirical Diminishing Returns Rate**: `37.5%`
- **Finding**: The experiments reveal substantial non-additivity and limited/mixed empirical evidence for diminishing marginal returns under the tested intervention protocol.

---

## 8. Dataset Split Partitioning (Phase 3.12)

Strict split partitioning preserved across all files without random mixing:
- **Train Split (`tum_fr1_desk`, frames 0–40)**: `330` interventions
- **Validation Split (`tum_fr1_desk`, frames 41–60)**: `190` interventions
- **Cross-Scene Test Split (`tum_fr2_xyz`)**: `184` interventions

---

## 9. Statistical Unit & Evaluation Methodology

The dataset contains $N_{\text{valid}} = 704$ interventions sampled across 5 protocol seeds $[42, 43, 44, 45, 46]$. These observations are structured hierarchically (clustered by scene, frame, and seed). Consequently, downstream evaluation in Phase 4 and beyond adheres to:
1. **Per-seed metric reporting**: Aggregated as $\text{mean} \pm \text{std}$ across seeds rather than assuming all $N_{\text{valid}}$ rows are independent identically distributed.
2. **Cluster / block bootstrap**: Resampling blocked by frame and seed to avoid artificially deflating standard errors.

All generated artifacts are tracked under `results/oracle_dataset/`:
- `oracle_dataset.json` (Full hierarchical dataset with multi-seed provenance)
- `oracle_dataset.csv` (Tabular format with features $s_i(t)$ and labels $U_i^\star$)
- `oracle_dataset_summary.json` (Machine-readable summary metrics)
- `group_interaction_analysis.json` (Isolated group non-additivity study)
- `oracle_validation_report.md` (Executive report)
