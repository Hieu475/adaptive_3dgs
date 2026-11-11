# Phase 12-I: Interaction Audit Report

## 1. Executive Summary

This report analyzes Gaussian interaction effects in the Adaptive 3DGS pipeline,
testing whether pointwise utility $U^*(i|\emptyset)$ is a sufficient statistic
for subset selection, or whether conditional utility $U^*(i|S)$ is needed.

- **Total Pairs Analyzed**: 500
- **Total Conditional Evaluations**: 750
- **Sub-additive Interactions**: 64.20%
- **Sign Flip Rate $R_{flip}$**: 16.67%

## 2. Pairwise Interaction $I_{ij}$

$$I_{ij} = \Delta Q(\{i,j\}) - \Delta Q_i - \Delta Q_j$$

| Metric | Value |
|--------|-------|
| Mean $I_{ij}$ | -0.000774 |
| Mean $|I_{ij}|$ | 0.001429 |
| % Sub-additive (redundant) | 64.20% |
| % Super-additive (synergy) | 35.40% |
| $R_{pair}$ mean | 1.7375 |

## 3. Interaction by Overlap Bin

| Overlap | Mean $I_{ij}$ | % Sub-additive |
|---------|----------------|----------------|
| Low (IoU < 0.1) | 0.000028 | 49.8% |
| Medium (0.1-0.5) | -0.001018 | 69.4% |
| High (IoU > 0.5) | -0.002660 | 95.7% |

## 4. Sign Flip Analysis

$$R_{flip} = P[\text{sign}(U^*(i|S)) \neq \text{sign}(U^*(i|\emptyset))]$$

| Overlap Bin | Sign Flip Rate |
|-------------|---------------|
| Low | 5.36% |
| Medium | 14.17% |
| High | 25.00% |

## 5. Conditional Utility Deviation

$$D_Q(i,S) = \Delta Q(i|S) - \Delta Q(i|\emptyset)$$

| Metric | Value |
|--------|-------|
| Mean $|D_Q|$ | 0.002100 |
| Mean $R_Q$ | -1.1394 |

## 6. GO / NO-GO Assessment

**Decision: GO**

- R_flip = 0.167 > 0.05
- |I_ij| = 0.001429 > 1e-4
