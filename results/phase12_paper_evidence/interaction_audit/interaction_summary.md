# Phase 12-I: Interaction Audit Summary Report

## 1. Overview
This report summarizes the findings from the interaction audit, evaluating the non-additivity of utility functions in Adaptive 3DGS.

## 2. Key Statistics
- **Pairwise Interaction Residual $I_{ij}$**: Mean = -0.0008, Std = 0.0018
- **Sign Flip Rate**: 16.67% (125/750)
- **Mean Utility Ratio $R_Q$**: -1.1394

## 3. Analysis Findings
- **Interaction vs Overlap**: (See `interaction_vs_iou.png`). Interaction residuals often correlate with spatial overlap, indicating non-additivity is localized.
- **Context Size Effects**: (See `rq_by_context_size.png`). As context size grows, the deviation from isolated utility tends to increase, validating the need for context-aware evaluation in deeper searches.
- **Sign Flips**: A non-zero sign flip rate indicates that operations beneficial in isolation may become detrimental in context (or vice versa).

## 4. GO/NO-GO Assessment
Based on the interaction magnitude, if $I_{{ij}}$ is substantial and sign flips are frequent, a naive greedy approach without re-evaluation is insufficient. The results support advancing to Phase 12-II to benchmark search strategies that handle these interactions.
