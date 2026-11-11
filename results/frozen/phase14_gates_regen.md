# Gates 1/2/3 regen after exact-oracle + pose fixes (2026-10-01)

## Gate 1 — REGEN DONE (pose-fixed)
- Fixed the same C2W/W2C transposed-viewpoint bug in `experiments/run_gate1_headroom.py`
  (renders were misaligned; old numbers polluted).
- Fresh run: `results/gate1_headroom/gate1_summary.json` (fr1_desk, 5 seeds [42-46]).
- Headroom H strictly positive (p=0.031, dz=3.08; +0.0026 dB at K=top-20%).
- Negative utility is stratum-dependent: flat 38.7%, texture 21.3%, edge 18.7%,
  depth-discontinuity 2.7% (headline 21.37% was a different stratum mix).
- OSE: heuristic 0.396 vs error-only 0.536 (n.s., p=1.0).
- Group additivity HERE: R_add(4)=0.95, R_add(16)=0.82 — nearly additive in this
  small-group joint-optimization regime. This differs from the extreme
  sub-additivity (R_add(16)~0.005) measured under the Phase-6 pairwise
  interaction protocol; the two estimands differ (joint-group gain ratio vs
  pairwise interaction residuals) and must not be conflated. Open: reconcile
  regimes in one protocol.
- Diminishing-returns spot test: REJECTED on 10 trials (flat marginals) — no claim made.

## Gate 2 — NOT rerun (documented rationale, frozen artifacts preserved)
- Gate 2 = offline ranking on the frozen UtilityDataset with frozen TwoHeadMLP
  checkpoints (SHA-pinned in docs/checkpoints.md). Retraining would destroy the
  frozen registry. Ranking uses pointwise U* only — unaffected by either the
  pose fix (no rendering at eval time; labels precomputed) or the greedy fork
  fix (pointwise, not conditional). Numbers stand with invariance rationale.

## Gate 3 — old protocol DEPRECATED, replaced
- `run_phase6_budget_sweep.py` header: unfair non-unified budgets, historical only.
- Replaced by matched-budget curve: `results/phase14_budget_curve/budget_curve.json`
  (budgets 5/10/15/20/30 ms x {ours, error_only} x 3 seeds, fr2_xyz 30f).
- Finding: saturation by ~15-20 ms at 30f horizons; policies tie short-horizon.
  The 150f main benchmark is where budgets/policies separate (horizon effect).

## Greedy contextual oracle (fork fix 2026-10)
- `research/oracle_utility.py::greedy_contextual_oracle_selection` now forks both
  branches from the same base state (prior version double-optimized S).
- Direction check: the bias inflated conditional gains, yet the old finding was
  advantage ≈ 0 — the Case-B conclusion (rank stability, no re-ranking gain) is
  robust to the fix direction. Full Phase-6 contextual regen remains future work;
  conditional numbers stay out of the main paper until then.
