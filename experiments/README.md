# Experiments registry — canonical vs legacy

Canonical (use these; everything else is legacy/provenance):
- `run_phase13_frozen_benchmark.py` — authoritative 6-policy x seeds benchmark (gsplat backend)
- `run_final_confirmation.py` — 8-seed confirmation harness (Phase-14: OURS 27.67dB, +2.63/+4.62dB, 98% headroom, Sec.12)
- `run_multiscene_benchmark.py` — cross-scene (tum_fr1_xyz, tum_fr2_xyz, replica_office0)
- `run_phase12i_interaction_audit.py` — GO/NO-GO interaction audit (do before any new model)
- `run_phase10_smoke.py` — <30s smoke (checkpoint+A1+B2+knapsack+StateStore)
- `validate_attribution_fidelity.py` / `validate_phase13_significance.py` — fidelity + Holm stats

Baselines (same budget interface):
- internal: `research/benchmark_policies.py` (NO_OP, ERROR_ONLY, ERROR_INFLUENCE, FULL)
- external: `research/baselines/rtg_slam_policy.py` (numpy reference, unit-tested)
  + `research/scheduler.py::RTG_SLAM_REIMPL` (torch mirror, used in harness)
  Authoritative corrected-substrate 150f x 8 seeds, `results/phase14_corrected/`:
  OURS 27.67±0.04 vs rtg 23.05 (+4.62, Holm p=0.0391 ✅) vs error_only +3.59 ✅
  vs error_influence +2.63 ✅ vs full -0.18 (98.0% headroom, 23K vs 88K map).
  Old `results/final_confirmation/` (19.24dB) is pose-bugged — see DEPRECATED.md,
  never compare absolutes across the fix.

Legacy: `run_phase6_*.py`, `run_phase5_*.py`, `run_phase9*.py`, `process_phase*.py`,
`run_oracle_*.py`, `run_budget_sweep.py`, etc. — kept for provenance, do NOT extend.
New work goes through the 6 canonical scripts above.

Configs: single source is `research/pipeline.py::_default_config()`;
`configs/default.yaml` mirrors it. Benchmark overrides: `budget=15ms, sh_degree=0/1`.
