# research_archive

Exploratory `analyze_*` scripts moved out of `experiments/` on 2026-11-06 because no paper source, test, other script, or result manifest references them
(52 files; mostly capacity-balanced, GPU-timing/telemetry, growth-starvation, age-quota and identity-revisit diagnostics).
They are kept flat in this directory so their `Path(__file__).parent.parent` repo-root logic still works; run them from the repo root, e.g. `python research_archive/analyze_long_horizon.py`.
The age-quota / stale-Gaussian-rescue scripts here are the most likely starting point for a method contribution beyond coverage throttling.

**Second batch (tracked, 34 files):** phase 6–12 scripts and historical ablation drivers (`run_phase12_*`, `run_step1_*`, `run_learned_utility_*`, `process_phase9*`, ...) that no paper source, test, other script, or result manifest references.
Their frozen outputs remain under `results/`; the scripts are kept for provenance only and were not re-run after the move.
