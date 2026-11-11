# Results

Result bundles, grouped by research topic. The paper-to-bundle mapping is in `paper_repro/README.md`;
`docs/layout_renames.json` maps every old flat path (`results/<bundle>`) to its new location.

| Group | Contents |
|---|---|
| `dilution/` | the main claim (compute dilution): `rerun_paper1`, `dilution_*`, `throttle_*`, `count_cap`, `compute_matched`, `depth_intervention`, `matched_dilution_controls`, `mechanism_v2`, warm-up/step-1 ablations |
| `controllers/` | engine and controller evaluations: `engine*`, `vgg_*`, `tt_*`, `retention*`, `joint_spawn_refine_replay`, `seeds`, `policies` |
| `holdout/` | held-out view evaluations: `holdout_*`, `heldout_*`, `phase14_holdout` |
| `baselines/` | external baselines: `rtg_slam_*`, `monogs_*`, `splatam`, `sota_comparison.md` |
| `tracking/` | estimated-pose and closed-loop tracking: `tracked_poses`, `closed_loop*`, `phase14_tracking*` |
| `replica/` | Replica E1/E2 pilots, schedules, and per-scene runs (`replica_*`) |
| `long_horizon/` | long-horizon and full-sequence runs |
| `frozen/` | frozen, provenance-hashed bundles: `final_confirmation` (legacy, pose-bugged), `phase13_frozen_benchmark`, `phase14_corrected*`, `phase14_projected_area_fix_*` |
| `legacy/` | earlier phases (`phase5`–`phase13`, oracle/utility/budget experiments), kept for provenance |
| `renders/` | qualitative renders |
| `diagnostics/` | speed, determinism, profiling, demo outputs, figures |
| `logs/` | run logs and `*.DONE` sentinels (git-ignored) |

Notes:
- Manifests and JSON inside bundles still record the paths used when they were written (for example `experiments/run_final_confirmation.py`). They were not rewritten, so their hashes remain valid. Use `docs/layout_renames.json` to translate them.
- Most large files are git-ignored (see `results/.gitignore`, whose patterns were remapped to the grouped layout); a few bundles are force-tracked.
