# Experiments registry — canonical vs legacy

Metric protocol note (2026-10-16): artifacts created before
`phase13-authoritative-v2-final-map-quality`,
`full-sequence-v2-final-map-quality-synced-latency`, or
`long-horizon-prefix-v1-final-map-quality-synced-latency` reported depth L1
from the pre-update render and PSNR/SSIM before prune/compaction. In new runs,
PSNR, SSIM, and depth L1 are rendered from the final per-frame map after
optimization, spawn/prune, and compaction. `depth_l1_pre` retains the
pre-update residual. Do not compare old and new depth L1 values directly.

Pose protocol correction (2026-10-17): `TUMDataset.__getitem__` returns C2W,
while the mapping pipeline expects W2C. The full-sequence TUM runner now calls
`TUMDataset.w2c_pose()` before processing. TUM long-horizon/full-sequence
artifacts produced by this runner before the `*-v3-w2c-*` / `*-v2-w2c-*`
protocol are not valid pose-controlled mapping evidence and must not be used as
primary results. Replica runtime poses are already converted by the Phase 10
loader. The current TUM runner also rejects selected frames whose nearest
ground-truth pose is more than 50 ms away by default; use `--start_frame` to
select a contiguous valid window and record its source-frame interval.

Canonical (use these; everything else is legacy/provenance):
- `run_frozen_benchmark.py` — authoritative 6-policy x seeds benchmark (gsplat backend)
- `run_final_confirmation.py` — 8-seed confirmation harness; the post projected-area-fix frozen bundle is `results/phase14_projected_area_fix_clean_7policy_8seeds/`.
- `run_multiscene_benchmark.py` — cross-scene (tum_fr1_xyz, tum_fr2_xyz, replica_office0)
- `run_interaction_audit.py` — GO/NO-GO interaction audit (do before any new model)
- `run_closed_loop_smoke.py` — <30s smoke (checkpoint+A1+B2+knapsack+StateStore)
- `validate_attribution_fidelity.py` / `validate_frozen_significance.py` — fidelity + Holm stats

Baselines (same budget interface):
- internal: `research/benchmark_policies.py` (NO_OP, ERROR_ONLY, ERROR_INFLUENCE, FULL)
- external: `research/baselines/rtg_slam_policy.py` (numpy reference, unit-tested)
  + `research/scheduler.py::RTG_SLAM_REIMPL` (torch mirror, used in harness)
  Post projected-area-fix clean-source 150f x 8 seeds, `results/phase14_projected_area_fix_clean_7policy_8seeds/`:
  OURS 27.64±0.04 vs RTG-SLAM 23.16 (+4.48), error_only +3.59,
  error_influence +2.61, temporal +2.70, FULL -0.21 dB; all six paired
  contrasts Holm p=0.0469; 97.6% headroom, 23.8K vs 90.6K map. The bundle
  passes `audit_experiment_manifests` as `paper_ready` and records the frozen
  source commit plus config/source/artifact hashes. The prior
  `results/phase14_corrected/` bundle is retained as historical reference.
  Old `results/final_confirmation/` (19.24dB) is pose-bugged — see DEPRECATED.md,
  never compare absolutes across the fix.

Legacy: `run_phase6_*.py`, `run_phase5_*.py`, `run_phase9*.py`, `process_phase*.py`,
`run_oracle_*.py`, `run_budget_sweep.py`, etc. — kept for provenance, do NOT extend.
New work goes through the 6 canonical scripts above.

Configs: single source is `research/pipeline.py::_default_config()`;
`configs/default.yaml` mirrors it. Benchmark overrides: `budget=15ms, sh_degree=0/1`.

## Provenance gate

Before copying numbers into a paper table, run:

```bash
python -m experiments.audit_experiment_manifests
```

The auditor checks commit cleanliness, config/snapshot hash, dataset, pose
convention, seed identity, renderer backend, and declared artifact hashes. It
labels bundles `paper_ready`, `exploratory`, `invalid`, or `legacy`; the known
pre-pose-fix `results/final_confirmation/` bundle is always `legacy`. To enforce
the gate in a paper-export workflow, pass `--require-paper-ready` and the exact
bundle paths. The auditor is conservative: missing provenance prevents a bundle
from being paper-ready even when its metrics look plausible.

Phase 13 and long-horizon manifests record the effective per-policy configs,
renderer backend, pose convention, and statistical replicate unit. Confidence
intervals use complete run summaries per seed; frames within one trajectory are
not treated as independent samples. A fixed seed does not imply bitwise CUDA
repeatability when gsplat atomic gradient accumulation is active.

## Same-snapshot replay checkpoints

The growth-starvation and joint spawn-refine replay runners can optionally save
a full frame-boundary checkpoint, including Gaussian identity state, optimizer
momentum, tracker, scheduler, importance statistics, background cache, and RNG
state. This is separate from `--save_map_snapshots`, which stores map state for
provenance only.

Save a checkpoint while running a replay batch:

```bash
python experiments/run_growth_starvation_factorial.py \
  --scenes tum_fr1_rpy --seeds 42 --snapshot_frames 30 --n_frames 60 \
  --outcome_horizon_frames 15 --save_replay_checkpoints \
  --out_dir results/replay_checkpoint_source
```

Replay the same snapshot in another run directory without rebuilding the
common prefix:

```bash
python experiments/run_growth_starvation_factorial.py \
  --scenes tum_fr1_rpy --seeds 42 --snapshot_frames 30 --n_frames 60 \
  --outcome_horizon_frames 15 \
  --replay_checkpoint_dir results/replay_checkpoint_source \
  --out_dir results/replay_checkpoint_replay
```

Loading checks scene, seed, frame, effective config hash, Gaussian count, and
map digest before continuing. Keep the source checkpoint and manifest with the
replay record. Checkpoints are experiment replay artifacts, not general-purpose
training resumes; they require a compatible code environment and configured
device family.

The joint replay runner can also compare both SSIM implementations from the
same restored snapshot and save the first selected-Gaussian gradients before
the first optimizer update. This captures the active subset and frozen
background-cache path used by `process_frame`:

```bash
python experiments/run_joint_spawn_refine_replay.py \
  --scenes replica_room1 --seeds 279 280 281 --snapshot_frames 30 \
  --n_frames 60 --micro_updates 5000 --spawn_counts 0 128 \
  --mature_fractions 0.5 --repeat_actions 3 --action_order_seed 815 \
  --ssim_implementation both --capture_first_backward_gradients \
  --replay_checkpoint_dir results/phase14_generalization_postfix/room1_coverage_saturating_holdout_seeds279_281 \
  --out_dir results/ssim_active_subset_gradient_room1
```

Each branch writes a compact metadata record into the run JSON and its captured
gradient tensors under `first_backward_gradients/`. The run validity gate
requires paired direct/separable branches to have matching active-index hashes
and gradient tensor shapes.

For trajectory diagnostics after a saved checkpoint, use the paired rollout
runner. `--freeze_scheduler_profiling` disables measured-latency updates to
empirical scheduler costs for an additional isolation condition:

```bash
python experiments/run_ssim_checkpoint_rollout.py \
  --scene replica_room1 --seed 280 --snapshot_frame 30 \
  --rollout_frames 30 --repeats_per_implementation 3 \
  --checkpoint_dir results/phase14_generalization_postfix/room1_coverage_saturating_holdout_seeds279_281 \
  --expected_snapshot_sha256 58d0d40ce99b90e962ba469eaf1e95877745b78bbaab2f67affb66dacd7691a6 \
  --freeze_scheduler_profiling \
  --out_dir results/ssim_paired_rollout_room1
```

Before each frame, the runner restores one common global RNG state for every
branch; the first direct repeat determines the next frame's RNG state. Keep
this common-random-number protocol and the profiling-feedback setting in the
result interpretation.

### Locked-decision 30-frame paired SSIM rollout

To isolate SSIM-gradient effects from map growth and scheduler reselection, run the
checkpoint rollout with `--freeze_spawn_and_update_decisions`. The direct canonical
branch records a persistent-ID to optimizer-allocation schedule; both implementations
replay that schedule with spawning, pruning, and periodic compaction disabled. Runtime
gates verify the checkpoint hash, persistent-ID roster, zero births, and per-frame
update count. Periodic compaction is disabled only in this fixed-decision mode so
persistent IDs remain comparable across frame 100.

Example for room1 seed 279 (frame-30 snapshot):

```bash
python experiments/run_ssim_checkpoint_rollout.py \
  --scene replica_room1 --seed 279 --snapshot_frame 30 --rollout_frames 30 \
  --repeats_per_implementation 3 \
  --checkpoint_dir results/phase14_generalization_postfix/room1_coverage_saturating_holdout_seeds279_281 \
  --expected_snapshot_sha256 5aea36a52691d73aa8e7ffab0ca9f201b5dc55925600945aec437996b4ce4999 \
  --freeze_spawn_and_update_decisions \
  --out_dir results/ssim_rollout_locked_room1_seed279
```

Use the seed-281 snapshot hash `6df235b38ce3f30946197660f5f1c5158b383ae263438a8ad633f032f8c5b718`
for the independent seed-281 replay. These records were produced from a dirty
working tree (`47fcdba…-dirty`); treat them as diagnostic evidence, not a clean-checkout
reproduction until rerun from a frozen revision. See the [paired rollout report](../results/phase14_generalization_postfix/ssim_fixed_decision_rollout_report.md)
for results and limitations.

The clean frozen-source rerun is archived in `results/phase14_generalization_postfix/ssim_frozen_source_snapshot.bundle`; its commit is recorded in `ssim_clean_checkout_provenance.json`. Both seed reruns passed the locked-decision runtime gates.

For the cross-scene office1 replay, use `--snapshot_frame 90 --rollout_frames 29`
(the sequence ends at frame 119), the seed-specific map hash from
`ssim_office1_clean_checkout_provenance.json`, and checkpoint directory
`results/phase14_generalization_postfix/replica_office1_timing_controlled_newseeds264_266`.
The corresponding clean source is archived in
`results/phase14_generalization_postfix/ssim_office1_frozen_source_snapshot.bundle`.

To summarize all current room1 and office1 fixed-decision runs without treating
frames as independent samples, run:

```bash
python experiments/analyze_ssim_locked_rollouts.py
```

This writes the per-checkpoint and scene-level descriptive summary to
`results/phase14_generalization_postfix/ssim_locked_rollout_analysis.md` and `.json`.

For E2 timing controls, `experiments/run_matched_dilution_controls.py` accepts
`--e2_target_spawns N` to freeze one total spawn quota across all selected
scenes and seeds. Without it, the runner estimates a separate total from each
seed's natural pilot. The Replica exploratory runs and analysis are documented
in `results/phase14_generalization_postfix/replica_e2_timing_pilot_report.md`
and `replica_e2_timing_fixed50k_report.md`. The 50k cohort was rerun from a clean
frozen source bundle; see `replica_e2_timing_cross_dose_report.md` and
`replica_e2_fixed50k_clean_provenance.json` for the comparison and provenance.
Use `--e2_candidate_mode error_first` for an error-mask-first proposal pool with
depth-valid fallback. The four-seed-per-scene clean comparison against
`uniform_depth` is in `replica_e2_timing_candidate_comparison.md`; regenerate its
aggregate with `python experiments/analyze_replica_e2_candidate_transfer.py`.
