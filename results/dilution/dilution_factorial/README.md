# Corrected-pose dilution study

Run `python experiments/run_dilution_factorial.py` for the two primary TUM
scenes, 150 evaluated frames, seeds 42–49. Each run is an independent JSON
containing its full config and frame trace; rerunning skips completed files.
`python experiments/analyze_dilution_factorial.py` produces paired contrasts,
seed-bootstrap intervals, and a frame-65 diagnostic.

Newly generated runs also record a `manifest.json` with code revision,
configuration hash, dataset, pose convention, seed set, and renderer backend.
The run-level manifest is marked exploratory while the worktree is dirty. Use
`python scripts/audit_experiment_manifests.py` to classify manifest provenance.

Factorial levels are `error_only` and `error_influence_temporal`, crossed with
throttle off, coverage only, and coverage plus active backlog. The historical
headline `ours` config had warmup disabled, which also disabled backlog
counting. The new `enable_backlog_throttling` option evaluates backlog
independently while keeping warmup selection off.

`error_budget4` raises only the modeled optimization budget from 15 to 60 ms;
its densification allowance stays at the 15 ms setting. It is a mechanism
control, not a matched-compute competitor. `error_cap23k` is a hard population
limit on both scenes. `error_cap39k` checks a cap near the coverage map size on
`fr1_xyz`; it is absent from `fr2_xyz`.

The two exploratory scenes, `fr1_desk` and `fr3_sitting_static`, use three
seeds and the six factorial cells. Depth L1 is a rendered-depth error in
meters, not 3D surface accuracy. `n_spawned` counts actual additions before
pruning; `n_new` is the net population change. Age is measured from each
Gaussian's creation frame, with `<10` and `>=10` frame groups. Budget shares
sum modeled costs for selected Gaussians in each group. Per-group selected and
population counts permit the per-Gaussian selection rate to be computed.

New summaries additionally report MatureUpdateCoverage (mature means age >=10
frames; selected mature count divided by mature eligible population, summed over
evaluated frames), GrowthPressure (net new Gaussians divided by active map size
per frame), and ComputeReturn (consecutive-frame PSNR change divided by
optimization kernel time in ms). ComputeReturn is an observational trajectory
proxy and may be negative; it does not identify the causal return of an individual
spawn/refine action. The existing population cap and 4x-budget conditions do not
constitute E1 matched-map-size or E2 matched-spawn-count controls. Those causal
comparisons remain open and must use common trajectory, initialization, seed,
and pose protocol before supporting a paper claim.

`python experiments/run_matched_dilution_controls.py` runs the next controls to
`results/matched_dilution_controls/`. E1 runs a fresh coverage pilot for each
scene/seed, then caps both arms at that pilot's final reported Gaussian count.
The pair is accepted only if both finish at that same count; this is a
**matched final map size** test, not a frame-by-frame matched-size test. E2
takes a fresh unthrottled pilot's total `n_spawned` as the common target for
early frames 1–75 versus late frames 76–150. Both arms sample uniformly from
depth-valid pixels so their spawn location proposal is consistent across the
timing intervention; the trace records how many sampled locations fell outside
the ordinary error mask. The pair is accepted only if both arms realize the
exact total. Invalid pairs remain in the output with their realized counts.
Report off-mask spawn counts because this is a controlled timing experiment,
not the ordinary error-mask densification policy.
