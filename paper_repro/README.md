# Paper reproduction index

Maps every table/figure of `docs/paper/iccv_official/{main,supp}.tex` (main text is organized around one claim, compute dilution; router, counterfactual and negative results are in the supplement) to the result bundle and the command that regenerates it.
GPU runs are long (RTX 4050: tens of minutes per scene × seeds). The paper numbers are already frozen in `results/`; most readers only need step 0.

**Step 0 (CPU, seconds):** `python paper_repro/check_numbers.py` recomputes the headline numbers from the frozen JSON and checks that `main.tex` and `README.md` contain them (and none of the superseded values).

| Paper item | Result bundle | Command(s) (run from repo root) | Status |
|---|---|---|---|
| Tab. 1 main benchmark, Figs. 2–3 | `results/rerun_paper1/main/` | `bash paper_repro/reproduce.sh main` (from tag `paper-rerun-1`), then `tables`, `figs` | executed 2026-11 from the tagged commit; `SOURCE.txt` records the commit |
| Tab. 2 factorial + dilution controls, Fig. 1 | `results/rerun_paper1/dilution_factorial/` | same command (`main` runs both); `figs` regenerates Fig. 1 | executed from the tagged commit |
| Supp. Tab. S5 depth interventions | `results/depth_intervention/` | `bash paper_repro/reproduce.sh depth` | pinned |
| Supp. Tabs. S4, S6 (router, 7-scene audit), mixed regime | `results/phase14_adaptive_perframe/` | `bash paper_repro/reproduce.sh adaptive` | pinned (script defaults) |
| Tabs. 4–5 Replica factorial, map-proportional budget, dose-response | `results/dilution_extension/`, `results/dilution_dose/`, `results/dilution_dose_replica_cap/` | `bash paper_repro/reproduce.sh extension`, `dose`, then `tables` | executed during the 2026-11 extension (not re-run since); tables are generated from the JSON by `make_tables.py` |
| Tab. 6 scene-level (15 scenes), Tab. 7 long horizon | `results/dilution_scenes/`, `results/dilution_long/` | `bash paper_repro/reproduce.sh scenes_budget`, `long`, then `tables` | executed 2026-11; the `n_frames` argument is a request, and Replica copies only provide 99 (office0) and 119 (office1) frames |
| Tab. 8 decomposition of coverage throttling | `results/throttle_ablation/` | `bash paper_repro/reproduce.sh ablation`, then `tables` | executed 2026-11 (4 seeds, 4 scenes) |
| Tab. 9 held-out views | `results/holdout_extension/` | `bash paper_repro/reproduce.sh holdout_ext`, then `tables` | executed 2026-11 (4 seeds, 4 scenes) |
| Tab. 10 quality at measured cost | `results/compute_matched/` | `bash paper_repro/reproduce.sh compute`, then `tables` | executed 2026-11; timings from one RTX 4050 laptop GPU |
| Tab. 11 estimated poses | `results/tracked_poses/` | `bash paper_repro/reproduce.sh tracked`, then `tables` | executed 2026-11 (TUM only; tracker does not use the map) |
| Tab. 8 (main) controllers; Supp. Tabs. S14-S16 | `results/vgg_test/`, `results/vgg_fresh/`, `results/tt_pareto/`, `results/holdout_vgg/` | `bash paper_repro/reproduce.sh controllers`, then `tables` | executed 2026-11; constants frozen on `results/vgg_dev/` and `results/tt_dev/` first; timings from one RTX 4050 laptop GPU |
| Holdout proxy (§6) | `results/phase14_holdout/` | `bash paper_repro/reproduce.sh holdout` | pinned |
| Baseline VO / ATE (Supp. Tab. S3) | `results/phase14_tracking100/` | `bash paper_repro/reproduce.sh tracking` | pinned |
| Main Figs. 1–4 and Supp. Figs. S1–S3 (PDF) | `docs/paper/figs/` | `python docs/paper/gen_iccv_figs.py` | pinned; Fig. 5 uses `docs/paper/gen_qual_fig_v4.py` (unverified) |
| Tab. 3 per-scene breakdown | `results/phase14_multiscene150/` | `bash paper_repro/reproduce.sh scenes` | pinned from each bundle's recorded scene/frames/seeds/policies; the two 8-seed merge JSONs (`*_8seed.json`) have no recorded merge script |
| Counterfactual-utility numbers (§1) | `results/gate1_headroom/` | `bash paper_repro/reproduce.sh counterfactual` | pinned (scripts take no arguments); the pose-fixed regeneration protocol is described in the paper, and exact reproduction of the 75-per-stratum set was not re-verified |

"Pinned" means the command line was taken from the script's own argument parser or the paper's reproducibility paragraph; it was **not** re-executed during the 2026-11-06 cleanup (GPU time).
Remaining unpinned items: the two `*_8seed.json` merges, and Fig. 5 (`gen_qual_fig_v4.py`), which were not traced to a recorded command.

Scripts moved out of `experiments/` into `research_archive/` during the cleanup are those that nothing in the paper, tests, or result manifests references; moving them does not change any frozen artifact.

## Known provenance gap

`results/phase14_projected_area_fix_clean_7policy_8seeds/manifest.json` records its run commits (`93791b1`, `c3484a6`) and writer commit (`75865c7`).
None of these exist in this repository's history (the runs were made from a clean checkout under `/tmp`), and `research/pipeline.py` differs from the hash in the manifest
(the other seven recorded source files match the current tree byte-for-byte). The paper therefore cites source-file hashes and worktree status rather than a commit hash.
Before submission, either recover that checkout's commit and import it, or re-run `reproduce.sh main` from a tagged commit and re-freeze the bundle.

- `closedloop` target: map-based tracking in the loop (`experiments/run_closed_loop.py`, `research/map_tracker.py`); table S12 (`tab_closedloop.tex`) from `results/closed_loop/`; table S13 (`tab_countcap.tex`, `run_count_cap.py`, `results/count_cap/`) compares TT+VGG with a size-matched hard Gaussian-count cap. TUM only (tracking fails on the local Replica sequences); tracker constants were set on the development scenes fr1_desk and fr3_structure_texture_near.
