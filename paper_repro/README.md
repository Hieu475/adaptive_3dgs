# Paper reproduction index

Maps every table/figure of `docs/paper/iccv_official/{main,supp}.tex` (main text is organized around one claim, compute dilution; router, counterfactual and negative results are in the supplement) to the result bundle and the command that regenerates it.
GPU runs are long (RTX 4050: tens of minutes per scene × seeds). The paper numbers are already frozen in `results/`; most readers only need step 0.

**Step 0 (CPU, seconds):** `python paper_repro/check_numbers.py` recomputes the headline numbers from the frozen JSON and checks that `main.tex` and `README.md` contain them (and none of the superseded values).

| Paper item | Result bundle | Command(s) (run from repo root) | Status |
|---|---|---|---|
| Tab. 1 main benchmark, Figs. 2–3 | `results/phase14_projected_area_fix_clean_7policy_8seeds/` | `bash paper_repro/reproduce.sh main` | pinned (from paper text) |
| Tab. 2 factorial + dilution controls, Fig. 1 | `results/dilution_factorial/` | `bash paper_repro/reproduce.sh factorial` | pinned |
| Supp. Tab. S5 depth interventions | `results/depth_intervention/` | `bash paper_repro/reproduce.sh depth` | pinned |
| Supp. Tabs. S4, S6 (router, 7-scene audit), mixed regime | `results/phase14_adaptive_perframe/` | `bash paper_repro/reproduce.sh adaptive` | pinned (script defaults) |
| Tabs. 4–5 Replica factorial, map-proportional budget, dose-response | `results/dilution_extension/`, `results/dilution_dose/`, `results/dilution_dose_replica_cap/` | `bash paper_repro/reproduce.sh extension`, `dose`, then `tables` | executed during the 2026-11 extension (not re-run since); tables are generated from the JSON by `make_tables.py` |
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
