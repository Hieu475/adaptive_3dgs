# Paper reproduction index

Maps every table/figure of `docs/paper/iccv_official/{main,supp}.tex` to the result bundle and the command that regenerates it.
GPU runs are long (RTX 4050: tens of minutes per scene × seeds). The paper numbers are already frozen in `results/`; most readers only need step 0.

**Step 0 (CPU, seconds):** `python paper_repro/check_numbers.py` recomputes the headline numbers from the frozen JSON and checks that `main.tex` and `README.md` contain them (and none of the superseded values).

| Paper item | Result bundle | Command(s) (run from repo root) | Status |
|---|---|---|---|
| Tab. 1 main benchmark, Fig. 2, Fig. 4 | `results/phase14_projected_area_fix_clean_7policy_8seeds/` | `bash paper_repro/reproduce.sh main` | pinned (from paper text) |
| Tab. 2 factorial + dilution controls, Fig. 7 | `results/dilution_factorial/` | `bash paper_repro/reproduce.sh factorial` | pinned |
| Tab. 4 depth interventions | `results/depth_intervention/` | `bash paper_repro/reproduce.sh depth` | pinned |
| Tab. 3 / 7-scene routing, mixed regime | `results/phase14_adaptive_perframe/` | `bash paper_repro/reproduce.sh adaptive` | pinned (script defaults); per-scene 7-scene bundle `results/phase14_multiscene150/` **not pinned** |
| Holdout proxy (§6) | `results/phase14_holdout/` | `bash paper_repro/reproduce.sh holdout` | pinned |
| Baseline VO / ATE (Supp. Tab. S3) | `results/phase14_tracking100/` | `bash paper_repro/reproduce.sh tracking` | pinned |
| Figures 1–7 (PDF) | `docs/paper/figs/` | `python docs/paper/gen_iccv_figs.py` | pinned; Fig. 5 uses `docs/paper/gen_qual_fig_v4.py` (unverified) |
| Tab. 5 per-scene breakdown, Tab. confirm2 | `results/phase14_multiscene150/` | – | **command not pinned**; see bundle manifests |
| Counterfactual-utility numbers (§1) | `results/oracle*/` | – | **command not pinned**; historical-phase scripts in `experiments/run_oracle_*.py` |

"Pinned" means the command line was taken from the script's own argument parser or the paper's reproducibility paragraph; it was **not** re-executed during the 2026-11-06 cleanup (GPU time).
"Not pinned" means the bundle exists but no single verified command is recorded yet; fill these in before submission.

Scripts moved out of `experiments/` into `research_archive/` during the cleanup are those that nothing in the paper, tests, or result manifests references; moving them does not change any frozen artifact.
