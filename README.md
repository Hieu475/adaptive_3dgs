# Map Growth and Policy Routing in Budgeted Online RGB-D 3D Gaussian Splatting

Research code and frozen result bundles for a controlled study of online RGB-D 3DGS under a **modeled 15 ms per-frame optimization budget**.
The paper is organized around one claim (compute dilution); auxiliary analyses live in the supplement. Source is in [`docs/paper/iccv_official/`](docs/paper/iccv_official/) (`main.tex`, `supp.tex`, `refs.bib`).

> **Scope.** This is a mapping study with ground-truth poses (tracking decoupled), horizons of up to 150 frames at 320×240 (TUM: 150; local Replica copies: 99–119 frames), on TUM RGB-D and Replica.
> The 15 ms figure is *modeled optimization time*, not end-to-end latency; measured throughput is about 10 FPS on an RTX 4050 Laptop GPU.
> Seeds measure repeatability on fixed trajectories, not generalization across scenes.

## 1. What the evidence supports

| # | Claim | Evidence (frozen artifact) |
|---|---|---|
| 1 | **Compute dilution as a compute-allocation effect.** Raising the optimization budget in proportion to map size (spawn allowance fixed) improves raw-error selection on all four scenes tested (+1.56 to +3.73 dB); with free growth PSNR rises monotonically with budget and tracks the fraction of mature Gaussians updated per frame (Spearman +1.00 in every scene). Coverage throttling is scene dependent and its effect comes almost entirely from its spawn cap (a fixed ×0.2 cap matches it on TUM; adding it to the proportional budget gives no gain): +3.53 / +1.71 dB on TUM fr2_xyz / fr1_xyz but −0.68 / −0.87 dB on Replica office0 / office1. | [`results/rerun_paper1/`](results/rerun_paper1/) (factorial), `results/dilution_extension/`, `results/dilution_dose*/` |
| 2 | **Combined policy** (temporal×influence selection + coverage throttling) reaches **27.74 dB** on TUM fr2_xyz, +2.36 dB over the strongest budgeted baseline and +4.21 dB over the RTG-SLAM-policy re-implementation. It recovers 98.3% of headroom above no-op (94.0% above the strongest budgeted baseline); FULL is 0.15 dB higher with a 3.2× larger map. | [`results/rerun_paper1/main/`](results/rerun_paper1/main/) |
| 3 | **Counterfactual utility.** Single-Gaussian interventions have negative measured utility in 2.7–38.7% of cases depending on geometry stratum (n = 75 per stratum); uniform-random selection on the same pools has an expected 20.3% negative rate. | `results/oracle*/` |
| 4 | **Negative results.** A tested offline utility model degrades closed-loop PSNR by 2.51 dB (pre-fix labels, one protocol); contextual re-ranking gives no online gain at large overhead. | supplement §F |
| 5 | **Router.** A frozen hole-rate router separates TUM from Replica, but within-scene depth interventions do **not** reverse the policy ranking, so hole rate is a dataset-associated cue, not an established sensor-noise mechanism. | `results/depth_intervention/` |

**Known weaknesses (stated in the paper).** The combined policy loses mean PSNR to raw-error selection on 10 of 14 sequences, including all eight Replica scenes; coverage also changes densification thresholds and hard caps remove needed capacity on Replica, so map size is not isolated as the sole cause, and more budget helps any compute-limited optimizer; offline density-control methods and full SLAM integration are not compared.

**Method (new).** Two label-free controllers: a *time-targeted budget* (TT, `research/time_budget.py`) that scales the optimization budget to hold the measured frame time at a target, and *value-gated growth* (VGG, `research/growth_controller.py`, `research/growth_value.py`) that renders the incoming frame with and without recently spawned Gaussians and throttles spawning when they are not worth keeping. With constants frozen on four development scenes, TT+VGG improves PSNR on all 11 held-out scenes (median +1.14, mean +1.29, worst +0.05 dB) at +4 ms mean frame time and on five fresh frame windows (+0.90 dB, 5 ms faster). Gains retain 45-79% on held-out views and largely vanish when rendering from the true pose with estimated tracking. Evaluation: `bash paper_repro/reproduce.sh controllers`.

The canonical result bundles are those in `results/rerun_paper1/` (main benchmark and TUM factorial), `results/dilution_extension/` and `results/dilution_dose*/`, all produced from the commit tagged `paper-rerun-1` (see `results/rerun_paper1/SOURCE.txt`) with the corrected evaluator (standard Gaussian-window SSIM; alpha-normalized rendered depth). Older bundles (`phase14_projected_area_fix_clean_7policy_8seeds`, `phase14_corrected`, `phase13_frozen_benchmark`, `phase10_e2e`, `final_confirmation`) are historical and their SSIM/depth values are not comparable; the pose-bugged numbers in `results/final_confirmation/DEPRECATED.md` must not be compared with current results.

### Main benchmark (TUM fr2_xyz, 150 frames × 8 seeds, B = 15 ms modeled)

| Policy | Mean PSNR | Final PSNR | SSIM | Map |
|:--|--:|--:|--:|--:|
| no_op | 18.85 | 14.75 | 0.4655 | 21,128 |
| error_only | 24.21 | 20.20 | 0.7023 | 72,366 |
| error_influence | 25.38 | 22.35 | 0.7338 | 86,988 |
| rtg_slam_reimpl | 23.53 | 20.57 | 0.6833 | 60,050 |
| **OURS** | **27.74** | **28.73** | **0.7978** | **21,642** |
| FULL (unconstrained ceiling) | 27.89 | 29.29 | 0.8036 | 70,295 |

Paired Wilcoxon over eight per-seed summaries; Holm-adjusted p = 0.0469 for every contrast (this is the minimum attainable with n = 8). Because trajectories and poses are fixed, treat seed-level effect sizes as repeatability, and scenes as the unit of generalization.

## 2. Method in one paragraph

Per frame, candidate Gaussians are scored (error × influence mass, optionally with a temporal EMA), packed by value density under the modeled budget with a decoupled cost model `C_i(K) = C_base + K·C_step`, and optimized for K = 5 microsteps. Fast Approximate Attribution (center-sampled projection) replaces exact attribution at Spearman ρ = 0.91 against it. **Coverage throttling** reduces the per-frame spawn allowance from 1.0 to 0.2 as coverage κ (fraction of valid pixels with transmission < 0.5) goes from 0.80 to 0.90. The learned-utility branch (TwoHeadMLP) is retained only as a documented negative result.

## 3. Reproducing the paper

Everything needed to regenerate or verify each table and figure is indexed in [`paper_repro/`](paper_repro/):

```bash
# CPU-only: verify that paper + README numbers match the frozen result bundles
python paper_repro/check_numbers.py

# See which command regenerates which table/figure (GPU runs are long)
cat paper_repro/README.md
bash paper_repro/reproduce.sh list
```

Build the paper: `cd docs/paper/iccv_official && latexmk -pdf main.tex supp.tex` (figures live in `docs/paper/figs`, generated by `docs/paper/gen_iccv_figs.py`).

Environment: Python 3.12, PyTorch 2.11 + CUDA 12.8, `gsplat` 1.5.3 (production rasterizer). See [`docs/environment.md`](docs/environment.md), [`docs/dataset.md`](docs/dataset.md) and [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md). The `cuda/*.cu` files are research prototypes and are not used in any reported result ([`cuda/README_PROTOTYPE.md`](cuda/README_PROTOTYPE.md)).

### Tests

```bash
pytest tests -q   # 521 passed
```

Three tests that failed at the previous `HEAD` were fixed during the cleanup: `format_budget_table` now prints `n/a` when no violation rate was measured, and the two `test_frozen_benchmark.py` fixtures were updated for fields the production code now requires (`*_depth_l1_pre`, `config_sha256`) and a renamed report section.

## 4. Repository layout

| Path | Contents |
|---|---|
| `research/` | Library: scheduler, attribution, losses, pipeline, utility models |
| `experiments/` | 127 experiment drivers/analysis scripts that the paper, tests, scripts or result manifests reference |
| `research_archive/` | 86 exploratory or historical scripts (phases 6–12, diagnostics) referenced by nothing current; see its README |
| `docs/archive/` | Superseded proposals, phase-6 report, early artifacts, project PDFs |
| `paper_repro/` | Table/figure → command index and the number-consistency checker |
| `results/` | Frozen result bundles (git-ignored except manifests and READMEs) |
| `docs/paper/` | Paper source (`iccv_official/`), figure scripts, `legacy/` superseded drafts |
| `tests/` | Unit and integration tests |
| `cuda/`, `cpp/` | Prototype kernels (not used in reported results) |

## 5. Development history

The repository began as a learned-utility project (phases 1–11: oracle utility, TwoHeadMLP, closed-loop integration) and was reoriented after pose-convention and attribution bugs were found. Source files formerly named `phaseN_*` were renamed by function (mapping in `docs/RENAMES.json`; module map in `research/README.md`); result directories keep their original names because they are frozen data. The phase-labelled results and the tags `phase10-frozen`, `phase11-frozen`, `phase13-frozen` document that history only; they are **not** evidence for the causal claims above.

## 6. Citation

Anonymous under review. A BibTeX entry will be added after the review period.
