# Map Growth and Policy Routing in Budgeted Online RGB-D 3D Gaussian Splatting

Research code and frozen result bundles for a controlled study of online RGB-D 3DGS under a **modeled 15 ms per-frame optimization budget**.
The paper is organized around one claim (compute dilution); auxiliary analyses live in the supplement. Source is in [`docs/paper/iccv_official/`](docs/paper/iccv_official/) (`main.tex`, `supp.tex`, `refs.bib`).

> **Scope.** This is a mapping study with ground-truth poses (tracking decoupled), 150-frame horizons at 320×240, on TUM RGB-D and Replica.
> The 15 ms figure is *modeled optimization time*, not end-to-end latency; measured throughput is about 10 FPS on an RTX 4050 Laptop GPU.
> Seeds measure repeatability on fixed trajectories, not generalization across scenes.

## 1. What the evidence supports

| # | Claim | Evidence (frozen artifact) |
|---|---|---|
| 1 | **Compute dilution.** As the map grows, established Gaussians receive a smaller share of a fixed optimization budget. Coverage throttling lifts raw-error selection by +3.33 / +1.77 dB (fr2_xyz / fr1_xyz); a 4× optimization budget with the original spawn allowance lifts it by +3.29 / +3.07 dB; mature-Gaussian selection rate on fr2_xyz rises from 0.36 to 0.82. | [`results/dilution_factorial/`](results/dilution_factorial/) |
| 2 | **Combined policy** (temporal×influence selection + coverage throttling) reaches **27.64 dB** on TUM fr2_xyz, +2.61 dB over the strongest budgeted baseline and +4.48 dB over the RTG-SLAM-policy re-implementation. It recovers 97.6% of headroom above no-op (92.5% above the strongest budgeted baseline); FULL is 0.21 dB higher with a 3.8× larger map. | [`results/phase14_projected_area_fix_clean_7policy_8seeds/`](results/phase14_projected_area_fix_clean_7policy_8seeds/) |
| 3 | **Counterfactual utility.** Single-Gaussian interventions have negative measured utility in 2.7–38.7% of cases depending on geometry stratum (n = 75 per stratum); uniform-random selection on the same pools has an expected 20.3% negative rate. | `results/oracle*/` |
| 4 | **Negative results.** A tested offline utility model degrades closed-loop PSNR by 2.51 dB (pre-fix labels, one protocol); contextual re-ranking gives no online gain at large overhead. | supplement §F |
| 5 | **Router.** A frozen hole-rate router separates TUM from Replica, but within-scene depth interventions do **not** reverse the policy ranking, so hole rate is a dataset-associated cue, not an established sensor-noise mechanism. | `results/depth_intervention/` |

**Known weaknesses (stated in the paper).** The combined policy loses mean PSNR to raw-error selection on 10 of 14 sequences, including all eight Replica scenes; coverage also changes densification thresholds, so map size is not isolated as the sole cause; offline density-control methods and full SLAM integration are not compared.

The single canonical result bundle is `phase14_projected_area_fix_clean_7policy_8seeds`. Older bundles (`phase14_corrected`, `phase13_frozen_benchmark`, `phase10_e2e`, `final_confirmation`) are historical; the pose-bugged numbers in `results/final_confirmation/DEPRECATED.md` must not be compared with current results.

### Main benchmark (TUM fr2_xyz, 150 frames × 8 seeds, B = 15 ms modeled)

| Policy | Mean PSNR | Final PSNR | SSIM | Map |
|:--|--:|--:|--:|--:|
| no_op | 19.06 | 14.88 | 0.6920 | 23,523 |
| error_only | 24.04 | 20.12 | 0.8838 | 73,383 |
| error_influence | 25.03 | 21.83 | 0.9083 | 105,438 |
| rtg_slam_reimpl | 23.16 | 20.04 | 0.8669 | 65,960 |
| **OURS** | **27.64** | **28.82** | **0.9231** | **23,765** |
| FULL (unconstrained ceiling) | 27.85 | 29.34 | 0.9245 | 90,582 |

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

Three tests that failed at the previous `HEAD` were fixed during the cleanup: `format_budget_table` now prints `n/a` when no violation rate was measured, and the two `test_phase13_frozen_benchmark.py` fixtures were updated for fields the production code now requires (`*_depth_l1_pre`, `config_sha256`) and a renamed report section.

## 4. Repository layout

| Path | Contents |
|---|---|
| `research/` | Library: scheduler, attribution, losses, pipeline, utility models |
| `experiments/` | Runnable experiment drivers and analysis scripts referenced by the paper or by result manifests |
| `research_archive/` | Exploratory analysis scripts not referenced by the paper, tests or manifests (see its README) |
| `paper_repro/` | Table/figure → command index and the number-consistency checker |
| `results/` | Frozen result bundles (git-ignored except manifests and READMEs) |
| `docs/paper/` | Paper source (`iccv_official/`), figure scripts, `legacy/` superseded drafts |
| `tests/` | Unit and integration tests |
| `cuda/`, `cpp/` | Prototype kernels (not used in reported results) |

## 5. Development history

The repository began as a learned-utility project (phases 1–11: oracle utility, TwoHeadMLP, closed-loop integration) and was reoriented after pose-convention and attribution bugs were found. The phase-labelled results and the tags `phase10-frozen`, `phase11-frozen`, `phase13-frozen` document that history only; they are **not** evidence for the causal claims above.

## 6. Citation

Anonymous under review. A BibTeX entry will be added after the review period.
