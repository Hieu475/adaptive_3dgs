# Published-SOTA contextual comparison (NOT apples-to-apples)

> [!CAUTION]
> Protocol mismatch is fundamental: published numbers below are full-sequence
> SLAM systems (tracking + mapping + often global refinement, full resolution,
> training-view or novel-view rendering per each paper's own protocol), while
> ours are **budgeted online mapping** (15 ms modeled budget, GT poses, 320x240,
> 99-150 frames, no global refinement). This table is for context only --- it
> answers "are we in a sane range?" not "do we beat SOTA?".

## Replica (per-scene PSNR; SOTA = training-view rendering, full sequences)

| Scene | OURS (fixed, ~100f) | error_only (routed) | Point-SLAM | MonoGS RGB-D | RTG-SLAM | SplaTAM |
|---|---|---|---|---|---|---|
| room0 | 20.85 | 22.78 | 32.40 | 34.83 | 28.49 | — |
| room1 | 20.42 | 24.29 | 34.08 | 36.43 | 31.27 | — |
| room2 | 22.85 | 25.88 | 35.50 | 37.49 | 32.96 | — |
| office0 | 26.85 (150f) | 30.17 (150f) | 38.26 | 39.95 | 37.32 | — |
| office1 | 29.29 | 33.48 | 39.16 | 42.09 | 36.12 | — |
| office2 | 22.08 | 24.67 | 33.99 | 36.24 | 31.14 | — |
| office3 | 22.45 | 24.47 | 33.48 | 36.70 | 31.19 | — |
| office4 | 23.57 | 25.93 | 33.49 | 37.06 | 33.81 | — |

Sources: Point-SLAM/MonoGS RGB-D Replica averages (MonoGS++ Tab.1: Point-SLAM avg 35.17,
MonoGS avg 37.50); RTG-SLAM per-scene (SEGS-SLAM ICCV'25 suppl. Tab.7: avg 32.79).
Honest reading: full SOTA systems with global optimization + full sequences score
~33-42 dB; our budgeted online mapping scores ~20-33 dB. The gap is expected
(no global refinement, 15 ms budget, short horizons) and is NOT a defeat ---
our claim is compute-allocation efficiency under budget (98% of *our* ceiling),
not absolute SOTA rendering.

## TUM fr2_xyz (PSNR)

| Method | PSNR | Notes |
|---|---|---|
| OURS (ours) | 27.67 (150f x8, budgeted) | this work |
| SplaTAM | 24.50 (full seq) | CaRtGS Tab.II / SplatMAP compilation |
| Photo-SLAM | 21.07 (full seq) | SplatMAP Tab.3 |
| MonoGS | 16.17 (full seq, monocular-ish eval) | SplatMAP Tab.3 |
| RTG-SLAM | 17.08 (full seq) | SEGS-SLAM suppl. |

On fr2_xyz our budgeted mapping PSNR exceeds published full-system numbers,
but protocol differences (frames, resolution, train-vs-novel views) forbid a
superiority claim --- report as "competitive range" only.

## TUM ATE (cm RMSE, published reference for our tracker baseline)

| Method | fr1_desk | fr2_xyz | fr3_office |
|---|---|---|---|
| Ours (dense GN baseline VO, 30f) | — | 0.8 | — |
| GS-SLAM | 3.3 | 1.3 | 6.6 |
| Point-SLAM | 2.6 | 1.3 | 3.2 |
| SplaTAM | 3.35 | 1.24 | 5.16 |
| MonoGS | 1.50 | 1.44 | 1.49 |
| Photo-SLAM | 2.60 | 0.35 | 1.00 |

Sources: PointSLAM++ Tab.3 / GS-SLAM Tab.2. Our 30f VO numbers are short-horizon
sanity checks, not full-trajectory SLAM tracking --- labeled as such.
