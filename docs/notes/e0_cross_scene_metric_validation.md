# E0 cross-scene metric validation

**Purpose:** check that the corrected RGB-D metrics separate missing render support from depth error where the map renders. This is a measurement audit, not a generalization or causal result.

**Protocol caveat:** the saved metric-v2 replay records pass their original validator, but the runner resumed a frame-29 checkpoint and then replayed frames beginning at frame 1. That is not a chronological online trajectory. Consequently, the scene-level quality values below are retained as diagnostics of metric separation only; they are invalid as mapping-quality or intervention evidence and must not be cited as confirmatory results.

## Protocol and provenance

- Source snapshot: commit `03c4c2e` (`research: audit RGB-D metrics and compute timing`).
- Scenes and seeds: Replica office0 and room2, seeds 57 and 58 (four matched pairs).
- 99 online frames; same candidate batch size (5,000), candidate observation at frame 30, insertion at frame 30 or 70, and 5,000 logical micro-updates per mapped frame.
- Fixed evaluation frames: 20, 40, 60, 80, and 99.
- All four pairs pass candidate, prefix, update-count, metric-v2, source-hash, and CUDA-timing validation in `results/mechanism_v2/metric_v2_cross_scene_n99/validation_against_e0_commit.json`.

## Held-out mean results

Each cell averages two seeds. These are descriptive means; two seeds do not establish population uncertainty.

| Scene | Insert timing | PSNR (dB) | Spatial SSIM | GT-valid depth L1 (m) | Overlap depth L1 (m) | Alpha coverage |
|---|---|---:|---:|---:|---:|---:|
| Replica office0 | Early | 12.875 | 0.324 | 1.303 | 0.051 | 0.400 |
| Replica office0 | Late | 12.895 | 0.311 | 1.333 | 0.049 | 0.390 |
| Replica room2 | Early | 11.675 | 0.493 | 1.218 | 0.391 | 0.640 |
| Replica room2 | Late | 14.215 | 0.642 | 1.206 | 1.127 | 0.950 |

## Interpretation

The split metrics expose different failure patterns in this constrained candidate replay. In office0, only about 39–40% of GT-valid pixels receive alpha support, so the large GT-valid depth L1 is mostly a coverage problem; the small overlap L1 describes only a minority of pixels and must not be used alone. In room2 late insertion, coverage is about 95% but overlap L1 remains about 1.13 m, indicating substantial depth mismatch in rendered regions as well. Automatic spawning is disabled in this runner and only one frozen 5,000-Gaussian batch is inserted; low coverage is therefore expected and is not a clean diagnostic of ordinary mapping or pose quality.

The corrected depth metrics therefore do not reduce to one scalar geometry verdict. Future results should report GT-valid error, overlap error, and support coverage together. Before treating these RGB-D values as geometric accuracy, add a geometry metric and review the corresponding rendered-depth maps. The replay remains a one-batch intervention on known poses and does not validate long-horizon online SLAM.
