# E0: RGB-D metric, pose, and budget audit

**Status:** metric and input-path audit implemented; one-seed replay is diagnostic only.

**Protocol:** `rgbd-metrics-v2`, replay artifact `results/mechanism_v2/metric_v2_final_arrays/replica_room2/seed57_final_arrays.json`.
**Validation:** `results/mechanism_v2/metric_v2_final_arrays/validation.json` validates 1/1 matched pair and confirms current source hashes.

## Findings

### Depth semantics and evaluation

The gsplat production path requests `RGB+D`. In gsplat 1.5.3, that depth is alpha-weighted accumulated depth, not expected camera-z depth. The previous evaluator treated it as ordinary depth, which biased depth errors toward zero in partially transparent pixels. Renderer output now exposes both forms: `depth` is accumulated depth divided by alpha where supported, and `depth_accumulated` is retained for correct frozen-background compositing. A numerical audit over saved held-out views confirms `depth_accumulated ≈ depth × alpha` (largest per-view median absolute discrepancy below `5e-8 m`).

Coverage is now defined using rendered alpha `≥ 0.5`, rather than `depth > 0`; tiny nonzero alpha is not evidence of useful map support. The GT-valid depth L1 treats unsupported predictions as zero, while overlap L1/RMSE are reported separately as diagnostics. This keeps low coverage from making an arm look artificially accurate.

### SSIM semantics

The old `ssim` value used global image statistics and was not spatial-window SSIM. It is preserved under `ssim_legacy`; canonical `ssim` now uses torchmetrics' Gaussian-window spatial SSIM with `data_range=1.0`. Historical SSIM values must not be compared directly with the new standard metric.

### Replica pose and depth input path

`scripts/download_replica_scene.py` converts 16-bit depth to meters by dividing by `6553.5`. It reads Replica `traj.txt` as camera-to-world and stores its inverse as `extrinsics.npy` (world-to-camera). On the local `room2` arrays, all 120 matrices have valid homogeneous bottom rows and rotation orthogonality error at most `1.2e-7`; `W2C × C2W` round-trip error is at most `2.4e-7`. The phase-10 sequence loader passes those W2C poses into mapping; the matched replay passes the same per-frame pose into held-out evaluation. Intrinsics are scaled independently in x and y to the working resolution.

Replica's loader retains positive depth values without a 0.4–4.0 m clipping rule; it treats finite positive values as valid. The local room2 depth array spans 0–5.02 m, with positive-depth median 2.31 m. This differs from the TUM preprocessing path, which clips to 0.4–4.0 m, so cross-dataset depth metrics need dataset-specific validity rules stated explicitly.

### What the corrected one-seed replay says

All four arms use the same frozen prefix and candidate batch, insert exactly 5,000 candidates, and perform exactly 565,000 logical Gaussian micro-updates. The experiment validator passes all listed matching and source-integrity gates.

| Arm | Held-out PSNR (dB) | Standard SSIM | GT-valid depth L1 (m) | Overlap depth L1 (m) | Coverage | Realized stale-update share |
|---|---:|---:|---:|---:|---:|---:|
| Early global | 11.783 | 0.4864 | 1.1342 | 0.3117 | 0.620 | 0.017 |
| Late global | 14.981 | 0.6401 | 1.0515 | 1.0367 | 0.992 | 0.013 |
| Early stale rescue | 10.598 | 0.4661 | 1.2123 | 0.5259 | 0.628 | 0.296 |
| Late stale rescue | 13.727 | 0.6180 | 1.1400 | 1.0158 | 0.924 | 0.262 |

This audit does **not** support a positive stale-rescue effect on this seed: the rescue arms have lower held-out PSNR and higher GT-valid depth L1 than their matching global arms. The requested 50% stale quota translated to 26–30% realized stale updates in rescue arms. Aged/stale status alone is therefore not evidence of useful update value.

Depth failure has at least two distinct patterns. Early-global coverage is only 0.62, but overlap L1 is 0.31 m. Late-global coverage reaches 0.99, yet overlap L1 is 1.04 m; missing support alone cannot explain the error. Per-view depth correlation also varies substantially, including near-zero or negative values on some held-out views. Pose matrix conventions and depth units pass the static input audit, but this does not rule out temporal pose alignment, scene calibration, or mapping-quality problems. Geometry conclusions should remain provisional until visual review and independent geometry metrics are added.

### Logical updates versus measured compute

The four arms each perform 565,000 logical updates, but their measured mean CUDA optimization-stage time is 61.19–62.78 ms per mapped frame; synchronized stage wall time is 61.21–62.80 ms, and end-to-end frame wall time is 71.53–72.66 ms. The configured 15 ms value is a modeled optimization budget. Equal logical update counts do not establish equal actual work or a 15 ms end-to-end deadline.

## Code and artifacts

Metric definitions and renderer behavior are in `research/mechanism/metrics.py`, `research/rasterizer.py`, and `research/background_cache.py`. The replay emits fixed-view RGB/depth/error/alpha arrays and timing fields. Unit tests cover standard SSIM, valid/overlap depth, expected versus accumulated depth, and cache compositing.

Reproduce the audit from the repository root:

```bash
python experiments/validate_matched_candidate_timing_replay.py \
  --root results/mechanism_v2/metric_v2_final_arrays \
  --scenes replica_room2 --seeds 57 --run_ids final_arrays \
  --out results/mechanism_v2/metric_v2_final_arrays/validation.json

python experiments/mechanism/analyze_depth_metric_audit.py \
  --run results/mechanism_v2/metric_v2_final_arrays/replica_room2/seed57_final_arrays.json \
  --visualization-root results/mechanism_v2/metric_v2_final_figures \
  --out-dir results/mechanism_v2/metric_v2_final_arrays
```

This is a substrate correction and one-seed diagnostic. It is not confirmatory causal evidence, and its values must not be compared with legacy runs that used the former depth or SSIM definitions. The next scientific step is to review per-view depth visualizations and repeat the corrected audit across scenes/seeds before interpreting starvation interventions.
