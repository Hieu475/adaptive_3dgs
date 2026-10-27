# Stale-mature rescue factorial: independent-seed replication

Seed57 contributes one seed-level estimate formed by averaging its three technical repeats. Seeds58–60 each contribute one independently initialized prefix and candidate batch. Thus n=4 seeds per scene; technical repeats are never counted as independent samples.

All factorial arms used 5,000 candidate Gaussians and exactly 565,000 micro-updates. Rescue requested a 50% stale-mature quota; the realized share is reported separately. Validation passed 6/6 new seed-scene records.

## Seed-level difference-in-differences

DiD = (late−early under stale rescue) − (late−early under global selection). Positive values mean rescue makes the late-vs-early timing effect more positive for that metric.

## replica_room2

| Seed | DiD PSNR (dB) | DiD Depth L1 (m) | DiD SSIM | Rescue−global PSNR early | Rescue−global PSNR late | Rescue stale share (early / late) |
|---:|---:|---:|---:|---:|---:|---:|
| 57 | -0.0354 | +0.03753 | +0.13609 | +1.7660 | +1.7306 | 0.288 / 0.262 |
| 58 | -0.2123 | +0.04401 | +0.13189 | +2.0880 | +1.8758 | 0.289 / 0.263 |
| 59 | -0.9446 | -0.00244 | +0.00046 | +2.5120 | +1.5675 | 0.288 / 0.262 |
| 60 | +0.1804 | +0.01928 | +0.16226 | +1.6631 | +1.8435 | 0.288 / 0.262 |

Seed-level mean ± SD across n=4 seeds (ranges are across seeds, not CIs):
- DiD PSNR: -0.2530 ± 0.4882 dB; range [-0.9446, +0.1804], sign 1 positive / 3 negative.
- DiD Depth L1: +0.02460 ± 0.02084 m; range [-0.00244, +0.04401].
- Rescue−global held-out PSNR at early insertion: +2.0073 dB (4/4 seeds positive); at late insertion: +1.7543 dB (4/4 positive).

## replica_office3

| Seed | DiD PSNR (dB) | DiD Depth L1 (m) | DiD SSIM | Rescue−global PSNR early | Rescue−global PSNR late | Rescue stale share (early / late) |
|---:|---:|---:|---:|---:|---:|---:|
| 57 | -1.2867 | +0.02434 | -0.00018 | +1.2061 | -0.0806 | 0.277 / 0.249 |
| 58 | -0.3186 | +0.04817 | +0.04550 | +0.3902 | +0.0716 | 0.278 / 0.249 |
| 59 | +0.5751 | +0.02467 | +0.06397 | -0.9468 | -0.3717 | 0.278 / 0.249 |
| 60 | -0.0456 | +0.04909 | +0.06412 | +0.0807 | +0.0351 | 0.278 / 0.249 |

Seed-level mean ± SD across n=4 seeds (ranges are across seeds, not CIs):
- DiD PSNR: -0.2689 ± 0.7747 dB; range [-1.2867, +0.5751], sign 1 positive / 3 negative.
- DiD Depth L1: +0.03657 ± 0.01394 m; range [+0.02434, +0.04909].
- Rescue−global held-out PSNR at early insertion: +0.1825 dB (3/4 seeds positive); at late insertion: -0.0864 dB (2/4 positive).

## Interpretation

This replication tests whether stale-mature reallocation changes the timing effect under a frozen-batch replay, not whether natural map growth is the sole cause of quality loss. Read PSNR, SSIM, and depth together: an improvement in one metric can coincide with regression in another. Four seeds per scene are useful replication evidence but remain too few for a strong generalization claim; do not interpret these summaries as a paper-ready significance test.

Raw runs and frozen artifacts are in `matched_candidate_timing_rescue_pilot/` and `matched_candidate_timing_rescue_replication/` under results.
