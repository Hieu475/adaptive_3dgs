# Timing × stale-mature-rescue factorial pilot

Two Replica scenes, seed57 only, three technical repeats from one frozen prefix/candidate batch per scene. The factorial crosses insertion timing (frame 30 vs 90) with update allocation (global importance vs 50% stale-mature rescue; stale means mature age ≥10 frames and last updated ≥10 frames ago). All 6 pairs passed checksum, exact-birth, zero-automatic-birth, exact-update, held-out-view, and rescue-first-stage gates.

The stale-rescue arm first assigns quota to visible stale mature Gaussians. Remaining updates go to the ordinary global-importance pool. Every arm uses 565,000 total micro-updates across 113 mapping frames; final map size is an outcome.

## Arm outcomes

Values are mean [min, max] over the three technical repeats; these are not confidence intervals.

## replica_room2

### Arm outcomes

| Arm | Held-out PSNR | Held-out Depth L1 (m) | Held-out SSIM | Stale update share | Stale coverage | Updates / arm |
|---|---:|---:|---:|---:|---:|---:|
| early_global | 11.0558 [11.0544, 11.0574] | 1.13172 [1.13141, 1.13198] | 0.07465 | 0.017 | 0.079 | 565000 |
| late_global | 11.2353 [11.2199, 11.2432] | 1.10539 [1.10482, 1.10645] | -0.00474 | 0.012 | 0.049 | 565000 |
| early_stale_rescue | 12.8217 [12.8211, 12.8224] | 1.14012 [1.14009, 1.14014] | 0.03036 | 0.288 | 0.728 | 565000 |
| late_stale_rescue | 12.9659 [12.9640, 12.9675] | 1.15132 [1.15128, 1.15138] | 0.08705 | 0.262 | 0.622 | 565000 |

### Paired contrasts

Late−early compares timing within allocation; rescue−global compares allocation within timing. DiD = (late−early under stale rescue) − (late−early under global).

| Contrast | Held-out ΔPSNR (dB) | Held-out ΔDepth L1 (m) | Held-out ΔSSIM | Δ stale update share |
|---|---:|---:|---:|---:|
| timing_global | +0.17957 [+0.16251, +0.18848] | -0.02633 [-0.02709, -0.02532] | -0.07939 | -0.0048 |
| timing_stale_rescue | +0.14421 [+0.14219, +0.14529] | +0.01120 [+0.01114, +0.01125] | +0.05670 | -0.0256 |
| rescue_early | +1.76597 [+1.76496, +1.76742] | +0.00840 [+0.00816, +0.00868] | -0.04429 | +0.2704 |
| rescue_late | +1.73061 [+1.72113, +1.74759] | +0.04593 [+0.04493, +0.04648] | +0.09180 | +0.2496 |
| difference_in_differences | -0.03536 [-0.04629, -0.01737] | +0.03753 [+0.03656, +0.03823] | +0.13609 | -0.0208 |

## replica_office3

### Arm outcomes

| Arm | Held-out PSNR | Held-out Depth L1 (m) | Held-out SSIM | Stale update share | Stale coverage | Updates / arm |
|---|---:|---:|---:|---:|---:|---:|
| early_global | 11.6481 [11.6408, 11.6523] | 1.40700 [1.40644, 1.40733] | 0.30722 | 0.015 | 0.073 | 565000 |
| late_global | 12.1827 [12.1802, 12.1852] | 1.38692 [1.38688, 1.38699] | 0.20393 | 0.011 | 0.033 | 565000 |
| early_stale_rescue | 12.8541 [12.8521, 12.8554] | 1.40000 [1.39992, 1.40006] | 0.35785 | 0.277 | 0.818 | 565000 |
| late_stale_rescue | 12.1021 [12.0974, 12.1112] | 1.40425 [1.40320, 1.40616] | 0.25437 | 0.249 | 0.694 | 565000 |

### Paired contrasts

Late−early compares timing within allocation; rescue−global compares allocation within timing. DiD = (late−early under stale rescue) − (late−early under global).

| Contrast | Held-out ΔPSNR (dB) | Held-out ΔDepth L1 (m) | Held-out ΔSSIM | Δ stale update share |
|---|---:|---:|---:|---:|
| timing_global | +0.53468 [+0.52911, +0.54440] | -0.02009 [-0.02036, -0.01956] | -0.10330 | -0.0045 |
| timing_stale_rescue | -0.75198 [-0.75761, -0.74095] | +0.00425 [+0.00328, +0.00609] | -0.10348 | -0.0282 |
| rescue_early | +1.20605 [+1.20102, +1.21466] | -0.00700 [-0.00740, -0.00642] | +0.05062 | +0.2616 |
| rescue_late | -0.08061 [-0.08736, -0.06904] | +0.01734 [+0.01622, +0.01928] | +0.05044 | +0.2379 |
| difference_in_differences | -1.28666 [-1.30202, -1.27006] | +0.02434 [+0.02295, +0.02646] | -0.00018 | -0.0237 |

## Interpretation and limits

The rescue first stage worked: stale-mature update share increased from about 1–2% under global allocation to about 25–29% under rescue, with exact total compute. The quality response was scene-dependent. In room2 stale rescue lifted held-out PSNR for both timing arms, but depth worsened and the timing-by-rescue interaction was small. In office3, rescue improved the early arm's held-out PSNR but slightly reduced the late arm's; the interaction was large and negative, while depth and SSIM moved differently. This is evidence that update allocation can affect outcomes under matched update counts, but it does not establish a universal starvation mechanism or controller.

This is one seed per scene and three technical repeats per frozen prefix. The repeats measure replay stability and are not independent scene or seed samples. Candidate dose is 5k. The intervention changes which primitives receive updates and therefore tests the effect of stale-mature allocation under this exact replay; it does not prove that natural map growth caused all observed quality changes. PSNR, depth, SSIM, trajectory averages, and latency are reported separately because they can disagree.

Artifacts: `validation.json`, per-run factorial records, the frozen checkpoints/candidate batches in the shared `frozen_artifacts/` directory, and source hashes under this result root.
