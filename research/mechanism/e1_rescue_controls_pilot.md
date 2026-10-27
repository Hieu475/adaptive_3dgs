# E1 selector-control pilot

**Status:** one-seed matched pilot; not confirmatory.

**Run:** `results/mechanism_v2/e1_rescue_controls_pilot_v4/replica_room2/seed57_e1_controls_v4.json`.
**Validation:** 1/1 pair passes `results/mechanism_v2/e1_rescue_controls_pilot_v4/validation.json`.

## Protocol correction

The first control prototype did not match compute: age-only arms selected only visible age strata and could stop at 378k–482k updates. A capacity fallback was added to prefer visible IDs and fill a quota from invisible IDs in the same age stratum. The next prototype enabled each allocation policy before candidate insertion, so the policy changed the common pre-treatment trajectory. That was corrected by keeping every arm on the same global selector until the intervention starts.

An additional audit found that the original replay loop restored a frame-29 checkpoint but then replayed from frame 1. Version 4 fixes this chronology. Early arms fork from the frame-29 checkpoint and begin at frame 30. A single common global rollout processes frames 30–89 once; late arms all fork from its saved frame-89 checkpoint and begin at frame 90. The saved late checkpoint has SHA-256 `cc637678216b29044e59837bdc801666d7e814b3db70ade51ebc5b79f520589c`.

Both timing groups use the same candidate batch and 425,000 total logical micro-updates per arm. Late arms include the same 285,000-update common prefix; their selector-specific branch has 140,000 updates. Early arms have 425,000 selector-specific updates. This matches total logical work while making the selector active for different parts of the trajectory.

The experiment disables automatic births and inserts one frozen batch of 5,000 candidates. It measures a fixed-candidate selector intervention, not natural online map growth. Six held-out views are scored after the trajectory using metric v2. All source hashes, candidate/prefix hashes, exact-birth/update, common-snapshot, metric-validity, and measured CUDA timing checks pass.

## Held-out outcomes

Each cell is one seed. The delta columns compare a selector with global at the same insertion time.

| Allocation | Early PSNR (dB) | Early Δ PSNR | Early depth L1 (m) | Late PSNR (dB) | Late Δ PSNR | Late depth L1 (m) | Late coverage | Realized stale-update share (early / late) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global importance | 7.309 | — | 1.8810 | 7.982 | — | 1.6678 | 0.310 | 0.013 / 0.004 |
| Age-only quota | 7.460 | +0.151 | 1.8546 | 7.988 | +0.006 | 1.6649 | 0.311 | 0.027 / 0.005 |
| Random mature | 7.429 | +0.120 | 1.8664 | 7.954 | −0.028 | 1.6733 | 0.306 | 0.041 / 0.002 |
| Stale + importance | 7.278 | −0.031 | 1.8871 | 7.966 | −0.016 | 1.6737 | 0.307 | 0.194 / 0.025 |
| Stale + residual, no visibility gate | 7.438 | +0.129 | 1.8570 | 7.936 | −0.046 | 1.6830 | 0.301 | 0.090 / 0.005 |
| Stale + residual + visibility | 7.451 | +0.142 | 1.8556 | 8.011 | +0.029 | 1.6587 | 0.315 | 0.142 / 0.032 |

Mean CUDA optimization-stage time across each arm's accounted logical-work interval ranged from 59.77 to 62.52 ms. Equal update counts therefore still do not imply exactly equal GPU time.

## Interpretation and next step

The selector ranking is not stable across insertion timing in this one seed. Age-only is numerically best in early held-out PSNR by 0.151 dB, while visibility-gated stale+residual is numerically best late by 0.029 dB over global. Stale+importance does not help either timing. These small, timing-dependent differences do not identify a general winner.

Requested quotas also differ from realized dose. Age-only's mature-update share is 0.941 early and 0.821 late despite a requested 0.50, because the inserted cohort itself becomes mature and quota transfers when the young stratum has insufficient capacity. Stale rescue realizes 0.194 early but only 0.025 late; stale residual realizes 0.090 and 0.005. The late replay therefore has a weak stale-treatment first stage. Report requested and realized allocation separately.

Corrected visualization at `results/mechanism_v2/chronological_metric_visuals/figures/replica_room2/seed57/` shows substantial unsupported regions at held-out views. This is expected in part because the experiment disables automatic spawning and adds only one fixed batch. Do not interpret these absolute quality values as normal online-mapping performance or infer a pose bug from coverage alone.

Next, repeat the same frozen-snapshot control on at least one additional scene and multiple seeds, retaining the same arm definitions and exact work budget. First improve the stale first stage in the late condition through a prespecified quota/dose design, then report the effect conditional on realized stale updates. Do not tune the selector from this one-seed result.
