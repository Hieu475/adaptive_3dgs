# E1 selector transfer: Replica room1

**Status:** two-seed exploratory transfer; not a generalization claim.
**Runs:** `results/mechanism_v2/e1_room1_replication_v1/replica_room1/seed{57,58}_room1_e1_v1.json`.
**Validation:** 2/2 pairs pass `results/mechanism_v2/e1_room1_replication_v1/validation.json`.

Each seed crosses early insertion at frame 30 and late insertion at frame 90 with the six selector modes in the E1 pilot report. Every arm uses the same frozen 5,000-candidate batch within seed, and all arms account for exactly 425,000 logical micro-updates. Late arms share the same frame-89 global snapshot and 285,000-update common prefix. The experiment disables automatic spawning and evaluates a fixed candidate intervention on known poses.

## Selector effect versus global

Numbers are held-out deltas in dB PSNR; brackets show seed 57 / seed 58. Positive is better than global at the same timing.

| Selector | Early Δ PSNR | Early Δ depth L1 (m) | Late Δ PSNR | Late Δ depth L1 (m) |
|---|---:|---:|---:|---:|
| Age-only quota | −0.060 / −0.066 | +0.0177 / +0.0215 | +0.036 / +0.030 | −0.0036 / −0.0022 |
| Random mature | −0.108 / −0.109 | +0.0287 / +0.0306 | +0.002 / +0.002 | +0.0001 / −0.0003 |
| Stale + importance | −0.032 / −0.036 | +0.0080 / +0.0122 | +0.004 / +0.002 | −0.0009 / −0.0003 |
| Stale + residual, no visibility gate | −0.096 / −0.106 | +0.0310 / +0.0309 | −0.011 / −0.014 | +0.0012 / +0.0017 |
| Stale + residual + visibility | −0.035 / −0.055 | +0.0102 / +0.0165 | −0.004 / −0.005 | +0.0003 / +0.0006 |

The signs replicate within this scene: every tested non-global selector regresses early; late age-only improves slightly, stale+importance is near zero, and residual-ranked stale selectors are neutral to slightly worse. This differs from room2's single-seed early deltas, so the room1 result reinforces scene dependence rather than selecting a winner.

The realized mature-update share was 0.941 early and 0.821 late for the requested 0.50 age quota, because the candidate cohort ages into the mature group and quota transfers when the young stratum cannot absorb its requested share. Stale-update shares also vary by policy and timing; requested quota is not realized dose.

## Limits

Global held-out PSNR is only about 5.5–6.0 dB in this fixed-batch setting. Mean alpha coverage is 0.277 early and 0.186 late, so most valid depth pixels have no support; the optimization-stage CUDA time is about 55.5–56.7 ms/frame. Automatic map growth is disabled, so these are local selector comparisons under a deliberately constrained candidate pool, not normal mapping performance. Two seeds in one scene do not support population-level scene inference. Keep the result as a transfer pilot and add scenes/seeds before selecting a mechanism or controller.
