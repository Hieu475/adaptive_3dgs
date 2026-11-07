# E1 room2 selector replication: seed 58

**Status:** exploratory two-seed replication; not a generalization claim.
**Raw runs:** `results/mechanism_v2/e1_rescue_controls_pilot_v4/replica_room2/seed57_e1_controls_v4.json` and `results/mechanism_v2/e1_room2_replication_v1/replica_room2/seed58_room2_e1_seed58.json`.
**Validity:** both runs pass the matched-candidate validator; seed 58 is recorded in `results/mechanism_v2/e1_room2_replication_v1/validation.json`.

## Protocol

The replay inserts the same frozen 5,000-Gaussian batch at frame 30 (early) or frame 90 (late), disables automatic births, and compares six allocation modes. Every arm accounts for 425,000 logical micro-updates, including the 285,000-update common late prefix. Six fixed held-out frames are scored after mapping. The early and late interventions use their respective frozen snapshots; this is a timing intervention, not natural map growth.

## Held-out selector effect versus global allocation

Entries are the mean across seeds 57 and 58. Positive PSNR is better; negative depth L1 is better. Values are descriptive seed means, with only two seeds.

| Allocation | Early ΔPSNR (dB) | Early ΔDepth L1 (m) | Late ΔPSNR (dB) | Late ΔDepth L1 (m) |
|---|---:|---:|---:|---:|
| Age-only | +0.153 | −0.0258 | +0.010 | −0.0036 |
| Random mature | +0.120 | −0.0145 | −0.024 | +0.0053 |
| Stale + importance | −0.037 | +0.0064 | −0.017 | +0.0061 |
| Stale + residual | +0.136 | −0.0247 | −0.038 | +0.0132 |
| Stale + residual + visibility | +0.145 | −0.0253 | +0.035 | −0.0115 |

Across both seeds, early age-only and residual-ranked stale allocation improve the held-out RGB and depth metrics over global selection. At late insertion, age-only is close to neutral, while stale+residual regresses on both metrics. The visibility-gated residual variant is mildly positive at both timings, but its effect is small and this pilot does not establish it as a reliable winner.

The stale+importance rescue arm has a negative held-out PSNR contrast at both timings and both seeds. Its nominal 50% stale quota is not its realized dose: stale Gaussians receive about 19.4–19.5% of early updates and 2.5% of late updates. Residual rescue receives about 9.0% early and 0.5% late; visibility-gated residual rescue receives about 14.1% early and 3.2% late. Late timing leaves very few candidates classified stale, so the interventions have weak and different realized doses across timing.

## Cost and interpretation

Logical update counts match exactly. Mean optimizer CUDA-stage time is about 59.8–61.9 ms per mapped frame across arms and seeds. This is measured CUDA-stage time, not end-to-end latency equivalence. Timing differences are modest in this pilot, but exact update matching alone does not make actual GPU work identical.

The room2 seed-57 pattern is reproduced for age-only, random-mature, stale+importance, residual, and visibility-gated residual arms: their held-out directions are consistent between seeds. The magnitude and practical value still need more seeds and scenes. Most notably, stale+residual changes from a positive early effect (mean +0.136 dB) to a negative late effect (mean −0.038 dB), while stale+visibility+residual is mildly positive late. This supports timing-dependent selector behavior, not a universal benefit from rescuing stale Gaussians.

The replay disables automatic spawning and uses one fixed candidate batch. Its absolute PSNR, depth error, and coverage therefore do not represent ordinary online mapping quality. This experiment isolates selector allocation on frozen candidates; it does not yet show that natural Gaussian growth causes starvation or quality degradation.

## Next experiment

Do not add another rescue heuristic from this result. The next causal step is a controlled growth-dose intervention with shared candidate identities, fixed update budgets, and a protected mature-update arm. It must vary map growth itself and report both requested and realized update shares, held-out quality, depth, coverage, and CUDA-stage time. Preserve the exact prefix/candidate checkpoints for repeat runs so seed variation can be separated from prefix regeneration.
