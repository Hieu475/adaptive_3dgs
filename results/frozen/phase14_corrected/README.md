# Phase 14 — Corrected-substrate authoritative benchmark (FROZEN)

Provenance bundle (all SHA-256 hashed in `manifest.json`):
- `manifest.json` — git SHA, hardware, dataset, pose/hygiene protocol, artifact hashes
- `phase14_results.json` — full record (metadata + policy_stats + paired_comparisons + raw_runs)
- `raw_runs.json` — 48 raw runs (6 policies x 8 seeds), extracted convenience copy
- `statistics.json` — test design + Holm paired comparisons + headroom eta
- `config_snapshot.yaml` — exact per-policy pipeline dicts (seed varies per run)
- `adaptive_office0_150f_8seed.json` — Replica adaptive validation (8 seeds)

**Scene:** `tum_fr2_xyz` | **Frames:** 150 | **Seeds:** 8 [42-49] | **Budget:** 15.0 ms
**Substrate:** corrected W2C pose (`torch.inverse`, was C2W bug) + depth hygiene
(inpaint + Sobel edge removal + 0.4–4 m) + strict densification gating
(`transmission>=0.20`, throttle when `error_ratio<0.15`).
**Supersedes:** `results/final_confirmation/` (pose-bugged 19.24 dB, see `DEPRECATED.md`).

## Policy summary (mean ± std across 8 seeds)

| Policy | Mean PSNR | Final PSNR | Mean SSIM | FPS | N_final |
|---|---|---|---|---|---|
| `no_op` | 18.86 ± 0.01 | 14.88 | 0.6882 | 106.3 | 23,523 |
| `error_only` | 24.07 ± 0.14 | 20.01 | 0.8850 | 8.9 | 74,640 |
| `error_influence` | 25.04 ± 0.06 | 21.89 | 0.9084 | 8.8 | 106,929 |
| `rtg_slam_reimpl` | 23.05 ± 0.26 | 20.04 | 0.8663 | 8.0 | 67,231 |
| **ours (pure throttling)** | **27.67 ± 0.04** | **28.89** | **0.9232** | **10.7** | **23,398** |
| `full` (ceiling) | 27.85 ± 0.01 | 29.41 | 0.9243 | 9.5 | 87,967 |

## Paired tests vs OURS (Wilcoxon n=8, Holm family-wise)

| Comparison | Δ Mean PSNR | Holm p | d_z |
|---|---|---|---|
| vs `no_op` | +8.80 | 0.0391 ✅ | 168.8 |
| vs `error_only` | +3.59 | 0.0391 ✅ | 21.2 |
| vs `error_influence` | +2.63 | 0.0391 ✅ | 29.8 |
| vs `rtg_slam_reimpl` | +4.62 | 0.0391 ✅ | 15.4 |
| vs `full` | -0.18 | 0.0391 ✅ | -3.3 |

- **Headroom recovery:** (27.67−18.86)/(27.85−18.86) = **98.0%**, gap to ceiling only **0.18 dB**
  with a **3.8× smaller map** (23K vs 88K). Selective policies share the modeled 15 ms budget;
  FULL is the unconstrained ceiling, not a matched competitor.
- Sources: `../rtg_slam_150f_8seeds/final_confirmation_results.json` (4 policies) +
  `../phase14_corrected_partial/final_confirmation_results.json` (no_op, full).
  Merged artifacts: `phase14_results.json` + `raw_runs.json` + `statistics.json` in this directory.
- Status: **FROZEN** — provenance bundle complete (`manifest.json` with SHA-256 of every artifact,
  `config_snapshot.yaml`, hardware, pose/hygiene protocol). Rebuild via `scripts/build_phase14_provenance.py`.
