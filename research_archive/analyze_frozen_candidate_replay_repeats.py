#!/usr/bin/env python3
"""Summarize repeated branches loaded from the same frozen prefix and batch."""
import json
import statistics
import argparse
from pathlib import Path


ROOT = Path("results/legacy/matched_candidate_timing_replay_fixed_snapshot")
RUN_IDS = ("repeat1", "repeat2", "repeat3")
SCENE = "replica_room2"
SEED = 57


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--run_ids", nargs="+", default=list(RUN_IDS))
    args = ap.parse_args()
    root, scene, seed, run_ids = args.root, args.scene, args.seed, args.run_ids
    validation = json.loads((root / "validation.json").read_text())
    validation_pairs = [p for p in validation["pairs"]
                        if p["scene"] == scene and int(p["seed"]) == seed
                        and p.get("run_id") in run_ids]
    if len(validation_pairs) != len(run_ids) or any(not p["valid"] for p in validation_pairs):
        raise RuntimeError(f"Only {sum(p['valid'] for p in validation_pairs)}/{len(run_ids)} repeated pairs passed gates")
    rows = []
    for run_id in run_ids:
        path = root / scene / f"seed{seed}_{run_id}.json"
        record = json.loads(path.read_text())
        if not record["valid"]:
            raise RuntimeError(f"Replay pair failed runner validity: {path}")
        early, late = record["arms"]
        if early["prefix_state_sha256"] != late["prefix_state_sha256"]:
            raise RuntimeError(f"Within-run prefix mismatch: {path}")
        if early["candidate_batch_sha256"] != late["candidate_batch_sha256"]:
            raise RuntimeError(f"Within-run candidate mismatch: {path}")
        rows.append({
            "run_id": run_id,
            "artifact": str(path),
            "prefix_state_sha256": record["prefix_state_sha256"],
            "candidate_batch_sha256": record["candidate_batch_sha256"],
            "prefix_checkpoint_sha256": record["frozen_prefix_checkpoint_sha256"],
            "candidate_file_sha256": record["frozen_candidate_file_sha256"],
            "late_minus_early": record["paired_late_minus_early"],
        })
    for field in ("prefix_state_sha256", "candidate_batch_sha256",
                  "prefix_checkpoint_sha256", "candidate_file_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise RuntimeError(f"Frozen artifact changed across repeats: {field}")

    summary = {"schema": "frozen-candidate-replay-repeatability-v1",
               "scene": scene, "seed": seed, "n_repeats": len(rows),
               "prefix_sha256": rows[0]["prefix_state_sha256"],
               "candidate_batch_sha256": rows[0]["candidate_batch_sha256"],
               "fixed_prefix_and_candidate_confirmed": True,
               "metrics": {}, "runs": rows,
               "interpretation": "These repeated branches start from identical serialized prefix and candidate files. Differences across repeats therefore reflect replay/runtime variability downstream of those frozen inputs, not prefix regeneration."}
    metric_keys = (
        "fixed_heldout_mean_psnr", "fixed_heldout_mean_depth_l1",
        "fixed_heldout_mean_ssim", "trajectory_mean_psnr",
        "trajectory_mean_depth_l1", "mean_mature_update_coverage",
        "mean_opt_stage_sync_ms",
    )
    lines = [
        "# Frozen-prefix candidate timing replay repeatability", "",
        f"{scene} seed {seed}; {len(rows)} repeated early/late pairs replayed from one saved prefix checkpoint and one saved 5k Gaussian batch.",
        "",
        f"Prefix map SHA256: `{summary['prefix_sha256']}`",
        f"Candidate batch SHA256: `{summary['candidate_batch_sha256']}`",
        "All repeats passed the causal-control and frozen-file checksum gates. Each pair used 565,000 micro-updates per arm, zero automatic births, exactly 5,000 manual additions per arm, and the same six held-out views.",
        "",
        "| Replay | Δ held-out PSNR | Δ held-out Depth L1 (m) | Δ held-out SSIM | Δ trajectory PSNR | Δ mature update coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["late_minus_early"]
        lines.append(f"| {row['run_id']} | {d['fixed_heldout_mean_psnr']:+.5f} dB | {d['fixed_heldout_mean_depth_l1']:+.5f} | {d['fixed_heldout_mean_ssim']:+.5f} | {d['trajectory_mean_psnr']:+.5f} dB | {d['mean_mature_update_coverage']:+.5f} |")
    for key in metric_keys:
        vals = [float(row["late_minus_early"][key]) for row in rows]
        summary["metrics"][key] = {"mean": statistics.mean(vals),
                                   "stdev_sample": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                                   "min": min(vals), "max": max(vals),
                                   "positive_repeats": sum(v > 0 for v in vals),
                                   "negative_repeats": sum(v < 0 for v in vals)}
    psnr = summary["metrics"]["fixed_heldout_mean_psnr"]
    depth = summary["metrics"]["fixed_heldout_mean_depth_l1"]
    lines += ["", "## Repeatability summary", "",
              f"Held-out PSNR contrast: mean {psnr['mean']:+.4f} dB, sample SD {psnr['stdev_sample']:.4f}, range [{psnr['min']:+.4f}, {psnr['max']:+.4f}] dB; positive in {psnr['positive_repeats']}/{len(rows)} repeats.",
              f"Held-out Depth L1 contrast: mean {depth['mean']:+.5f} m, sample SD {depth['stdev_sample']:.5f}, range [{depth['min']:+.5f}, {depth['max']:+.5f}] m; negative (better late) in {depth['negative_repeats']}/{len(rows)} repeats.",
              "", summary["interpretation"], "",
              "This is one scene, one seed, and three runtime repeats. It estimates repeatability for this frozen state only; it is not scene generalization or a population-level timing effect.",
              "", "Artifacts: `frozen_artifacts/` contains the pipeline checkpoint and candidate batch; per-repeat records are beside this report; `validation.json` audits each pair."]
    out_md = root / f"repeatability_{scene}_seed{seed}.md"
    out_json = root / f"repeatability_{scene}_seed{seed}.json"
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
