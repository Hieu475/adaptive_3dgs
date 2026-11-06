#!/usr/bin/env python3
"""Analyze same-snapshot four-level mature-quota dose-response replay."""
import hashlib
import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = tuple(range(42, 50))
QUOTAS = (25, 75, 96, 100)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def quantile(xs, q):
    xs = sorted(xs); p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def bootstrap_ci(xs, seed, n=50000):
    rng = random.Random(seed)
    draws = [sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n)]
    return [quantile(draws, .025), quantile(draws, .975)]


def sign_test_p(xs):
    xs = [x for x in xs if x != 0]
    if not xs: return 1.0
    n = len(xs); k = min(sum(x > 0 for x in xs), sum(x < 0 for x in xs))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [None] * len(pvalues); running = 0.0; m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m-rank) * pvalues[idx]))
        out[idx] = running
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=90, choices=(70, 90))
    ap.add_argument("--scene", default="replica_office1")
    ap.add_argument("--input", type=Path, default=None,
                    help="Experiment directory; defaults to the Replica frame-specific directory")
    args = ap.parse_args()
    frame = args.frame
    scene = args.scene
    if args.input is None:
        base = ROOT / f"results/phase14_generalization_postfix/quota_response_frame{frame}"
    else:
        base = args.input if args.input.is_absolute() else ROOT / args.input
    prereg_path = base / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    recs, hashes = {}, {}
    for seed in SEEDS:
        path = base / scene / f"seed{seed}_frame{frame}.json"
        r = json.loads(path.read_text())
        if (r["scene"], r["seed"], r["snapshot_frame"]) != (scene, seed, frame):
            raise ValueError(f"Identity mismatch: {path}")
        if not r["all_count_and_snapshot_gates_passed"]:
            raise ValueError(f"Validity gate failed: {path}")
        arms = {x["action"]["name"]: x for x in r["arms"]}
        for q in QUOTAS:
            for spawn in (0, 128):
                name = f"spawn{spawn}_mature{q}"
                a = arms[name]
                mature = round(5000*q/100); young = 5000-mature
                if (a["realized_mature_micro_updates"], a["realized_young_micro_updates"], a["n_spawned"]) != (mature,young,spawn):
                    raise ValueError(f"Quota/spawn mismatch: {path} {name}")
        recs[seed] = (r, arms)
        hashes[str(path.relative_to(ROOT))] = sha256(path)

    gain_by_q = {q: [] for q in QUOTAS}
    depth_by_q = {q: [] for q in QUOTAS}
    time_by_q = {q: [] for q in QUOTAS}
    contrasts = {"75_minus_25": [], "100_minus_96": []}
    contrast_depth = {k: [] for k in contrasts}
    contrast_time = {k: [] for k in contrasts}
    per_seed = []
    for seed in SEEDS:
        r, arms = recs[seed]
        pre = r["pre_action_snapshot_eval"]["psnr"]
        row = {"seed": seed, "pre_action_psnr": pre, "snapshot_hash": r["base_snapshot_hash"], "zero_spawn": {}}
        for q in QUOTAS:
            a = arms[f"spawn0_mature{q}"]
            gain = a["final_map_eval_psnr"] - pre
            gain_by_q[q].append(gain)
            depth_by_q[q].append(a["final_map_eval_depth_l1"])
            time_by_q[q].append(a["opt_stage_sync_ms"])
            row["zero_spawn"][str(q)] = {"psnr_gain_db": gain,
                                         "final_depth_l1_m": a["final_map_eval_depth_l1"],
                                         "sync_opt_ms": a["opt_stage_sync_ms"],
                                         "mature_updates": a["realized_mature_micro_updates"],
                                         "young_updates": a["realized_young_micro_updates"]}
        for key, (hi, lo) in {"75_minus_25":(75,25), "100_minus_96":(100,96)}.items():
            ah, al = arms[f"spawn0_mature{hi}"], arms[f"spawn0_mature{lo}"]
            contrasts[key].append(ah["final_map_eval_psnr"] - al["final_map_eval_psnr"])
            contrast_depth[key].append(ah["final_map_eval_depth_l1"] - al["final_map_eval_depth_l1"])
            contrast_time[key].append(ah["opt_stage_sync_ms"] - al["opt_stage_sync_ms"])
            row.setdefault("primary_contrasts_db", {})[key] = contrasts[key][-1]
        per_seed.append(row)

    raw_p = [sign_test_p(contrasts[k]) for k in ("75_minus_25", "100_minus_96")]
    adj_p = holm(raw_p)
    primary = {}
    for idx, key in enumerate(("75_minus_25", "100_minus_96")):
        xs = contrasts[key]
        primary[key] = {
            "mean_effect_db": sum(xs)/len(xs),
            "seed_bootstrap_95_ci_db": bootstrap_ci(xs, 90090+idx),
            "positive_seeds": sum(x>0 for x in xs), "n_seeds": len(xs),
            "exact_two_sided_sign_test_raw_p": raw_p[idx],
            "holm_adjusted_p": adj_p[idx],
            "per_seed_effect_db": {str(s): contrasts[key][i] for i,s in enumerate(SEEDS)},
            "mean_depth_l1_change_m": sum(contrast_depth[key])/len(SEEDS),
            "mean_sync_opt_time_change_ms": sum(contrast_time[key])/len(SEEDS),
        }

    result = {
        "status": "all preregistered snapshot and action gates passed",
        "scene": scene,
        "snapshot_frame": frame,
        "preregistration": prereg,
        "valid_seed_snapshots": 8,
        "quota_curve_zero_spawn": {
            str(q): {"mean_psnr_gain_db": sum(gain_by_q[q])/8,
                    "seed_bootstrap_95_ci_db": bootstrap_ci(gain_by_q[q], 90100+q),
                    "mean_final_depth_l1_m": sum(depth_by_q[q])/8,
                    "mean_sync_optimization_time_ms": sum(time_by_q[q])/8}
            for q in QUOTAS
        },
        "primary_contrasts": primary,
        "spawn128_descriptive_gains_db": {
            str(q): sum(arms[f"spawn128_mature{q}"]["final_map_eval_psnr"] - r["pre_action_snapshot_eval"]["psnr"]
                        for r,arms in recs.values()) / 8
            for q in QUOTAS
        },
        "seed_records": per_seed,
        "provenance": {"manifest_sha256": sha256(base/"manifest.json"),
            "preregistration_sha256": sha256(prereg_path),
            "runner_sha256": sha256(ROOT/"experiments/run_joint_spawn_refine_replay.py"),
            "analyzer_sha256": sha256(Path(__file__)), "record_sha256": hashes},
        "interpretation_limit": prereg["interpretation_limit"],
    }
    (base/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=[f"# {scene} frame-{frame} mature-quota dose response","",
        "All four mature-update quotas were replayed on the same prefix snapshot for each of eight seeds. Every arm used 5,000 updates; map size was matched across quotas within each spawn dose.","",
        "## Zero-spawn quota curve","","| Mature quota | Mean final-map PSNR gain (dB; seed CI) | Mean depth L1 (m) | Mean sync optimization time (ms) |","|--:|--:|--:|--:|"]
    for q in QUOTAS:
        x=result["quota_curve_zero_spawn"][str(q)]; c=x["seed_bootstrap_95_ci_db"]
        lines.append(f"| {q}% | {x['mean_psnr_gain_db']:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {x['mean_final_depth_l1_m']:.5f} | {x['mean_sync_optimization_time_ms']:.2f} |")
    lines += ["","## Preregistered contrasts","","| Contrast (higher minus lower mature quota) | Mean PSNR effect (dB; seed CI) | Positive seeds | Raw exact p | Holm p | Mean depth-L1 change (m) | Mean sync-time change (ms) |","|:--|--:|--:|--:|--:|--:|--:|"]
    for key,label in (("75_minus_25","75% − 25%"),("100_minus_96","100% − 96%")):
        x=primary[key]; c=x["seed_bootstrap_95_ci_db"]
        lines.append(f"| {label} | {x['mean_effect_db']:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}] | {x['positive_seeds']}/8 | {x['exact_two_sided_sign_test_raw_p']:.4f} | {x['holm_adjusted_p']:.4f} | {x['mean_depth_l1_change_m']:+.5f} | {x['mean_sync_opt_time_change_ms']:+.3f} |")
    lines += ["","All eight seed snapshots passed the preregistered map, render, quota, spawn, candidate-pool, and cardinality gates. The exact same per-seed snapshot and reference render were used across quota levels.","",
        "This is a one-frame local dose-response at one known regression scene and stage. It can test nonlinearity at this snapshot; it does not establish a continuously applied policy or generalization. Other stage runs are separate and are not pooled into these tests.","",
        "Per-seed outcomes and provenance hashes are in `analysis.json`.",""]
    (base/"report.md").write_text("\n".join(lines))
    print(base/"report.md")


if __name__ == "__main__": main()
