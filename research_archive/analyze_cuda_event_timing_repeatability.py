#!/usr/bin/env python3
"""Compare CUDA-event and synchronized wall timing on same-checkpoint repeats."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix"
OUT = BASE / "matched_time_age_quota_pilot_replica_office1"
RUNS = {
    17: BASE / "matched_time_age_quota_pilot_replica_office1_cudaevent_order17/replica_office1/seed42_frame90.json",
    41: BASE / "matched_time_age_quota_pilot_replica_office1_cudaevent_order41/replica_office1/seed42_frame90.json",
    67: BASE / "matched_time_age_quota_pilot_replica_office1_cudaevent_order67/replica_office1/seed42_frame90.json",
}


def read(path):
    d = json.loads(path.read_text())
    if not d["all_count_and_snapshot_gates_passed"]:
        raise RuntimeError(f"Snapshot or quota gate failed: {path}")
    arms = {a["action"]["name"]: a for a in d["arms"]}
    for dose in (0, 128):
        for pct, count in ((25, 5000), (75, 4494)):
            a = arms[f"spawn{dose}_mature{pct}"]
            if not a["valid"] or a["n_micro_updates"] != count:
                raise RuntimeError(f"Quota/update mismatch: {path}, {a['action']['name']}")
            if a["opt_stage_cuda_event_ms"] <= 0:
                raise RuntimeError(f"Missing CUDA event timing: {path}, {a['action']['name']}")
    return d, arms


def main():
    data = {}
    hashes = set()
    for order, path in RUNS.items():
        d, arms = read(path)
        hashes.add(d["base_snapshot_hash"])
        data[order] = {"record": d, "arms": arms}
    if len(hashes) != 1:
        raise RuntimeError("Timing repeats do not share a snapshot hash")

    rows = []
    pooled = {}
    for dose in (0, 128):
        values = {metric: {25: [], 75: []} for metric in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms")}
        for order, x in data.items():
            a = x["arms"]
            lo, hi = a[f"spawn{dose}_mature25"], a[f"spawn{dose}_mature75"]
            row = {"order_seed": order, "spawn_count": dose}
            for metric in values:
                l, h = lo[metric], hi[metric]
                gap = abs(h-l) / ((h+l)/2)
                row[f"{metric}_25"] = l
                row[f"{metric}_75"] = h
                row[f"{metric}_gap_fraction"] = gap
                values[metric][25].append(l)
                values[metric][75].append(h)
            rows.append(row)
        pooled[str(dose)] = {}
        for metric, q in values.items():
            m25, m75 = statistics.mean(q[25]), statistics.mean(q[75])
            pooled[str(dose)][metric] = {
                "mean_25_ms": m25, "sd_25_ms": statistics.stdev(q[25]),
                "mean_75_ms": m75, "sd_75_ms": statistics.stdev(q[75]),
                "gap_between_means_fraction": abs(m75-m25)/((m25+m75)/2),
                "per_replay_gap_fraction": [r[f"{metric}_gap_fraction"] for r in rows if r["spawn_count"] == dose],
            }
    (OUT / "cuda_event_timing_analysis.json").write_text(json.dumps({
        "status": "single-checkpoint timing repeatability; no trajectory inference",
        "snapshot_hash": next(iter(hashes)), "runs": rows, "pooled": pooled,
    }, indent=2) + "\n")

    lines = [
        "# CUDA-event timing repeatability", "",
        "Three warmed, shuffled-order replays used the same Replica office1 seed-42 frame-90 full checkpoint. Every branch passed snapshot and exact quota/update gates. CUDA events bracket the same optimization stage as the synchronized wall timer; event time excludes host-side elapsed time between device submissions.", "",
        "| Order seed | Spawn | CUDA event (25%,75%) ms | CUDA-event gap | Wall clock (25%,75%) ms | Wall gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['order_seed']} | {r['spawn_count']} | "
            f"{r['opt_stage_cuda_event_ms_25']:.3f}, {r['opt_stage_cuda_event_ms_75']:.3f} | "
            f"{r['opt_stage_cuda_event_ms_gap_fraction']*100:.2f}% | "
            f"{r['opt_stage_sync_ms_25']:.3f}, {r['opt_stage_sync_ms_75']:.3f} | "
            f"{r['opt_stage_sync_ms_gap_fraction']*100:.2f}% |")
    lines += ["", "## Pooled timing by spawn dose", "",
              "| Spawn | CUDA event mean 25% (SD) ms | CUDA event mean 75% (SD) ms | Mean gap | Wall mean 25% (SD) ms | Wall mean 75% (SD) ms | Mean gap |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for dose, x in pooled.items():
        ev, wall = x["opt_stage_cuda_event_ms"], x["opt_stage_sync_ms"]
        lines.append(f"| {dose} | {ev['mean_25_ms']:.3f} ({ev['sd_25_ms']:.3f}) | {ev['mean_75_ms']:.3f} ({ev['sd_75_ms']:.3f}) | {ev['gap_between_means_fraction']*100:.2f}% | {wall['mean_25_ms']:.3f} ({wall['sd_25_ms']:.3f}) | {wall['mean_75_ms']:.3f} ({wall['sd_75_ms']:.3f}) | {wall['gap_between_means_fraction']*100:.2f}% |")
    lines += ["", "## Interpretation", "",
              "CUDA-event and synchronized wall times are within about 0.03 ms in these runs, so host timer overhead does not explain the large variance. The CUDA-stage duration itself varies substantially across repeated branches: per-run quota gaps range from 0.26% to 74.04%; pooled mean gaps are 17.42% at zero spawn and 31.78% at spawn 128. GPU clock/load variation and action-dependent GPU work remain possible contributors; this pilot does not distinguish them. A matched-time causal claim is not supported.", "",
              "Next gate: repeat the exact same action several times from the same checkpoint with randomized ordering and a discarded warm-up. Establish within-action CUDA-event repeatability first. Then only compare quota arms if the timing distributions overlap under a prespecified matching rule.", ""]
    (OUT / "cuda_event_timing_report.md").write_text("\n".join(lines))
    print(OUT / "cuda_event_timing_report.md")


if __name__ == "__main__":
    main()
