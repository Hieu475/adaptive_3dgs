#!/usr/bin/env python3
"""Analyze randomized telemetry-on/off repeated branches on one checkpoint."""
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/phase14_generalization_postfix/gpu_telemetry_observer_effect_seed263"
RECORD = OUT / "replica_office1/seed263_frame90.json"
REPEATS = 5


def describe(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "sd": sd, "min": min(xs), "max": max(xs),
            "cv_fraction": sd / m if m else None}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads(RECORD.read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())
    assert record["all_count_and_snapshot_gates_passed"]
    assert record["base_snapshot_hash"] == prereg["source_snapshot_hash"]
    assert len(record["arms"]) == 2 * 2 * REPEATS
    grouped = defaultdict(dict)
    for arm in record["arms"]:
        q = round(100 * float(arm["action"]["mature_fraction"]))
        mode = "on" if arm["gpu_telemetry_enabled"] else "off"
        repeat = int(arm["action"]["repeat_index"])
        key = (q, repeat)
        if mode in grouped[key]:
            raise ValueError(f"duplicate quota/repeat/mode {key} {mode}")
        if mode == "on" and (arm["gpu_telemetry"] is None or
                              arm["gpu_telemetry"]["samples_inside_stage"] < 1):
            raise ValueError(f"telemetry-on branch has no samples: {arm['action']['name']}")
        if mode == "off" and arm["gpu_telemetry"] is not None:
            raise ValueError(f"telemetry-off branch unexpectedly contains telemetry: {arm['action']['name']}")
        grouped[key][mode] = arm
    assert all(set(x) == {"on", "off"} for x in grouped.values())

    result = {"status": "exploratory observer-effect pilot; one saved checkpoint",
              "scene": record["scene"], "seed": record["seed"],
              "snapshot_frame": record["snapshot_frame"], "snapshot_hash": record["base_snapshot_hash"],
              "n_pairs_per_quota": REPEATS, "branch_count": len(record["arms"]),
              "execution_order": record["action_execution_order"],
              "warmup": {"sync_wall_ms": record["warmup_record"]["opt_stage_sync_ms"],
                         "cuda_event_ms": record["warmup_record"]["opt_stage_cuda_event_ms"],
                         "gpu_telemetry": record["warmup_record"].get("gpu_telemetry")},
              "per_quota": {},
              "source_sha256": {"preregistration": sha256(OUT / "preregistration.json"),
                                "manifest": sha256(OUT / "manifest.json"),
                                "record": sha256(RECORD)}}
    for q in (25, 75):
        pairs = [grouped[(q, i)] for i in range(REPEATS)]
        quota = {"telemetry_on": {}, "telemetry_off": {}, "paired_on_minus_off": {}}
        for mode in ("on", "off"):
            arms = [p[mode] for p in pairs]
            for field in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms", "frame_time_ms"):
                quota["telemetry_" + mode][field] = describe([float(a[field]) for a in arms])
        for field in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms", "frame_time_ms"):
            deltas = [float(p["on"][field]) - float(p["off"][field]) for p in pairs]
            quota["paired_on_minus_off"][field] = {
                **describe(deltas),
                "mean_percent_of_off_mean": 100 * statistics.mean(deltas) /
                                             statistics.mean([float(p["off"][field]) for p in pairs]),
                "per_repeat_deltas": deltas,
                "positive_pairs": sum(x > 0 for x in deltas),
                "negative_pairs": sum(x < 0 for x in deltas),
            }
        on_telemetry = [p["on"]["gpu_telemetry"] for p in pairs]
        quota["telemetry_samples_per_branch"] = [x["samples_inside_stage"] for x in on_telemetry]
        fields = ("graphics_clock_mhz", "memory_clock_mhz", "temperature_c",
                  "power_mw", "gpu_utilization_percent", "memory_used_bytes")
        quota["gpu_state_on_means"] = {}
        for field in fields:
            vals = [t["window_stats"][field]["mean"] for t in on_telemetry
                    if t["window_stats"][field] is not None and
                    "mean" in t["window_stats"][field]]
            quota["gpu_state_on_means"][field] = statistics.mean(vals) if vals else None
        result["per_quota"][str(q)] = quota

    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# NVML telemetry observer-effect pilot: Replica office1 seed263", "",
             "Five same-snapshot paired repeats per quota were run with telemetry-on and telemetry-off branches randomly interleaved in one process. Snapshot, candidate pool, spawn count, quota, and update counts were fixed. The only intended difference was whether the background NVML sampler was sampling during the branch. Repeats are not independent seed samples.", "",
             "| Quota | Timing metric | Telemetry off mean ± SD | Telemetry on mean ± SD | Paired on−off mean ± SD | Relative mean shift | Pair directions |", "|---:|---|---:|---:|---:|---:|---:|"]
    for q in (25, 75):
        x = result["per_quota"][str(q)]
        for field, label in (("opt_stage_cuda_event_ms", "CUDA event ms"),
                             ("opt_stage_sync_ms", "Sync wall ms")):
            off, on, delta = (x["telemetry_off"][field], x["telemetry_on"][field],
                              x["paired_on_minus_off"][field])
            lines.append(f"| {q}% | {label} | {off['mean']:.3f} ± {off['sd']:.3f} | {on['mean']:.3f} ± {on['sd']:.3f} | {delta['mean']:+.3f} ± {delta['sd']:.3f} | {delta['mean_percent_of_off_mean']:+.2f}% | {delta['positive_pairs']}+ / {delta['negative_pairs']}− |")
    lines += ["", "## Telemetry capture", ""]
    for q in (25, 75):
        x = result["per_quota"][str(q)]
        lines.append(f"- Quota{q}: on-arm samples inside stage {x['telemetry_samples_per_branch']}; mean state: graphics {x['gpu_state_on_means']['graphics_clock_mhz']:.0f} MHz, memory {x['gpu_state_on_means']['memory_clock_mhz']:.0f} MHz, temperature {x['gpu_state_on_means']['temperature_c']:.1f}°C, power {x['gpu_state_on_means']['power_mw']/1000:.2f} W.")
    lines += ["", f"The discarded warm-up was {result['warmup']['cuda_event_ms']:.2f} ms (CUDA event), with telemetry captured. Branch logs show {len(record['action_execution_order'])} randomized actions; telemetry-off branches contain no in-stage samples and telemetry-on branches contain 3–4.", "", "## Interpretation and limits", "", "The observed mean telemetry-on timing shifts are small relative to stage duration and do not show a consistent direction across quotas. With only five repeated pairs from one checkpoint, this is a descriptive observer-effect check, not a statistical equivalence test. It suggests the sampler did not produce a large, consistent timing shift in this invocation; it does not rule out occasional NVML/host scheduling outliers or cross-run GPU-state variation.", "", "There is no matched-time claim here. The 25%-vs-75% arm timing gap remains a separate property of the action workloads and was not used to alter update counts. The old timing runs still lack telemetry, so this does not explain their run-to-run shifts.", "", "## Provenance", "", f"- Preregistration SHA-256: `{result['source_sha256']['preregistration']}`", f"- Manifest SHA-256: `{result['source_sha256']['manifest']}`", f"- Run record SHA-256: `{result['source_sha256']['record']}`", f"- Snapshot hash: `{result['snapshot_hash']}`", f"- Git revision: `{json.loads((OUT/'manifest.json').read_text()).get('git_sha')}` (dirty worktree)", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
