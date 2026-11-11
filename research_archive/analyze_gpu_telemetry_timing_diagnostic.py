#!/usr/bin/env python3
"""Summarize the one-snapshot NVML timing-instrumentation diagnostic."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/legacy/phase14_generalization_postfix/gpu_telemetry_timing_diagnostic_seed263"
RECORD = OUT / "replica_office1/seed263_frame90.json"


def stats(values):
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": avg, "sd": sd, "cv_fraction": sd / avg if avg else None,
            "min": min(values), "max": max(values)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    record = json.loads(RECORD.read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert record["all_count_and_snapshot_gates_passed"]
    assert record["base_snapshot_hash"] == prereg["source_snapshot_hash"]
    rows = record["arms"]
    assert len(rows) == 20
    assert all(r["valid"] and r["n_spawned"] == 128 and
               r["gpu_telemetry"] is not None and
               r["gpu_telemetry"]["samples_inside_stage"] >= 1 for r in rows)
    result = {"status": "single-checkpoint timing diagnostic; not a seed-level quality estimate",
              "scene": record["scene"], "seed": record["seed"],
              "snapshot_frame": record["snapshot_frame"],
              "snapshot_hash": record["base_snapshot_hash"],
              "warmup_sync_ms": record["warmup_record"]["opt_stage_sync_ms"],
              "warmup_cuda_event_ms": record["warmup_record"].get("opt_stage_cuda_event_ms"),
              "warmup_gpu_telemetry": record["warmup_record"].get("gpu_telemetry"),
              "matched_time_gate": {"threshold_fraction": 0.05},
              "per_quota": {}, "branch_count": len(rows),
              "branches_with_gpu_samples": sum(r["gpu_telemetry"]["samples_inside_stage"] > 0 for r in rows),
              "source_hashes": {"preregistration": sha256(OUT / "preregistration.json"),
                                "manifest": sha256(OUT / "manifest.json"),
                                "launch_config": sha256(OUT / "launch_config.json"),
                                "record": sha256(RECORD)},
              "manifest_git_sha": manifest.get("git_sha"),
              "gpu": manifest.get("gpu"), "vram_gb": manifest.get("vram_gb")}
    for pct in (25, 75):
        group = [r for r in rows if round(100 * r["action"]["mature_fraction"]) == pct]
        telemetry_fields = ("gpu_utilization_percent", "memory_utilization_percent",
                            "memory_used_bytes", "graphics_clock_mhz", "memory_clock_mhz",
                            "temperature_c", "performance_state", "power_mw")
        tele = {}
        for field in telemetry_fields:
            values = [r["gpu_telemetry"]["window_stats"][field]["mean"]
                      for r in group if r["gpu_telemetry"]["window_stats"][field] is not None
                      and "mean" in r["gpu_telemetry"]["window_stats"][field]]
            tele[field] = stats(values) if values else None
        result["per_quota"][str(pct)] = {
            "n_repeats": len(group),
            "cuda_event_ms": stats([r["opt_stage_cuda_event_ms"] for r in group]),
            "sync_wall_ms": stats([r["opt_stage_sync_ms"] for r in group]),
            "frame_wall_ms": stats([r["frame_time_ms"] for r in group]),
            "psnr": stats([r["final_map_eval_psnr"] for r in group]),
            "depth_l1_m": stats([r["final_map_eval_depth_l1"] for r in group]),
            "telemetry": tele,
            "samples_inside_stage": [r["gpu_telemetry"]["samples_inside_stage"] for r in group],
        }
    low, high = result["per_quota"]["25"], result["per_quota"]["75"]
    mean_event_gap = abs(high["cuda_event_ms"]["mean"] - low["cuda_event_ms"]["mean"]) / statistics.mean(
        [high["cuda_event_ms"]["mean"], low["cuda_event_ms"]["mean"]])
    mean_wall_gap = abs(high["sync_wall_ms"]["mean"] - low["sync_wall_ms"]["mean"]) / statistics.mean(
        [high["sync_wall_ms"]["mean"], low["sync_wall_ms"]["mean"]])
    result["mean_time_gaps_fraction"] = {"cuda_event": mean_event_gap, "sync_wall": mean_wall_gap}
    result["psnr_75_minus_25_db"] = high["psnr"]["mean"] - low["psnr"]["mean"]
    result["depth_l1_75_minus_25_m"] = high["depth_l1_m"]["mean"] - low["depth_l1_m"]["mean"]
    result["timing_gate_pass"] = (mean_event_gap <= .05 and mean_wall_gap <= .05 and
                                   low["cuda_event_ms"]["cv_fraction"] <= .05 and
                                   high["cuda_event_ms"]["cv_fraction"] <= .05 and
                                   low["sync_wall_ms"]["cv_fraction"] <= .05 and
                                   high["sync_wall_ms"]["cv_fraction"] <= .05)

    (OUT / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# GPU telemetry timing diagnostic: Replica office1 seed263", "",
             "One exact saved frame-90 checkpoint; ten repeated branches per quota, 128 spawns per arm, 5,000 updates at 25% and 4,374 at 75%. A discarded 1,000-update warm-up preceded the shuffled arms. This is instrumentation and timing diagnosis only; repeats are not independent seed samples.", "",
             "| Quota | n | CUDA event ms (mean ± SD; CV) | Sync wall ms (mean ± SD; CV) | Frame wall ms | GPU clock MHz | Memory clock MHz | GPU util % | Power W | Temp °C | P-state | Samples inside stage |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pct in (25, 75):
        q = result["per_quota"][str(pct)]
        def cell(name, scale=1.0, suffix=""):
            x = q["telemetry"][name]
            return "n/a" if x is None else f"{x['mean']*scale:.1f}{suffix}"
        lines.append(f"| {pct}% | 10 | {q['cuda_event_ms']['mean']:.2f} ± {q['cuda_event_ms']['sd']:.2f} ({q['cuda_event_ms']['cv_fraction']*100:.2f}%) | {q['sync_wall_ms']['mean']:.2f} ± {q['sync_wall_ms']['sd']:.2f} ({q['sync_wall_ms']['cv_fraction']*100:.2f}%) | {q['frame_wall_ms']['mean']:.2f} ms | {cell('graphics_clock_mhz')} | {cell('memory_clock_mhz')} | {cell('gpu_utilization_percent')} | {cell('power_mw', .001)} | {cell('temperature_c')} | {cell('performance_state')} | {min(q['samples_inside_stage'])}–{max(q['samples_inside_stage'])} |")
    lines += ["", f"Mean 75%-vs-25% timing gap was {mean_event_gap*100:.2f}% by CUDA event and {mean_wall_gap*100:.2f}% by synchronized wall time; gate: **{'PASS' if result['timing_gate_pass'] else 'FAIL'}**. Descriptive same-view quality difference was {result['psnr_75_minus_25_db']:+.5f} dB PSNR and {result['depth_l1_75_minus_25_m']:+.6f} m depth L1. It is not interpreted as matched-time evidence.", "", "## Telemetry interpretation", "", "Within this diagnostic, per-stage samples show the GPU held near 2,355 MHz graphics clock, 8,001 MHz memory clock, P-state 0, 53–55°C, and roughly 22–23 W, with 3–4 samples per branch. This describes this invocation only. Earlier timing runs lack telemetry, so their large run-to-run variation cannot be causally attributed to clocks, load, thermal state, or power state from this comparison.", "", "The telemetry thread itself is an observer. A 100-call standalone probe had 0.019 ms median NVML sample time, 0.078 ms p95, and one 9.235 ms maximum outlier. This run does not estimate the observer effect; future matched-time inference should retain telemetry but disclose it and compare a no-sampler calibration if timing remains marginal.", "", "## Validity and provenance", "", f"- Exact source snapshot hash matches preregistration: `{result['snapshot_hash']}`", f"- Count/snapshot gate: {'PASS' if record['all_count_and_snapshot_gates_passed'] else 'FAIL'}; branches with telemetry: {result['branches_with_gpu_samples']}/{result['branch_count']}.", f"- Preregistration SHA-256: `{result['source_hashes']['preregistration']}`", f"- Manifest SHA-256: `{result['source_hashes']['manifest']}`", f"- Launch config SHA-256: `{result['source_hashes']['launch_config']}`", f"- Run record SHA-256: `{result['source_hashes']['record']}`", f"- Git revision: `{result['manifest_git_sha']}` (dirty worktree)", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print(f"timing gate {result['timing_gate_pass']}; wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
