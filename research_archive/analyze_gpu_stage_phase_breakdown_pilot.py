#!/usr/bin/env python3
"""Summarize cache/forward/backward CUDA-event profiling on two snapshots."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase14_generalization_postfix/gpu_stage_phase_breakdown_pilot"
PREREG = json.loads((BASE / "preregistration.json").read_text())
STUDIES = {
    "replica_office1": (263, "b1d519e5461f8f02b6dede99875004158515d79bfcc358a60a5b1f06030560de"),
    "tum_fr1_rpy": (267, "4a43cc1b9793edca4052bc264885b353b8a8f8e4be461f8326443436259e9092"),
}


def summary(xs):
    return {"mean": statistics.mean(xs),
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {"status": "profiling diagnostic; two selected checkpoints",
              "preregistration_sha256": sha256(BASE / "preregistration.json"),
              "scenes": {}}
    for scene, (seed, expected_checkpoint) in STUDIES.items():
        path = BASE / scene / scene / f"seed{seed}_frame90.json"
        record = json.loads(path.read_text())
        ckpt_hash = record.get("replay_checkpoint_source_sha256") or sha256(
            Path(record["replay_checkpoint_source"]))
        prereg_key = f"{scene}_seed{seed}_frame90"
        frozen = PREREG["checkpoints"][prereg_key]
        assert frozen["checkpoint_sha256"] == expected_checkpoint == ckpt_hash
        assert frozen["snapshot_hash"] == record["base_snapshot_hash"]
        assert record["all_count_and_snapshot_gates_passed"] and len(record["arms"]) == 10
        quota_summaries = {}
        for q in (25, 75):
            rows = [a for a in record["arms"] if round(100 * float(a["action"]["mature_fraction"])) == q]
            assert len(rows) == 5 and all(a["valid"] and a["gpu_telemetry_enabled"] for a in rows)
            fields = ("opt_stage_cuda_event_ms", "cache_cuda_event_ms",
                      "optimizer_forward_cuda_event_ms", "optimizer_backward_cuda_event_ms",
                      "noncache_cuda_event_ms", "cache_time_host_ms", "opt_time_host_ms",
                      "n_cache_rebuilds", "n_cache_frozen_gaussians", "n_optimized",
                      "selected_young", "selected_mature", "mean_k", "optimizer_iteration_count",
                      "peak_vram_mb")
            values = {f: summary([float(a[f]) for a in rows]) for f in fields}
            values["residual_cuda_event_ms"] = summary([
                float(a["opt_stage_cuda_event_ms"])-float(a["cache_cuda_event_ms"])
                -float(a["optimizer_forward_cuda_event_ms"])
                -float(a["optimizer_backward_cuda_event_ms"]) for a in rows])
            values["cache_share_percent"] = 100 * values["cache_cuda_event_ms"]["mean"] / values["opt_stage_cuda_event_ms"]["mean"]
            values["backward_share_percent"] = 100 * values["optimizer_backward_cuda_event_ms"]["mean"] / values["opt_stage_cuda_event_ms"]["mean"]
            values["telemetry"] = {
                field: summary([float(a["gpu_telemetry"]["window_stats"][field]["mean"])
                                for a in rows if a["gpu_telemetry"]["window_stats"][field] is not None])
                for field in ("graphics_clock_mhz", "memory_clock_mhz", "temperature_c", "power_mw")}
            quota_summaries[str(q)] = values
        result["scenes"][scene] = {"seed": seed, "snapshot_hash": record["base_snapshot_hash"],
                                    "checkpoint_sha256": ckpt_hash, "valid_branches": len(record["arms"]),
                                    "warmup": record["warmup_record"], "quotas": quota_summaries,
                                    "record_sha256": sha256(path),
                                    "manifest_sha256": sha256(BASE / scene / "manifest.json")}
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Optimizer-stage phase breakdown pilot", "",
             "Two selected frame-90 snapshots; five repeats per quota, exact same-snapshot replay, spawn128, telemetry enabled. CUDA events separately bracket frozen-background cache builds, active-subset plus forward/render/loss, and backward plus optimizer work. Event instrumentation adds overhead; these timing means are diagnostic and should not be compared as clean performance baselines.", "",
             "| Scene | Quota | Stage CUDA ms | Cache CUDA ms | Forward/render/loss ms | Backward/step ms | Residual ms | Cache % | Backward % | Cache rebuilds | Frozen Gaussians | Optimized Gaussians | Young/mature selected | Mean K | Iterations |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scene, data in result["scenes"].items():
        for q in (25, 75):
            s=data["quotas"][str(q)]
            def m(field): return s[field]["mean"]
            lines.append(f"| {scene} | {q}% | {m('opt_stage_cuda_event_ms'):.2f} | {m('cache_cuda_event_ms'):.2f} | {m('optimizer_forward_cuda_event_ms'):.2f} | {m('optimizer_backward_cuda_event_ms'):.2f} | {m('residual_cuda_event_ms'):.2f} | {s['cache_share_percent']:.1f}% | {s['backward_share_percent']:.1f}% | {m('n_cache_rebuilds'):.0f} | {m('n_cache_frozen_gaussians'):.0f} | {m('n_optimized'):.0f} | {m('selected_young'):.0f}/{m('selected_mature'):.0f} | {m('mean_k'):.2f} | {m('optimizer_iteration_count'):.0f} |")
    lines += ["", "## Per-quota timing variation and device state", "", "| Scene | Quota | Stage CUDA mean ± SD (ms) | Forward SD (ms) | Backward/step SD (ms) | Graphics MHz | Power W | Temperature °C |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scene,data in result["scenes"].items():
        for q in (25,75):
            s=data["quotas"][str(q)]
            lines.append(f"| {scene} | {q}% | {s['opt_stage_cuda_event_ms']['mean']:.2f} ± {s['opt_stage_cuda_event_ms']['sd']:.2f} | {s['optimizer_forward_cuda_event_ms']['sd']:.2f} | {s['optimizer_backward_cuda_event_ms']['sd']:.2f} | {s['telemetry']['graphics_clock_mhz']['mean']:.0f} | {s['telemetry']['power_mw']['mean']/1000:.2f} | {s['telemetry']['temperature_c']['mean']:.1f} |")
    lines += ["", "## Interpretation", "",
              "In these two checkpoints the frozen-background cache contributes only about 1.6–2.4 ms (roughly 2% of the measured optimizer stage). The event-summed backward/optimizer section dominates at about 50–78 ms; forward/render/loss contributes about 13–17 ms. Cache construction is therefore not the primary source of the observed stage duration in this pilot.", "",
              "TUM's profiled backward/optimizer section is longer than Replica's despite fewer total Gaussians, indicating total map cardinality alone does not explain work. The selected counts and age split differ by scene and quota. Quota25's backward timing is more variable in the Replica profile than its quota75 counterpart; the current two snapshots do not explain why. Cache rebuild count is one in all measured arms, so repeated progressive cache rebuilds are not driving the variation here.", "",
              "The residual is total optimizer-stage CUDA event time minus the three measured components. It includes CUDA stream idle gaps and event boundary overhead; it is not labeled as pure host/scheduler time. Host `cache_time_ms` tracks cache CUDA events in this small sample but is not synchronization-correct.", "",
              "This is a two-checkpoint instrumentation pilot with five repeated branches per quota and inserted CUDA events. It does not establish a general workload model or matched-time quality effect. A broader profile should be run only if the per-step decomposition appears useful and after confirming event instrumentation does not materially change total timing.", "",
              "## Provenance", "",
              f"- Preregistration SHA-256: `{result['preregistration_sha256']}`"]
    for scene,data in result["scenes"].items():
        lines.append(f"- {scene} seed{data['seed']} checkpoint SHA-256: `{data['checkpoint_sha256']}`; run-record SHA-256: `{data['record_sha256']}`; manifest SHA-256: `{data['manifest_sha256']}`")
    lines.append("")
    (BASE / "report.md").write_text("\n".join(lines))
    print(f"wrote {BASE / 'report.md'}")


if __name__ == "__main__":
    main()
