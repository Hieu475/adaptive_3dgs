#!/usr/bin/env python3
"""Analyze CUDA-event stage phase breakdown across eight saved checkpoints."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/gpu_stage_phase_breakdown_multicheckpoint"
PREREG = json.loads((BASE / "preregistration.json").read_text())
SOURCE_ROOT = ROOT / "results/legacy/phase14_generalization_postfix/matched_time_spawn128_cross_scene_replication_v1"
STUDIES = {
    "replica_office1": (BASE / "replica_office1", SOURCE_ROOT,
                        [260, 261, 262, 263]),
    "tum_fr1_rpy": (BASE / "tum_fr1_rpy", SOURCE_ROOT / "tum_seeds264_267",
                    [264, 265, 266, 267]),
}
COMPONENTS = ("opt_stage_cuda_event_ms", "cache_cuda_event_ms",
              "optimizer_forward_cuda_event_ms", "optimizer_backward_cuda_event_ms",
              "noncache_cuda_event_ms", "cache_time_host_ms", "opt_time_host_ms")


def mean_sd(xs):
    return {"mean": statistics.mean(xs),
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {"status": "systems profiling; eight selected checkpoints",
              "preregistration_sha256": sha256(BASE / "preregistration.json"),
              "scenes": {}}
    for scene, (folder, source_root, seeds) in STUDIES.items():
        seed_results = []
        for seed in seeds:
            path = folder / scene / f"seed{seed}_frame90.json"
            source_record = json.loads((source_root / scene / f"seed{seed}_frame90.json").read_text())
            record = json.loads(path.read_text())
            expected = PREREG["checkpoints"][scene][str(seed)]
            checkpoint_path = source_root / "snapshots" / scene / f"seed{seed}_frame90.checkpoint.pt"
            checkpoint_hash = sha256(checkpoint_path)
            assert checkpoint_hash == expected
            assert record.get("replay_checkpoint_source_sha256") == expected
            assert record["base_snapshot_hash"] == source_record["base_snapshot_hash"]
            assert record["all_count_and_snapshot_gates_passed"] and len(record["arms"]) == 10
            assert record["warmup_record"].get("cache_cuda_event_ms") is not None
            q_data = {}
            for q in (25, 75):
                rows = [a for a in record["arms"] if round(100*float(a["action"]["mature_fraction"])) == q]
                assert len(rows) == 5 and all(a["valid"] and a["gpu_telemetry_enabled"] for a in rows)
                qstats = {field: mean_sd([float(a[field]) for a in rows]) for field in COMPONENTS}
                qstats["residual_cuda_event_ms"] = mean_sd([
                    float(a["opt_stage_cuda_event_ms"])-float(a["cache_cuda_event_ms"])
                    -float(a["optimizer_forward_cuda_event_ms"])
                    -float(a["optimizer_backward_cuda_event_ms"]) for a in rows])
                qstats["cache_share_percent"] = 100*qstats["cache_cuda_event_ms"]["mean"]/qstats["opt_stage_cuda_event_ms"]["mean"]
                qstats["backward_share_percent"] = 100*qstats["optimizer_backward_cuda_event_ms"]["mean"]/qstats["opt_stage_cuda_event_ms"]["mean"]
                counters = ("n_cache_rebuilds", "n_cache_frozen_gaussians", "n_optimized",
                            "selected_young", "selected_mature", "mean_k", "optimizer_iteration_count")
                qstats["work_counters"] = {field: mean_sd([float(a[field]) for a in rows]) for field in counters}
                t_fields = ("graphics_clock_mhz", "memory_clock_mhz", "temperature_c", "power_mw")
                qstats["telemetry"] = {}
                for field in t_fields:
                    values = [a["gpu_telemetry"]["window_stats"][field]["mean"] for a in rows
                              if a["gpu_telemetry"]["window_stats"][field] is not None]
                    qstats["telemetry"][field] = mean_sd(values)
                q_data[str(q)] = qstats
            seed_results.append({"seed": seed, "snapshot_hash": record["base_snapshot_hash"],
                                 "checkpoint_sha256": checkpoint_hash, "valid_branches": 10,
                                 "record_sha256": sha256(path), "quotas": q_data})
        summaries = {}
        for q in (25,75):
            summaries[str(q)] = {}
            for field in COMPONENTS:
                vals=[s["quotas"][str(q)][field]["mean"] for s in seed_results]
                summaries[str(q)][field]=mean_sd(vals)
            cache_shares=[s["quotas"][str(q)]["cache_share_percent"] for s in seed_results]
            backward_shares=[s["quotas"][str(q)]["backward_share_percent"] for s in seed_results]
            summaries[str(q)]["cache_share_percent"]=mean_sd(cache_shares)
            summaries[str(q)]["backward_share_percent"]=mean_sd(backward_shares)
            summaries[str(q)]["checkpoints_cache_under_5pct"]=sum(x<5 for x in cache_shares)
        result["scenes"][scene]={"n_checkpoints":len(seed_results),"valid_branches":sum(s["valid_branches"] for s in seed_results),
                                  "per_seed":seed_results,"across_checkpoint_means":summaries}
    (BASE/"analysis.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Multi-checkpoint optimizer-stage phase breakdown", "",
           "Eight saved frame-90 checkpoints (four per scene), five repeats per quota/checkpoint; all 80 scored branches passed snapshot/spawn/quota gates. CUDA events measure cache construction, active-subset plus forward/render/loss, and backward plus optimizer. All runs used NVML telemetry. Instrumentation is enabled in all cells; timings are diagnostic and not clean performance baselines.", ""]
    for scene,data in result["scenes"].items():
        lines += [f"## {scene}","",
                  "| Seed | Quota | Stage ms | Cache ms | Forward ms | Backward/step ms | Residual ms | Cache share | Backward share | Cache rebuilds | Frozen Gs | Optimized | Young/mature selected | Mean K | CV stage | GPU MHz | Power W |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in data["per_seed"]:
            for q in (25,75):
                x=s["quotas"][str(q)]
                w=x["work_counters"]
                lines.append(f"| {s['seed']} | {q}% | {x['opt_stage_cuda_event_ms']['mean']:.2f} ± {x['opt_stage_cuda_event_ms']['sd']:.2f} | {x['cache_cuda_event_ms']['mean']:.2f} | {x['optimizer_forward_cuda_event_ms']['mean']:.2f} | {x['optimizer_backward_cuda_event_ms']['mean']:.2f} | {x['residual_cuda_event_ms']['mean']:.2f} | {x['cache_share_percent']:.1f}% | {x['backward_share_percent']:.1f}% | {w['n_cache_rebuilds']['mean']:.0f} | {w['n_cache_frozen_gaussians']['mean']:.0f} | {w['n_optimized']['mean']:.0f} | {w['selected_young']['mean']:.0f}/{w['selected_mature']['mean']:.0f} | {w['mean_k']['mean']:.2f} | {x['opt_stage_cuda_event_ms']['sd']/x['opt_stage_cuda_event_ms']['mean']*100:.1f}% | {x['telemetry']['graphics_clock_mhz']['mean']:.0f} | {x['telemetry']['power_mw']['mean']/1000:.1f} |")
        lines += ["", "### Across-checkpoint phase summary (mean of four checkpoint means ± SD)", "",
                  "| Quota | Stage ms | Cache ms | Forward ms | Backward/step ms | Cache share | Backward share | Checkpoints cache <5% |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for q in (25,75):
            x=data["across_checkpoint_means"][str(q)]
            def fmt(field): return f"{x[field]['mean']:.2f} ± {x[field]['sd']:.2f}"
            lines.append(f"| {q}% | {fmt('opt_stage_cuda_event_ms')} | {fmt('cache_cuda_event_ms')} | {fmt('optimizer_forward_cuda_event_ms')} | {fmt('optimizer_backward_cuda_event_ms')} | {x['cache_share_percent']['mean']:.1f}% ± {x['cache_share_percent']['sd']:.1f} | {x['backward_share_percent']['mean']:.1f}% ± {x['backward_share_percent']['sd']:.1f} | {x['checkpoints_cache_under_5pct']}/4 |")
        lines.append("")
    lines += ["## Interpretation", "",
              "The per-checkpoint rows show whether cache cost or backward/optimizer work explains the cross-scene and quota timing differences. Cache shares are reported per checkpoint; the across-checkpoint summary is descriptive and does not treat repeats as independent.", "",
              "The CUDA-event residual (total minus cache, forward, and backward/step events) includes stream idle gaps and event-boundary overhead; it is not pure host scheduler time. The host cache timer is retained for reference but is not synchronization-correct. No equal-time quality claim is made because quota25 and quota75 use different update counts.", "",
              "## Provenance", "",
              f"- Preregistration SHA-256: `{result['preregistration_sha256']}`",
              "- Per-checkpoint source checkpoint and run-record hashes are stored in `analysis.json`.",
              "- The manifests identify dirty-worktree runs; clean-checkout reproduction remains necessary before paper use.", ""]
    (BASE/"report.md").write_text("\n".join(lines))
    print(f"wrote {BASE/'report.md'}")


if __name__=="__main__": main()
