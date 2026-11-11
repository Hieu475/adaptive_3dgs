#!/usr/bin/env python3
"""Analyze per-checkpoint NVML observer-effect timing comparisons."""
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/gpu_telemetry_observer_effect_multicheckpoint"
PREREG = json.loads((BASE / "preregistration.json").read_text())
SCENES = {
    "replica_office1": (BASE / "replica_office1", [260, 261, 262, 263]),
    "tum_fr1_rpy": (BASE / "tum_fr1_rpy", [264, 265, 266, 267]),
}


def mean_sd(xs):
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "sd": sd, "min": min(xs), "max": max(xs),
            "cv_fraction": sd / m if m else None}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scene_data = {}
    for scene, (folder, seeds) in SCENES.items():
        per_seed = []
        for seed in seeds:
            path = folder / scene / f"seed{seed}_frame90.json"
            record = json.loads(path.read_text())
            expected_ckpt = PREREG["checkpoints"][scene][str(seed)]
            source_ckpt = Path(record["replay_checkpoint_source"])
            observed_ckpt = record.get("replay_checkpoint_source_sha256") or sha256(source_ckpt)
            if observed_ckpt != expected_ckpt:
                raise ValueError(f"Checkpoint hash mismatch for {scene} seed {seed}")
            if record["base_snapshot_hash"] == "" or not record["all_count_and_snapshot_gates_passed"]:
                raise ValueError(f"Snapshot/count gate failed for {scene} seed {seed}")
            arms = record["arms"]
            if len(arms) != 20:
                raise ValueError(f"Expected 20 branches for {scene} seed {seed}")
            pairs = defaultdict(dict)
            for arm in arms:
                q = round(100 * float(arm["action"]["mature_fraction"]))
                if q not in (25, 75) or arm["action"]["spawn_count"] != 128:
                    raise ValueError(f"Unexpected intervention in {path}")
                mode = "on" if arm["gpu_telemetry_enabled"] else "off"
                rep = int(arm["action"]["repeat_index"])
                if mode in pairs[(q, rep)]:
                    raise ValueError(f"Duplicate paired cell {scene} {seed} {q} {rep} {mode}")
                if mode == "on":
                    tele = arm["gpu_telemetry"]
                    if tele is None or tele["samples_inside_stage"] < 1:
                        raise ValueError(f"Missing telemetry samples in {path}: {arm['action']['name']}")
                elif arm["gpu_telemetry"] is not None:
                    raise ValueError(f"Telemetry-off arm contains samples: {arm['action']['name']}")
                if not arm["valid"]:
                    raise ValueError(f"Action count/quota invalid: {arm['action']['name']}")
                pairs[(q, rep)][mode] = arm
            if len(pairs) != 10 or any(set(pair) != {"on", "off"} for pair in pairs.values()):
                raise ValueError(f"Incomplete telemetry pairs for {scene} seed {seed}")

            seed_result = {"scene": scene, "seed": seed,
                           "snapshot_hash": record["base_snapshot_hash"],
                           "checkpoint_sha256": observed_ckpt,
                           "valid_branches": sum(bool(a["valid"]) for a in arms),
                           "warmup": {"sync_wall_ms": record["warmup_record"]["opt_stage_sync_ms"],
                                      "cuda_event_ms": record["warmup_record"]["opt_stage_cuda_event_ms"],
                                      "gpu_telemetry": record["warmup_record"].get("gpu_telemetry")},
                           "per_quota": {}}
            for q in (25, 75):
                q_pairs = [pairs[(q, i)] for i in range(5)]
                q_result = {"telemetry_on": {}, "telemetry_off": {},
                            "paired_on_minus_off": {}, "on_gpu_state_mean": {},
                            "on_samples_per_branch": []}
                for mode in ("on", "off"):
                    rows = [p[mode] for p in q_pairs]
                    for field in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms", "frame_time_ms"):
                        q_result["telemetry_" + mode][field] = mean_sd([float(a[field]) for a in rows])
                for field in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms", "frame_time_ms"):
                    deltas = [float(p["on"][field]) - float(p["off"][field]) for p in q_pairs]
                    q_result["paired_on_minus_off"][field] = {
                        **mean_sd(deltas),
                        "mean_percent_of_off_mean": 100 * statistics.mean(deltas) /
                            statistics.mean([float(p["off"][field]) for p in q_pairs]),
                        "per_repeat_deltas": deltas,
                        "positive_pairs": sum(x > 0 for x in deltas),
                        "negative_pairs": sum(x < 0 for x in deltas),
                    }
                telemetry_rows = [p["on"]["gpu_telemetry"] for p in q_pairs]
                q_result["on_samples_per_branch"] = [x["samples_inside_stage"] for x in telemetry_rows]
                for field in ("graphics_clock_mhz", "memory_clock_mhz", "temperature_c",
                              "power_mw", "gpu_utilization_percent", "memory_used_bytes"):
                    vals = [t["window_stats"][field]["mean"] for t in telemetry_rows
                            if t["window_stats"][field] is not None and "mean" in t["window_stats"][field]]
                    q_result["on_gpu_state_mean"][field] = statistics.mean(vals) if vals else None
                seed_result["per_quota"][str(q)] = q_result
            per_seed.append(seed_result)

        scene_summary = {}
        for q in (25, 75):
            scene_summary[str(q)] = {}
            for field in ("opt_stage_cuda_event_ms", "opt_stage_sync_ms"):
                seed_deltas = [s["per_quota"][str(q)]["paired_on_minus_off"][field]["mean"]
                               for s in per_seed]
                seed_rel = [s["per_quota"][str(q)]["paired_on_minus_off"][field]["mean_percent_of_off_mean"]
                            for s in per_seed]
                scene_summary[str(q)][field] = {
                    "mean_delta_across_checkpoints_ms": statistics.mean(seed_deltas),
                    "sd_delta_across_checkpoints_ms": statistics.stdev(seed_deltas),
                    "positive_checkpoints": sum(x > 0 for x in seed_deltas),
                    "negative_checkpoints": sum(x < 0 for x in seed_deltas),
                    "per_checkpoint_mean_delta_ms": seed_deltas,
                    "mean_relative_shift_percent_across_checkpoints": statistics.mean(seed_rel),
                    "relative_shift_range_percent": [min(seed_rel), max(seed_rel)],
                }
        scene_data[scene] = {"n_checkpoints": len(seeds), "valid_branches": sum(s["valid_branches"] for s in per_seed),
                             "per_seed": per_seed, "observer_effect_summary": scene_summary}

    result = {"status": "exploratory observer-effect study; repeated branches nested within checkpoints",
              "independent_units": "four saved checkpoints within each scene; do not pool scenes",
              "preregistration_sha256": sha256(BASE / "preregistration.json"),
              "scenes": scene_data,
              "limitations": ["Only four saved checkpoints per scene and five repeated pairs per quota/checkpoint.",
                              "No statistical equivalence or quota quality claim.",
                              "Observer-effect estimates are conditional on these replay workloads and GPU state."]}
    (BASE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Multi-checkpoint NVML observer-effect timing study", "",
             "Four exact saved checkpoints per scene; 5 matched repeat pairs per quota and telemetry mode within each checkpoint. Each pair differed only in whether the background NVML sampler was enabled. All arms used spawn128, fixed quota-specific update counts, the same common snapshot/RNG reset, and a 1,000-update discarded warm-up. The 8 checkpoint snapshots are the independent units within their scenes; repeated actions are nested. No cross-scene pooling is used.", "",
             "## Per-checkpoint results", ""]
    for scene, data in scene_data.items():
        lines += [f"### {scene}", "",
                  "| Seed | Valid branches | 25% CUDA on−off ms | 25% wall on−off ms | 75% CUDA on−off ms | 75% wall on−off ms | Mean GPU clock MHz | Power W |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for seed in data["per_seed"]:
            q25, q75 = seed["per_quota"]["25"], seed["per_quota"]["75"]
            def delta(q, metric):
                d=q["paired_on_minus_off"][metric]
                return f"{d['mean']:+.3f} ± {d['sd']:.3f}"
            clock=q25["on_gpu_state_mean"]["graphics_clock_mhz"]
            power=q25["on_gpu_state_mean"]["power_mw"]
            lines.append(f"| {seed['seed']} | {seed['valid_branches']}/20 | {delta(q25,'opt_stage_cuda_event_ms')} | {delta(q25,'opt_stage_sync_ms')} | {delta(q75,'opt_stage_cuda_event_ms')} | {delta(q75,'opt_stage_sync_ms')} | {clock:.0f} | {power/1000:.2f} |")
        lines += ["", "#### Absolute stage timing and repeat CV", "",
                  "| Seed | 25% off/on CUDA event ms | 25% off/on CV | 75% off/on CUDA event ms | 75% off/on CV | Mean GPU temp °C |", "|---:|---:|---:|---:|---:|---:|"]
        for seed in data["per_seed"]:
            q25, q75 = seed["per_quota"]["25"], seed["per_quota"]["75"]
            a=q25["telemetry_off"]["opt_stage_cuda_event_ms"]
            b=q25["telemetry_on"]["opt_stage_cuda_event_ms"]
            c=q75["telemetry_off"]["opt_stage_cuda_event_ms"]
            d=q75["telemetry_on"]["opt_stage_cuda_event_ms"]
            temp=q25["on_gpu_state_mean"]["temperature_c"]
            lines.append(f"| {seed['seed']} | {a['mean']:.2f} / {b['mean']:.2f} | {a['cv_fraction']*100:.1f}% / {b['cv_fraction']*100:.1f}% | {c['mean']:.2f} / {d['mean']:.2f} | {c['cv_fraction']*100:.1f}% / {d['cv_fraction']*100:.1f}% | {temp:.1f} |")
        lines += ["", "#### Scene-level observer shifts (descriptive over checkpoint means)", "",
                  "| Quota | Timing metric | Mean on−off ms (SD across checkpoints) | Relative shift mean [range] | Checkpoints +/− |", "|---:|---|---:|---:|---:|"]
        for q in (25,75):
            for field,label in (("opt_stage_cuda_event_ms","CUDA event"),("opt_stage_sync_ms","Sync wall")):
                x=data["observer_effect_summary"][str(q)][field]
                lo,hi=x["relative_shift_range_percent"]
                lines.append(f"| {q}% | {label} | {x['mean_delta_across_checkpoints_ms']:+.3f} ({x['sd_delta_across_checkpoints_ms']:.3f}) | {x['mean_relative_shift_percent_across_checkpoints']:+.2f}% [{lo:+.2f}, {hi:+.2f}] | {x['positive_checkpoints']}/{x['negative_checkpoints']} |")
        lines.append("")
    lines += ["## Interpretation", ""]
    for scene, data in scene_data.items():
        d25=data["observer_effect_summary"]["25"]["opt_stage_cuda_event_ms"]
        d75=data["observer_effect_summary"]["75"]["opt_stage_cuda_event_ms"]
        lines.append(f"- **{scene}:** relative telemetry-on shift averaged {d25['mean_relative_shift_percent_across_checkpoints']:+.2f}% at quota25 (signs +{d25['positive_checkpoints']}/−{d25['negative_checkpoints']}) and {d75['mean_relative_shift_percent_across_checkpoints']:+.2f}% at quota75 (signs +{d75['positive_checkpoints']}/−{d75['negative_checkpoints']}).")
    lines += ["", "The report separates two sources of variation: telemetry-on minus telemetry-off timing within an exact checkpoint, and timing changes across checkpoints. Read the per-seed deltas first; a scene-level mean can conceal opposite effects. A small mean shift does not prove equivalence, particularly with four checkpoints. The quota25 and quota75 workloads are summarized separately because observer impact may depend on the work pattern.", "",
              "This study does not test a matched-time quality effect. Fixed quota25 used 5,000 updates and quota75 used 4,374, as frozen from the earlier pilot. Timing gates and same-view quality are descriptive; no update counts were retuned after observing results. Telemetry-off branches pause sampling but keep the NVML library initialized and the sampler thread idle.", "",
              "## Provenance", "",
              f"- Preregistration SHA-256: `{result['preregistration_sha256']}`",
              "- Per-run manifests, checkpoint hashes, exact arm order, timestamps, and telemetry samples are stored in `replica_office1/` and `tum_fr1_rpy/`.",
              "- Runs used a dirty worktree; this is exploratory systems evidence pending clean-checkout reproduction.", ""]
    (BASE / "report.md").write_text("\n".join(lines))
    print(f"wrote {BASE / 'report.md'}")


if __name__ == "__main__":
    main()
