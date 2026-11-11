#!/usr/bin/env python3
"""Summarize the autograd-backward versus optimizer.step timing pilot."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/legacy/phase14_generalization_postfix/gpu_backward_adam_breakdown_pilot"
PREREG = json.loads((BASE / "preregistration.json").read_text())
STUDIES = {"replica_office1": (263, "b1d519e5461f8f02b6dede99875004158515d79bfcc358a60a5b1f06030560de"),
           "tum_fr1_rpy": (267, "4a43cc1b9793edca4052bc264885b353b8a8f8e4be461f8326443436259e9092")}
FIELDS = ("opt_stage_cuda_event_ms", "cache_cuda_event_ms", "optimizer_forward_cuda_event_ms",
          "optimizer_backward_only_cuda_event_ms", "optimizer_step_cuda_event_ms",
          "optimizer_backward_cuda_event_ms")


def stats(xs):
    return {"mean": statistics.mean(xs), "sd": statistics.stdev(xs),
            "min": min(xs), "max": max(xs)}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = {"status": "instrumented timing diagnostic; two selected checkpoints",
           "preregistration_sha256": sha256(BASE/"preregistration.json"), "scenes": {}}
    for scene,(seed,expected_hash) in STUDIES.items():
        path=BASE/scene/scene/f"seed{seed}_frame90.json"
        d=json.loads(path.read_text())
        assert d["all_count_and_snapshot_gates_passed"] and len(d["arms"])==10
        assert d["replay_checkpoint_source_sha256"]==expected_hash
        assert d["base_snapshot_hash"]==PREREG["checkpoints"][f"{scene}_seed{seed}_frame90"]["snapshot_hash"]
        quota={}
        for q in (25,75):
            rows=[a for a in d["arms"] if round(100*a["action"]["mature_fraction"])==q]
            assert len(rows)==5 and all(a["valid"] for a in rows)
            quota[str(q)]={f:stats([float(a[f]) for a in rows]) for f in FIELDS}
            quota[str(q)]["iterations_mean"]=statistics.mean(a["optimizer_iteration_count"] for a in rows)
        out["scenes"][scene]={"seed":seed,"snapshot_hash":d["base_snapshot_hash"],"checkpoint_sha256":expected_hash,
                              "valid_branches":10,"quotas":quota,"record_sha256":sha256(path),
                              "manifest_sha256":sha256(BASE/scene/"manifest.json")}
    (BASE/"analysis.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    lines=["# Autograd backward vs optimizer-step timing pilot","",
           "The combined backward/optimizer CUDA-event block dominated the eight-checkpoint profile. This follow-up brackets `loss.backward()` and `optimizer.step()` separately on the same two selected checkpoints. Five same-snapshot repeats per quota; spawn128; 5,000 updates at quota25 and 4,374 at quota75. Event instrumentation is enabled; timings are diagnostic, not clean performance baselines.","",
           "| Scene | Quota | Total stage ms | Cache ms | Forward/render/loss ms | Autograd backward ms | optimizer.step ms | Combined backward+step ms | Iterations |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scene,d in out["scenes"].items():
        for q in (25,75):
            x=d["quotas"][str(q)]
            def f(k):return f"{x[k]['mean']:.2f} ± {x[k]['sd']:.2f}"
            lines.append(f"| {scene} | {q}% | {f('opt_stage_cuda_event_ms')} | {f('cache_cuda_event_ms')} | {f('optimizer_forward_cuda_event_ms')} | {f('optimizer_backward_only_cuda_event_ms')} | {f('optimizer_step_cuda_event_ms')} | {f('optimizer_backward_cuda_event_ms')} | {x['iterations_mean']:.0f} |")
    lines += ["","## Interpretation","",
              "Autograd backward is the largest measured component in all four cells: about 44.7–47.2 ms on Replica and 68.0–68.6 ms on TUM. `optimizer.step()` is about 7.1–7.2 ms on Replica and 7.5–8.3 ms on TUM. Thus the roughly 21–23 ms cross-scene gap in backward/optimizer time is mostly in the backward path, while the optimizer-step gap is comparatively small. Cache remains around 1.7–2.6 ms.","",
              "The backward interval covers the full autograd graph for the active render and losses, including renderer gradients; it does not isolate a particular gsplat kernel. CUDA event intervals include stream gaps. This two-checkpoint diagnostic does not establish why the TUM backward path takes longer, nor does it constitute equal-time quality evidence. The next useful systems measurement is to profile renderer forward/backward kernel work by active primitive/tile workload on more snapshots, without changing policy claims.","",
              "## Provenance","",f"- Preregistration SHA-256: `{out['preregistration_sha256']}`"]
    for s,d in out["scenes"].items():
        lines.append(f"- {s} seed{d['seed']} checkpoint SHA-256: `{d['checkpoint_sha256']}`; run record SHA-256: `{d['record_sha256']}`; manifest SHA-256: `{d['manifest_sha256']}`")
    lines.append("")
    (BASE/"report.md").write_text("\n".join(lines))
    print(BASE/"report.md")


if __name__=="__main__":main()
