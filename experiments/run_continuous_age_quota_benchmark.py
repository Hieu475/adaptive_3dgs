#!/usr/bin/env python3
"""Continuous 75%-mature allocation vs global importance on held-out trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_phase13_frozen_benchmark import build_pipeline_config, load_phase10_sequence
from experiments.run_postfix_frame_diagnostics import scene_features
from research.pipeline import OnlineReconstructionPipeline
from research.reproducibility import set_seed

W, H = 320, 240
SCENES = ("replica_room2", "tum_fr1_desk")
SEEDS = tuple(range(42, 50))
UPDATES = 5000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_arm(scene, seed, frames, intrinsics, arm, device, n_updates,
            capture_age_diagnostics=False, input_features=None):
    set_seed(seed)
    cfg = build_pipeline_config("ours", seed, budget_ms=15.0, W=W, H=H, device=device)
    control = {"micro_updates_per_frame": n_updates}
    if capture_age_diagnostics:
        control["capture_age_diagnostics"] = True
    if arm in ("mature25", "mature75", "novelty_rule"):
        control.update({"mature_age_frames": 10,
                        "mature_update_fraction": 0.25 if arm == "mature25" else 0.75})
    cfg["training"]["optimization_control"] = control
    cfg["system"]["synchronized_stage_timing"] = True
    pipe = OnlineReconstructionPipeline(config=cfg, device=device)
    pipe.initialize(rgb=frames[0]["rgb"], depth=frames[0]["depth"],
                    intrinsics=intrinsics, pose=frames[0].get("pose", torch.eye(4, device=device)))
    records = []
    t_run = time.perf_counter()
    for idx in range(1, len(frames)):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        f = frames[idx]
        requested_fraction = None
        if arm in ("mature25", "mature75"):
            requested_fraction = 0.25 if arm == "mature25" else 0.75
            pipe.config["training"]["optimization_control"]["mature_update_fraction"] = requested_fraction
        elif arm == "novelty_rule":
            if input_features is None:
                raise RuntimeError("novelty_rule requires input-only per-frame diagnostics")
            novelty = float(input_features[idx]["new_surface_ratio_voxel_5cm"])
            requested_fraction = 0.25 if novelty < 0.06 else 0.75
            pipe.config["training"]["optimization_control"]["mature_update_fraction"] = requested_fraction
        m = pipe.process_frame(rgb=f["rgb"], depth=f["depth"], gt_pose=f.get("pose"))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        row = {
            "frame": int(idx),
            "psnr": float(m["psnr"]), "ssim": float(m["ssim"]),
            "depth_l1": float(m["depth_l1"]),
            "n_gaussians": int(m["n_gaussians"]),
            "n_micro_updates": int(m.get("n_micro_updates", -1)),
            "requested_mature_fraction": requested_fraction,
            "realized_mature_updates": m.get("age_quota_mature_micro_updates"),
            "realized_young_updates": m.get("age_quota_young_micro_updates"),
            "mature_eligible": int(m.get("mature_eligible", 0)),
            "mature_update_coverage": float(m.get("mature_update_coverage", 0.0)),
            "opt_stage_sync_ms": float(m.get("opt_stage_sync_ms", 0.0)),
            "frame_wall_ms": (time.perf_counter() - t0) * 1000.0,
            "psnr_pre": float(m.get("psnr_pre", m["psnr"])),
            "psnr_post": float(m.get("psnr_post", m["psnr"])),
            "depth_l1_pre": float(m.get("depth_l1_pre", m["depth_l1"])),
            "depth_l1_post": float(m.get("depth_l1_post", m["depth_l1"])),
            "growth_pressure": float(m.get("growth_pressure", 0.0)),
            "age_stratified_diagnostics": m.get("age_stratified_diagnostics"),
        }
        if input_features is not None:
            row.update(input_features[idx])
        if not all(math.isfinite(row[k]) for k in ("psnr", "ssim", "depth_l1", "frame_wall_ms")):
            raise RuntimeError(f"nonfinite metric at {scene}/{seed}/{arm}/frame{idx}")
        records.append(row)
    summary = {
        "scene": scene, "seed": seed, "arm": arm, "n_frames_including_init": len(frames),
        "mean_psnr": float(np.mean([r["psnr"] for r in records])),
        "final_psnr": records[-1]["psnr"],
        "mean_ssim": float(np.mean([r["ssim"] for r in records])),
        "mean_depth_l1": float(np.mean([r["depth_l1"] for r in records])),
        "final_depth_l1": records[-1]["depth_l1"],
        "final_n_gaussians": records[-1]["n_gaussians"],
        "mean_frame_wall_ms": float(np.mean([r["frame_wall_ms"] for r in records])),
        "p50_frame_wall_ms": float(np.percentile([r["frame_wall_ms"] for r in records], 50)),
        "p95_frame_wall_ms": float(np.percentile([r["frame_wall_ms"] for r in records], 95)),
        "p99_frame_wall_ms": float(np.percentile([r["frame_wall_ms"] for r in records], 99)),
        "mean_opt_stage_sync_ms": float(np.mean([r["opt_stage_sync_ms"] for r in records])),
        "update_count_gate_pass": all(r["n_micro_updates"] == n_updates for r in records),
        "exact_age_split_fraction": (sum(
            r["realized_mature_updates"] == round(n_updates * r["requested_mature_fraction"])
            and r["realized_young_updates"] == n_updates - round(n_updates * r["requested_mature_fraction"])
            for r in records) / len(records)) if arm in ("mature25", "mature75", "novelty_rule") else None,
        "mean_requested_mature_fraction": (float(np.mean([r["requested_mature_fraction"] for r in records]))
                                            if arm in ("mature25", "mature75", "novelty_rule") else None),
        "total_wall_sec": time.perf_counter() - t_run,
    }
    pipe.cleanup()
    del pipe
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"summary": summary, "frames": records}


def main():
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    source_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    source_clean_at_launch = not bool(source_status.strip())
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--n_frames", type=int, default=120)
    ap.add_argument("--micro_updates", type=int, default=UPDATES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="results/phase14_generalization_postfix/continuous_age_quota")
    ap.add_argument("--preregistration", default=None,
                    help="Optional preregistration path recorded in the run manifest")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--capture_age_diagnostics", action="store_true",
                    help="Opt-in read-only age-stratified visibility/error summaries; diagnostic runs are not latency measurements")
    ap.add_argument("--arms", nargs="+", choices=("global_importance", "mature25", "mature75", "novelty_rule"),
                    default=["global_importance", "mature75"])
    ap.add_argument("--torch_num_threads", type=int, default=None)
    ap.add_argument("--torch_num_interop_threads", type=int, default=None)
    a = ap.parse_args()
    if a.torch_num_threads is not None:
        if a.torch_num_threads < 1:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(a.torch_num_threads)
    if a.torch_num_interop_threads is not None:
        if a.torch_num_interop_threads < 1:
            raise ValueError("torch_num_interop_threads must be positive")
        torch.set_num_interop_threads(a.torch_num_interop_threads)
    out = (ROOT / a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        prereg_rel = (a.preregistration if a.preregistration else
                      "results/phase14_generalization_postfix/continuous_age_quota/diagnostics_tum_fr1_desk/preregistration.md"
                      if a.capture_age_diagnostics else
                      "results/phase14_generalization_postfix/continuous_age_quota/novelty_rule_transfer/preregistration.md"
                      if "novelty_rule" in a.arms else
                      "results/phase14_generalization_postfix/continuous_age_quota/preregistration.md")
        prereg = ROOT / prereg_rel
        manifest = {
            "experiment": ("exploratory_continuous_age_quota_diagnostics" if a.capture_age_diagnostics
                           else "continuous_age_quota_novelty_rule_transfer" if "novelty_rule" in a.arms
                           else "continuous_age_quota_vs_global_importance"),
            "protocol_version": ("phase14-continuous-age-quota-diagnostics-v1" if a.capture_age_diagnostics
                                  else "phase14-novelty-rule-transfer-v1" if "novelty_rule" in a.arms
                                  else "phase14-continuous-age-quota-v1"),
            "source_revision": source_revision,
            "source_snapshot_clean_at_launch": source_clean_at_launch,
            "snapshot_commit_clean_at_launch": source_revision if source_clean_at_launch else None,
            "runtime_environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
                "device": a.device,
                "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None,
            },
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source_file_sha256": {
                rel: sha256(ROOT / rel) for rel in (
                    "research/pipeline.py",
                    "research/scheduler.py",
                    "research/rasterizer.py",
                    "experiments/run_phase13_frozen_benchmark.py",
                    "research/phase10_runtime.py",
                    "experiments/run_postfix_frame_diagnostics.py",
                )
            },
            "preregistration_sha256": sha256(prereg),
            "preregistration_path": prereg_rel,
            "scenes": a.scenes, "seeds": a.seeds, "n_frames": a.n_frames,
            "micro_updates_per_frame": a.micro_updates,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "capture_age_diagnostics": a.capture_age_diagnostics,
            "arms": a.arms,
            "novelty_rule": "if 5cm new-surface ratio < 0.06 request 25% mature; otherwise 75% mature",
            "mature_age_frames": 10, "resolution_wh": [W, H],
            "pose_convention": "ground-truth W2C via Phase 10 loader",
            "renderer_backend": "gsplat" if a.device.startswith("cuda") else "reference",
            "budget_scope": "15ms modeled optimization budget retained in config; exact update count is 5000/frame; not a wall-clock guarantee",
            "status": "preregistered; pilot and confirmatory runs must use separate output directories",
            "runs": [],
        }
        if "tum_fr3_structure_texture_near" in a.scenes:
            dataset_archive = ROOT / "datasets/TUM/rgbd_dataset_freiburg3_structure_texture_near.tgz"
            manifest["dataset_archive_path"] = str(dataset_archive.relative_to(ROOT))
            manifest["dataset_archive_sha256"] = sha256(dataset_archive) if dataset_archive.exists() else None
        if "tum_fr3_walking_halfsphere" in a.scenes:
            dataset_archive = ROOT / "datasets/TUM/rgbd_dataset_freiburg3_walking_halfsphere.tgz"
            manifest["dataset_archive_path"] = str(dataset_archive.relative_to(ROOT))
            manifest["dataset_archive_sha256"] = sha256(dataset_archive) if dataset_archive.exists() else None
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    for scene in a.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene, n_frames=a.n_frames,
                                                   H=H, W=W, device=a.device)
        input_features = scene_features(frames, intrinsics) if (
            a.capture_age_diagnostics or "novelty_rule" in a.arms) else None
        print(f"{scene}: loaded {len(frames)} frames", flush=True)
        for seed in a.seeds:
            for arm in a.arms:
                target = out / scene / f"seed{seed}_{arm}.json"
                if target.exists() and a.resume:
                    print(f"skip {target}", flush=True)
                    continue
                result = run_arm(scene, seed, frames, intrinsics, arm, a.device, a.micro_updates,
                                 capture_age_diagnostics=a.capture_age_diagnostics,
                                 input_features=input_features)
                payload = {"summary": result["summary"], "frames": result["frames"],
                           "metadata": {"scene": scene, "seed": seed, "arm": arm,
                                        "source_revision": manifest["source_revision"],
                                        "runner_sha256": manifest["runner_sha256"],
                                        "diagnostic_only_latency": a.capture_age_diagnostics}}
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload, allow_nan=False) + "\n")
                entry = {"path": str(target.relative_to(out)), "sha256": sha256(target),
                         "summary": result["summary"]}
                manifest["runs"] = [x for x in manifest["runs"] if x["path"] != entry["path"]] + [entry]
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"saved {scene} seed={seed} arm={arm} mean_psnr={result['summary']['mean_psnr']:.3f} "
                      f"quota_exact={result['summary']['exact_age_split_fraction']}", flush=True)


if __name__ == "__main__":
    main()
