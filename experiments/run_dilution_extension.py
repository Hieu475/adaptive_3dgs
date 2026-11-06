#!/usr/bin/env python3
"""Dilution extension: Replica factorial and a map-size-proportional budget control.

Reuses the conditions of ``run_dilution_factorial`` and adds ``error_budget_prop``:
raw-error selection without throttling whose modeled optimization budget at frame t is
``budget_ms * clamp(N_t / N_ref, 1, max_scale)``, with N_ref the map size at the first
processed frame. The spawn allowance stays at ``budget_ms`` (same convention as the
fixed x4 control), so extra compute cannot buy more splats. A constant budget per
Gaussian is the less coarse counterpart of the x4 intervention.

Every scene/seed/condition is written immediately, so interrupted runs resume.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_dilution_factorial import config_for as base_config_for  # noqa: E402
from experiments.run_dilution_factorial import summarize  # noqa: E402
from experiments.run_phase13_frozen_benchmark import (  # noqa: E402
    load_phase10_sequence, run_policy_trajectory,
)
from research.pipeline import OnlineReconstructionPipeline  # noqa: E402
from research.reproducibility import create_provenance_manifest  # noqa: E402

CONDITIONS = ("error_off", "error_coverage", "error_coverage_backlog",
              "temporal_off", "temporal_coverage", "temporal_coverage_backlog",
              "error_budget4", "error_budget_prop")
MAX_SCALE = 8.0
BUDGET_SCALES: list = []

_ORIGINAL_PROCESS_FRAME = OnlineReconstructionPipeline.process_frame


def _process_frame_with_proportional_budget(self, *args, **kwargs):
    sched_cfg = self.config["scheduler"]
    if sched_cfg.get("budget_prop_to_map"):
        n = self.gaussian_model.num_gaussians
        if not hasattr(self, "_prop_ref_n"):
            self._prop_ref_n = max(int(n), 1)
            self._prop_base_ms = float(self.scheduler.gpu_budget_ms)
        scale = min(max(n / self._prop_ref_n, 1.0), MAX_SCALE)
        self.scheduler.gpu_budget_ms = self._prop_base_ms * scale
        BUDGET_SCALES.append(scale)
    return _ORIGINAL_PROCESS_FRAME(self, *args, **kwargs)


OnlineReconstructionPipeline.process_frame = _process_frame_with_proportional_budget


def config_for(condition, seed, budget_ms, device):
    if condition != "error_budget_prop":
        return base_config_for(condition, seed, budget_ms, device)
    policy, cfg = base_config_for("error_off", seed, budget_ms, device)
    cfg["scheduler"]["budget_prop_to_map"] = True
    cfg["scheduler"]["densify_budget_override_ms"] = budget_ms
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["replica_office0", "replica_office1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--n_frames", type=int, default=150)
    ap.add_argument("--budget_ms", type=float, default=15.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "dilution_extension")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = create_provenance_manifest(
            config={"scenes": args.scenes, "seeds": args.seeds, "conditions": args.conditions,
                    "n_frames": args.n_frames, "budget_ms": args.budget_ms,
                    "max_budget_scale": MAX_SCALE, "age_mature_frames": 10},
            dataset_name=",".join(args.scenes),
            renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
            extra_metadata={"experiment": "dilution_extension", "status": "exploratory",
                            "pose_convention": "loader default (TUM C2W inverted; Replica extrinsics as stored)",
                            "seed_list": args.seeds,
                            "proportional_budget": "budget_ms * clamp(N_t/N_ref, 1, max_scale); spawn allowance fixed at budget_ms"},
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.70, 0)
    for scene in args.scenes:
        frames, intrinsics = load_phase10_sequence(scene_name=scene, n_frames=args.n_frames + 1,
                                                   H=240, W=320, device=args.device)
        for seed in args.seeds:
            for condition in args.conditions:
                target = args.out_dir / scene / f"{condition}_seed{seed}.json"
                if target.exists() and not args.force:
                    print(f"skip {target}", flush=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                policy, cfg = config_for(condition, seed, args.budget_ms, args.device)
                BUDGET_SCALES.clear()
                print(f"RUN {scene} {condition} seed={seed}", flush=True)
                summary, logs = run_policy_trajectory(
                    policy=policy, seed=seed, frames=frames, intrinsics=intrinsics,
                    budget_ms=args.budget_ms, device=args.device, W=320, H=240, custom_config=cfg)
                run_manifest = create_provenance_manifest(
                    config=cfg, dataset_name=scene,
                    renderer_backend="gsplat" if args.device.startswith("cuda") else "reference",
                    extra_metadata={"experiment": "dilution_extension", "condition": condition,
                                    "seed": seed, "status": "exploratory"})
                result = summarize(summary, logs)
                if BUDGET_SCALES:
                    result["mean_budget_scale"] = sum(BUDGET_SCALES) / len(BUDGET_SCALES)
                    result["max_budget_scale"] = max(BUDGET_SCALES)
                record = {"scene": scene, "seed": seed, "condition": condition,
                          "manifest": run_manifest, "config": cfg, "summary": result,
                          "budget_scale_per_frame": list(BUDGET_SCALES), "frames": logs}
                temp = target.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2))
                temp.replace(target)
                print(f"DONE {scene} {condition} seed={seed}: PSNR={summary['mean_psnr']:.3f}, "
                      f"depth={summary['mean_depth_l1']:.4f}, map={summary['N_final']}, "
                      f"mature_cov={result['mature_update_coverage']:.3f}, "
                      f"scale={result.get('mean_budget_scale', 1.0):.2f}", flush=True)


if __name__ == "__main__":
    main()
