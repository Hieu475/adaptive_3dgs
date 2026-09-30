#!/usr/bin/env python3
r"""Factorial Ablation Study: Causal Component Isolation between Selection and Throttling.

Factorial Design:
  Factor A: Selection Policy:
    - error_only: Rank primitives purely by photometric/depth error
    - error_influence: Rank by Error × Contribution Mass (FAA)
    - learned_utility: Two-Head MLP predicting \hat{U} = \hat{\Delta Q} / \hat{\Delta T}
  Factor B: Map Growth Throttling:
    - OFF: Unconstrained densification (coverage_throttling=False, max_warmup_queue=999999)
    - ON: Dual Throttling (coverage_throttling=True, max_warmup_queue=500)

Conditions:
  - B0: Error Only               | Throttling OFF
  - B1: Error × Influence        | Throttling OFF
  - B2: Learned Utility          | Throttling OFF
  - B3: Error Only               | Throttling ON
  - B4: Error × Influence        | Throttling ON
  - B5: Learned Utility          | Throttling ON

Causal Isolations:
  - Utility Effect (w/o Throttling): B2 - B1
  - Throttling Effect on Heuristic:  B4 - B1
  - Utility Effect (w/ Throttling):  B5 - B4
  - Interaction Effect:             (B5 - B4) - (B2 - B1)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_phase13_frozen_benchmark import (
    load_phase10_sequence,
    build_pipeline_config,
    run_policy_trajectory,
)
from research.scheduler import OptimizationPolicy


FACTORIAL_CONDITIONS = [
    {"id": "B0", "name": "Error Only (No Throttle)",         "policy": "error_only",       "throttling": False, "use_learned": False},
    {"id": "B1", "name": "Error × Influence (No Throttle)",  "policy": "error_influence",  "throttling": False, "use_learned": False},
    {"id": "B2", "name": "Learned Utility (No Throttle)",    "policy": "learned_utility",  "throttling": False, "use_learned": True},
    {"id": "B3", "name": "Error Only + Throttled",           "policy": "error_only",       "throttling": True,  "use_learned": False},
    {"id": "B4", "name": "Error × Influence + Throttled",    "policy": "error_influence",  "throttling": True,  "use_learned": False},
    {"id": "B5", "name": "Learned Utility + Throttled",      "policy": "learned_utility",  "throttling": True,  "use_learned": True},
]


def run_factorial_experiment(
    scene: str = "tum_fr2_xyz",
    n_frames: int = 150,
    seeds: List[int] = None,
    budget_ms: float = 15.0,
    device: str = "cuda",
    out_dir: str = "results/factorial_ablation",
    condition_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute factorial matrix across conditions and seeds."""
    if seeds is None:
        seeds = [42, 43, 44]

    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240

    active_conditions = FACTORIAL_CONDITIONS
    if condition_ids:
        active_conditions = [c for c in FACTORIAL_CONDITIONS if c["id"] in condition_ids]

    print("=" * 95)
    print("  FACTORIAL ABLATION: SELECTION POLICY × MAP GROWTH THROTTLING")
    print("=" * 95)
    print(f"  Scene:      {scene}")
    print(f"  Frames:     {n_frames}")
    print(f"  Seeds:      {seeds}")
    print(f"  Conditions: {[c['id'] for c in active_conditions]}")
    print(f"  Device:     {device}")
    print(f"  Output:     {out_path}")
    print("=" * 95)

    # 1. Preload scene
    print(f"\n>> Loading {scene} ({n_frames + 1} frames)...")
    frames, intrinsics = load_phase10_sequence(
        scene_name=scene, n_frames=n_frames + 1, H=H, W=W, device=device
    )
    print(f">> Loaded {len(frames)} frames successfully.\n")

    raw_runs = []
    total_runs = len(active_conditions) * len(seeds)
    idx = 0
    t_start = time.perf_counter()

    for cond in active_conditions:
        for seed in seeds:
            idx += 1
            print(f"[{idx}/{total_runs}] Condition {cond['id']} ({cond['name']}) | Seed={seed}")

            cfg = build_pipeline_config(
                policy=cond["policy"],
                seed=seed,
                budget_ms=budget_ms,
                W=W,
                H=H,
                device=device,
                throttling=cond["throttling"],
                use_learned_utility=cond["use_learned"],
            )

            try:
                summary, _ = run_policy_trajectory(
                    policy=cond["policy"],
                    seed=seed,
                    frames=frames,
                    intrinsics=intrinsics,
                    budget_ms=budget_ms,
                    device=device,
                    W=W,
                    H=H,
                    custom_config=cfg,
                )
                rec = {
                    "condition_id": cond["id"],
                    "condition_name": cond["name"],
                    "policy": cond["policy"],
                    "throttling": cond["throttling"],
                    "use_learned": cond["use_learned"],
                    "seed": seed,
                    "mean_psnr": summary["mean_psnr"],
                    "final_psnr": summary["final_psnr"],
                    "mean_ssim": summary["mean_ssim"],
                    "final_ssim": summary["final_ssim"],
                    "mean_depth_l1": summary["mean_depth_l1"],
                    "N_final": summary["N_final"],
                    "fps": summary["fps"],
                    "mean_opt_time_ms": summary["mean_opt_time_ms"],
                    "mean_frame_time_ms": summary["mean_frame_time_ms"],
                }
                raw_runs.append(rec)
                print(f"   -> Mean PSNR: {rec['mean_psnr']:.2f} dB | Final PSNR: {rec['final_psnr']:.2f} dB | "
                      f"SSIM: {rec['mean_ssim']:.4f} | N_final: {rec['N_final']:,} | FPS: {rec['fps']:.1f}")
            except Exception as e:
                print(f"   -> ERROR in {cond['id']} seed {seed}: {e}")
                raise e

    # 2. Aggregate per condition
    aggregated = {}
    for cond in active_conditions:
        cid = cond["id"]
        c_runs = [r for r in raw_runs if r["condition_id"] == cid]
        if not c_runs:
            continue
        psnr_vals = [r["mean_psnr"] for r in c_runs]
        final_psnr_vals = [r["final_psnr"] for r in c_runs]
        ssim_vals = [r["mean_ssim"] for r in c_runs]
        n_final_vals = [r["N_final"] for r in c_runs]
        fps_vals = [r["fps"] for r in c_runs]

        aggregated[cid] = {
            "id": cid,
            "name": cond["name"],
            "policy": cond["policy"],
            "throttling": cond["throttling"],
            "use_learned": cond["use_learned"],
            "mean_psnr": float(np.mean(psnr_vals)),
            "std_psnr": float(np.std(psnr_vals)),
            "final_psnr": float(np.mean(final_psnr_vals)),
            "std_final_psnr": float(np.std(final_psnr_vals)),
            "mean_ssim": float(np.mean(ssim_vals)),
            "std_ssim": float(np.std(ssim_vals)),
            "N_final": float(np.mean(n_final_vals)),
            "fps": float(np.mean(fps_vals)),
        }

    # 3. Compute Causal Effect Isolations
    effects = {}
    if "B0" in aggregated and "B3" in aggregated:
        effects["Throttling_on_Error"] = aggregated["B3"]["mean_psnr"] - aggregated["B0"]["mean_psnr"]
    if "B1" in aggregated and "B4" in aggregated:
        effects["Throttling_on_ErrorInfluence"] = aggregated["B4"]["mean_psnr"] - aggregated["B1"]["mean_psnr"]
    if "B1" in aggregated and "B2" in aggregated:
        effects["LearnedUtility_no_Throttling"] = aggregated["B2"]["mean_psnr"] - aggregated["B1"]["mean_psnr"]
    if "B4" in aggregated and "B5" in aggregated:
        effects["LearnedUtility_with_Throttling"] = aggregated["B5"]["mean_psnr"] - aggregated["B4"]["mean_psnr"]
    if "LearnedUtility_with_Throttling" in effects and "LearnedUtility_no_Throttling" in effects:
        effects["Interaction_Effect"] = effects["LearnedUtility_with_Throttling"] - effects["LearnedUtility_no_Throttling"]

    report_data = {
        "scene": scene,
        "n_frames": n_frames,
        "seeds": seeds,
        "budget_ms": budget_ms,
        "aggregated": aggregated,
        "effects": effects,
        "raw_runs": raw_runs,
    }

    # Save JSON
    with open(out_path / "factorial_ablation_results.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Summary
    md_content = generate_markdown_report(report_data)
    with open(out_path / "factorial_ablation_report.md", "w") as f:
        f.write(md_content)

    print(f"\n>> Factorial study completed. Report saved to: {out_path / 'factorial_ablation_report.md'}\n")
    print(md_content)
    return report_data


def generate_markdown_report(data: Dict[str, Any]) -> str:
    agg = data["aggregated"]
    eff = data["effects"]
    lines = [
        "# Factorial Ablation Study: Causal Component Isolation",
        "",
        f"**Scene:** `{data['scene']}` | **Frames:** {data['n_frames']} | **Seeds:** {data['seeds']} | **Budget:** {data['budget_ms']} ms",
        "",
        "## 1. Full 2×3 Factorial Matrix",
        "",
        "| ID | Selection Policy | Throttling | Mean PSNR (dB) | Final PSNR (dB) | Mean SSIM | Map Size ($N$) | FPS |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for cid in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        if cid in agg:
            c = agg[cid]
            thr_str = "ON" if c["throttling"] else "OFF"
            lines.append(
                f"| **{cid}** | `{c['policy']}` | **{thr_str}** | "
                f"{c['mean_psnr']:.2f} ± {c['std_psnr']:.2f} | "
                f"{c['final_psnr']:.2f} ± {c['std_final_psnr']:.2f} | "
                f"{c['mean_ssim']:.4f} | {int(c['N_final']):,} | {c['fps']:.1f} |"
            )

    lines.extend([
        "",
        "## 2. Causal Effect Decomposition",
        "",
        "| Research Question | Factorial Contrast | Empirical Effect ($\\\\Delta$ Mean PSNR) | Scientific Interpretation |",
        "|:---|:---:|:---:|:---|",
    ])

    if "Throttling_on_Error" in eff:
        d = eff["Throttling_on_Error"]
        lines.append(f"| **Does Throttling rescue raw Error?** | $B_3 - B_0$ | **{d:+.2f} dB** | Pure compute conservation effect |")
    if "Throttling_on_ErrorInfluence" in eff:
        d = eff["Throttling_on_ErrorInfluence"]
        lines.append(f"| **Does Throttling rescue Error×Influence?** | $B_4 - B_1$ | **{d:+.2f} dB** | Impact of map containment on sensitivity ranking |")
    if "LearnedUtility_no_Throttling" in eff:
        d = eff["LearnedUtility_no_Throttling"]
        lines.append(f"| **Does Learned Utility beat Heuristic (No Throttle)?** | $B_2 - B_1$ | **{d:+.2f} dB** | Intrinsic value of learned utility without throttling |")
    if "LearnedUtility_with_Throttling" in eff:
        d = eff["LearnedUtility_with_Throttling"]
        lines.append(f"| **Does Learned Utility add value with Throttling?** | $B_5 - B_4$ | **{d:+.2f} dB** | Marginal gain of learned utility in controlled regime |")
    if "Interaction_Effect" in eff:
        d = eff["Interaction_Effect"]
        lines.append(f"| **Interaction (Super-additivity / Synergy)** | $(B_5 - B_4) - (B_2 - B_1)$ | **{d:+.2f} dB** | Non-linear interaction between model & throttling |")

    lines.extend([
        "",
        "## 3. Core Scientific Conclusion",
        "",
    ])
    if "Throttling_on_ErrorInfluence" in eff and "LearnedUtility_with_Throttling" in eff:
        thr_gain = eff["Throttling_on_ErrorInfluence"]
        learn_gain = eff["LearnedUtility_with_Throttling"]
        if abs(learn_gain) < 0.20 and thr_gain > 1.0:
            lines.append(
                "> [!IMPORTANT]\n"
                f"> **Causal Attribution:** Map growth throttling accounts for the primary performance gain ({thr_gain:+.2f} dB), "
                f"while pointwise learned utility provides marginal increment ({learn_gain:+.2f} dB). "
                "This empirically proves that compute dilution via map bloating was the central bottleneck, "
                "supporting the thesis that pointwise utility alone is insufficient without structural subset conditioning."
            )
        else:
            lines.append(
                "> [!NOTE]\n"
                f"> **Observed Gains:** Throttling effect = {thr_gain:+.2f} dB, Learned utility marginal gain = {learn_gain:+.2f} dB."
            )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factorial Ablation Study")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--n_frames", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budget_ms", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results/factorial_ablation")
    parser.add_argument("--conditions", type=str, nargs="+", default=None)
    args = parser.parse_args()

    run_factorial_experiment(
        scene=args.scene,
        n_frames=args.n_frames,
        seeds=args.seeds,
        budget_ms=args.budget_ms,
        device=args.device,
        out_dir=args.out_dir,
        condition_ids=args.conditions,
    )
