#!/usr/bin/env python3
"""Phase 6: Budget-Constrained Selection Benchmark (RQ5) — REFORMED.

Fixes 3 Critical bugs from Phase 6 v1:
  C1: Uses actual joint group optimization (evaluate_selected_group) instead of
      summing individual ΔQ values. ΔQ_realized = Q(S_B) - Q(∅).
  C2: Uses real model.positions and real canonical features from live pipeline,
      NOT torch.randn() or np.zeros().
  C3: Uses real attribution rendering (contrib_indices, contrib_weights) for
      context computation.

Compares:
  - NO_OP: S = ∅
  - RANDOM: Random uniform permutation under budget
  - ERROR_ONLY: Rank by photometric + geometric error
  - ERROR_INFLUENCE: Rank by error × attribution mass
  - HEURISTIC: Knapsack heuristic (Importance / Cost)
  - PHASE4_LEARNED: Pointwise TwoHeadMLP U_hat = f(s_i)
  - PHASE6_STATIC: Context model with S = ∅ (static 1-pass)
  - PHASE6_ADAPTIVE (OURS): Adaptive Greedy with dynamic context S_t re-ranking
  - ORACLE_REFERENCE: Ground truth marginal reference

Fairness Contract:
  All policies face the exact same compute budget B and safety factor alpha:
      sum_{i in S_B} (alpha * C_i) <= B

Usage:
    python experiments/run_phase6_selection.py --scene tum_fr2_xyz --seed 42
    python experiments/run_phase6_selection.py --scene tum_fr2_xyz --seeds 42,43,44,45,46
    python experiments/run_phase6_selection.py --quick  # Quick prototype
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional

import numpy as np
from scipy.stats import wilcoxon
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.attribution import render_with_attribution, compute_gaussian_statistics
from research.phase6_context import ContextConfig
from research.phase6_model import FrozenContextPredictor
from research.phase6_evaluator import Phase6Evaluator, evaluate_selected_group
from research.utility_predictor import FrozenUtilityPredictor
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_budget_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Setup (same as Phase 5 authoritative benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def load_tum_sequence(data_path, camera, n_frames, H, W, device):
    """Loads and scales TUM frames strictly according to protocol resolution."""
    dataset = TUMDataset(data_path, max_frames=n_frames, camera=camera)
    frames = []
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H

    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0, dataset.cx * scale_x],
        [0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0, 0, 1.0]
    ], dtype=torch.float32, device=device)

    for i in range(min(n_frames, len(dataset))):
        item = dataset[i]
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)

        rgb_scaled = torch.nn.functional.interpolate(
            rgb, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0)

        frames.append({
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device)
        })

    return frames, intrinsics


def build_pipeline(H, W, device):
    """Build OnlineReconstructionPipeline with standard config."""
    config = {
        'gaussian': {
            'sh_degree': 0,
            'initial_opacity': 0.5,
            'max_gaussians': 30000,
            'initial_scale': 0.02,
        },
        'rendering': {
            'tile_size': 16,
            'image_width': W,
            'image_height': H,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {
            'gpu_budget_ms': 25.0,
            'policy': 'budget_aware',
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }
    return OnlineReconstructionPipeline(config=config, device=device)


def extract_features_and_attribution(pipeline, rgb_gt, depth_gt):
    """Extract real canonical features AND attribution from live pipeline.

    Returns:
        all_features: (N, 11) numpy array — REAL canonical features
        positions: (N, 3) tensor — REAL model positions
        contrib_indices: (H, W, K) tensor — REAL attribution indices
        contrib_weights: (H, W, K) tensor — REAL attribution weights
    """
    model = pipeline.gaussian_model
    N = model.num_gaussians
    H, W = rgb_gt.shape[:2]

    if N == 0:
        return (
            np.zeros((0, 11), dtype=np.float32),
            torch.zeros(0, 3, device=pipeline.device),
            torch.zeros(H, W, 4, dtype=torch.long, device=pipeline.device),
            torch.zeros(H, W, 4, device=pipeline.device),
        )

    # Real attribution rendering (C3 FIX)
    attr_out = render_with_attribution(
        means3D=model.positions,
        cov3D=model.build_covariance(),
        colors=model.get_colors(),
        opacities=model.opacities.squeeze(-1),
        extrinsics=pipeline.current_pose,
        intrinsics=pipeline.intrinsics,
        image_width=W,
        image_height=H,
        tile_size=pipeline.config.get('rendering', {}).get('tile_size', 16),
        top_k=pipeline.config.get('rendering', {}).get('attribution_top_k', 4),
    )

    contrib_indices = attr_out['contrib_indices']
    contrib_weights = attr_out['contrib_weights']

    # Compute per-Gaussian statistics
    stats = compute_gaussian_statistics(
        rendered_color=attr_out['color'],
        rendered_depth=attr_out['depth'],
        gt_color=rgb_gt,
        gt_depth=depth_gt,
        contrib_weights=contrib_weights,
        contrib_indices=contrib_indices,
        n_gaussians=N,
    )

    # Build canonical 11-dim features
    store = getattr(model, 'state_store', None)
    features = np.zeros((N, 11), dtype=np.float32)
    for i in range(N):
        features[i, 0] = float(stats['color_error'][i])
        features[i, 1] = float(stats['depth_error'][i])
        features[i, 2] = float(
            (stats['color_error'][i] + stats['depth_error'][i]) * stats['influence_mass'][i]
        )
        features[i, 3] = float(stats['pixel_count'][i])
        features[i, 4] = float(stats['influence_mass'][i])

        if store is not None and i < len(store.position_drift):
            features[i, 5] = float(store.position_drift[i].item())
            features[i, 6] = float(store.residual_drift_ema[i].item())
            features[i, 7] = float(store.uncertainty[i].item())
            age_val = max(1, int(store.ages[i].item()))
            features[i, 9] = float(store.update_counts[i].item()) / age_val
            features[i, 10] = float(store.ages[i].item())

        features[i, 8] = float(stats['projected_area'][i])

    # Real positions (C2 FIX)
    positions = model.positions.detach()

    return features, positions, contrib_indices, contrib_weights


def build_candidates_from_oracle(
    oracle_engine, candidate_indices, rgb_gt, depth_gt,
    contrib_indices, contrib_weights, all_features,
):
    """Build candidate dicts with REAL oracle measurements.

    Each candidate gets actual measured ΔQ and cost from single-Gaussian
    oracle optimization (for ranking). The EVALUATION uses joint group
    optimization separately via evaluate_selected_group.
    """
    candidates = []
    for idx in candidate_indices:
        idx = int(idx)

        # Get influence mask for this single Gaussian
        influence_mask = oracle_engine._get_influence_mask(
            [idx], contrib_indices, contrib_weights,
        )

        # Measure single-Gaussian oracle utility
        snap = oracle_engine.snapshot_state()
        try:
            opt_res = oracle_engine.optimize_gaussian_group(
                indices=[idx],
                n_steps=5,
                rgb=rgb_gt,
                depth=depth_gt,
                influence_mask=influence_mask,
            )
        finally:
            oracle_engine.restore_state(snap)

        delta_q = float(opt_res["delta_quality_global"])
        cost_ms = float(opt_res["measured_trial_cost_ms"])
        oracle_utility = delta_q / max(0.001, cost_ms)

        # Build candidate dict with real features
        feats = all_features[idx] if idx < len(all_features) else np.zeros(11, dtype=np.float32)

        cand = {
            "gaussian_id": idx,
            "frame": oracle_engine.pipeline.frame_count if hasattr(oracle_engine.pipeline, 'frame_count') else 0,
            "features": {
                "rgb_error": float(feats[0]),
                "depth_error": float(feats[1]),
                "gradient_norm": float(feats[2]),
                "visibility_count": float(feats[3]),
                "influence_mass": float(feats[4]),
            },
            "predicted_importance": float(feats[0] + feats[1]),
            "predicted_utility": oracle_utility,
            "predicted_delta_q": delta_q,
            "predicted_delta_t": cost_ms,
            "measured_trial_cost_ms": cost_ms,
            "delta_quality_global": delta_q,
            "oracle_utility_joint_global": oracle_utility,
        }
        candidates.append(cand)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Core Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_budget_sweep(
    evaluator: Phase6Evaluator,
    oracle_engine: OracleUtilityExperiment,
    candidates: List[Dict[str, Any]],
    positions: torch.Tensor,
    all_features: np.ndarray,
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    rgb_gt: torch.Tensor,
    depth_gt: torch.Tensor,
    budgets: List[float],
    budget_type: str,
    current_frame: int,
    seed: int = 42,
    reject_negative: bool = False,
    oracle_reference_gain: Optional[float] = None,
    policies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluates all policies across budget levels with ACTUAL joint optimization."""
    if policies is None:
        policies = [
            "no_op", "random", "error_only", "error_influence",
            "heuristic", "phase4_learned",
            "phase6_static", "phase6_adaptive",
            "oracle_reference",
        ]

    results = []
    for b in budgets:
        for pol in policies:
            do_reject = reject_negative if pol in (
                "phase4_learned", "phase6_static", "phase6_adaptive",
                "oracle_reference",
            ) else False

            pct_str = f"{b:.1f}ms" if budget_type == "wall_clock" else f"{b:.1f}"

            res = evaluator.evaluate_policy(
                policy=pol,
                candidates=candidates,
                budget=b,
                current_frame=current_frame,
                oracle_engine=oracle_engine,
                rgb_gt=rgb_gt,
                depth_gt=depth_gt,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
                positions=positions,
                all_features=all_features,
                oracle_reference_gain=oracle_reference_gain,
                seed=seed,
                reject_negative=do_reject,
                budget_type=budget_type,
                budget_pct_str=pct_str,
            )
            results.append(res)

            print(f"    {pol:20s} | B={b:8.2f} | k={res['k_count']:3d} | "
                  f"ΔQ={res['actual_delta_q']:.2e} | "
                  f"sum_ind={res['sum_individual_dq']:.2e} | "
                  f"gap={res['non_additivity_gap']:.2e} | "
                  f"T={res['actual_cost_ms']:.1f}ms")

    return results


def compute_paired_statistics(
    results: List[Dict[str, Any]],
    baseline_policy: str = "phase4_learned",
    target_policy: str = "phase6_adaptive",
) -> Dict[str, Any]:
    """Runs paired Wilcoxon signed-rank tests.

    NOTE: For proper statistical testing, this should be called with
    results from MULTIPLE SEEDS at each budget level, not multiple
    budget points from a single seed.
    """
    budgets = sorted(set(r["budget_val"] for r in results))
    dq_base = []
    dq_target = []

    for b in budgets:
        r_b = next((r for r in results
                     if abs(r["budget_val"] - b) < 1e-4
                     and r["policy"] == baseline_policy), None)
        r_t = next((r for r in results
                     if abs(r["budget_val"] - b) < 1e-4
                     and r["policy"] == target_policy), None)
        if r_b and r_t:
            dq_base.append(r_b["actual_delta_q"])
            dq_target.append(r_t["actual_delta_q"])

    diffs = np.array(dq_target) - np.array(dq_base)
    n = len(diffs)

    if n >= 4 and not np.all(diffs == 0):
        try:
            stat, p_val = wilcoxon(diffs, alternative="greater")
            stat_val, p_val_val = float(stat), float(p_val)
        except Exception:
            stat_val, p_val_val = 0.0, 1.0
    else:
        stat_val, p_val_val = 0.0, 1.0

    mean_diff = float(np.mean(diffs)) if n > 0 else 0.0
    win_rate = float(np.mean(diffs > 0)) if n > 0 else 0.0

    return {
        "n_observations": n,
        "baseline_policy": baseline_policy,
        "target_policy": target_policy,
        "mean_difference": mean_diff,
        "win_rate": win_rate,
        "wilcoxon_stat": stat_val,
        "wilcoxon_pval": p_val_val,
        "statistically_significant": bool(p_val_val < 0.05),
        "note": "Each observation is one budget level from same seed. "
                "For proper stats, use multi-seed runs (5 seeds per budget).",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6 RQ5 Budget Selection Benchmark (REFORMED)"
    )
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed run (e.g. 42,43,44,45,46)")
    parser.add_argument("--frame", type=int, default=20,
                        help="Frame index to evaluate at (default: 20)")
    parser.add_argument("--warmup-frames", type=int, default=None,
                        help="Number of frames to warm up pipeline (default: --frame)")
    parser.add_argument("--n-candidates", type=int, default=25,
                        help="Number of candidate Gaussians to evaluate (default: 25)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reject-negative", action="store_true", default=False)
    parser.add_argument("--quick", action="store_true", default=False,
                        help="Quick mode: fewer budgets, fewer candidates")
    parser.add_argument("--p6-checkpoint", type=str, default=None)
    parser.add_argument("--p6-normalization", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(
        repo_root, "results", "phase6_context_utility", "selection"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load protocol
    protocol = load_protocol()
    H, W = get_resolution(protocol, args.scene)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    warmup_frames = args.warmup_frames if args.warmup_frames is not None else args.frame
    n_frames_to_load = warmup_frames + 5  # extra buffer

    print("=" * 80)
    print("  PHASE 6: RQ5 BUDGET-CONSTRAINED SELECTION BENCHMARK (REFORMED)")
    print("=" * 80)
    print(f"  Scene: {args.scene}")
    print(f"  Seeds: {seeds}")
    print(f"  Eval Frame: {args.frame}")
    print(f"  Warmup Frames: {warmup_frames}")
    print(f"  Candidates: {args.n_candidates}")
    print(f"  Device: {device}")
    print(f"  Reject Negative: {args.reject_negative}")
    print()
    print("  CRITICAL FIXES APPLIED:")
    print("    C1: Actual joint group optimization (NOT sum of individual ΔQ)")
    print("    C2: Real model.positions (NOT torch.randn)")
    print("    C3: Real attribution (NOT bypassed)")
    print()

    # Load Phase 6 predictor
    p6_ckpt = args.p6_checkpoint or os.path.join(
        repo_root, "results", "phase6_context_utility", "checkpoints",
        f"context_mlp_V11_seed_{seeds[0]}.pt"
    )
    p6_norm = args.p6_normalization or os.path.join(
        repo_root, "results", "phase6_context_utility", "normalization_V11.json"
    )

    p6_predictor = None
    if os.path.exists(p6_ckpt) and os.path.exists(p6_norm):
        p6_predictor = FrozenContextPredictor(p6_ckpt, p6_norm, device=device)
        print(f"  Phase 6 predictor: {p6_ckpt}")
    else:
        print(f"  [WARN] Phase 6 predictor not found, skipping phase6_* policies")

    # Determine policies
    all_policies = [
        "no_op", "random", "error_only", "error_influence",
        "heuristic", "phase4_learned", "oracle_reference",
    ]
    if p6_predictor is not None:
        all_policies.extend(["phase6_static", "phase6_adaptive"])

    # Load scene data
    scene_cfg = get_dataset_config(protocol, args.scene)
    data_path = scene_cfg['path']
    camera = scene_cfg.get('camera', 'freiburg1')

    print(f"\n[1/4] Loading {args.scene} ({n_frames_to_load} frames)...")
    frames, intrinsics = load_tum_sequence(
        data_path, camera, n_frames_to_load, H, W, device,
    )

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"  SEED {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*80}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build fresh pipeline for each seed
        print(f"[2/4] Building pipeline and warming up to frame {warmup_frames}...")
        pipeline = build_pipeline(H, W, device)
        pipeline.initialize(frames[0])

        for fi in range(1, min(warmup_frames, len(frames))):
            pipeline.process_frame(frames[fi])

        N = pipeline.gaussian_model.num_gaussians
        print(f"  Active Gaussians: N = {N}")

        # Extract REAL features and attribution (C2 + C3 FIX)
        eval_frame = frames[min(args.frame, len(frames) - 1)]
        rgb_gt = eval_frame['rgb']
        depth_gt = eval_frame['depth']

        print(f"[3/4] Extracting real features and attribution...")
        t0 = time.perf_counter()
        all_features, positions, contrib_indices, contrib_weights = \
            extract_features_and_attribution(pipeline, rgb_gt, depth_gt)
        t_feat_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  Features: ({all_features.shape[0]}, {all_features.shape[1]})")
        print(f"  Positions: {positions.shape}")
        print(f"  Attribution: contrib_indices={contrib_indices.shape}, "
              f"contrib_weights={contrib_weights.shape}")
        print(f"  Feature extraction time: {t_feat_ms:.1f}ms")

        # Sample candidate Gaussians
        n_cand = min(args.n_candidates, N)
        if args.quick:
            n_cand = min(10, N)

        rng = np.random.default_rng(seed)
        candidate_indices = rng.choice(N, size=n_cand, replace=False).tolist()

        # Build candidates with real oracle measurements
        print(f"  Building {n_cand} candidates with real oracle measurements...")
        oracle_engine = OracleUtilityExperiment(
            pipeline=pipeline,
            n_opt_steps=5,
            seed=seed,
        )

        candidates = build_candidates_from_oracle(
            oracle_engine, candidate_indices, rgb_gt, depth_gt,
            contrib_indices, contrib_weights, all_features,
        )

        current_frame = pipeline.frame_count if hasattr(pipeline, 'frame_count') else args.frame
        # Set frame on all candidates
        for c in candidates:
            c["frame"] = current_frame

        print(f"  Candidates built: {len(candidates)}")

        # Compute oracle reference gain (at max budget)
        total_cost = sum(c["measured_trial_cost_ms"] for c in candidates)
        print(f"  Total candidate pool cost: {total_cost:.2f} ms")

        # Oracle reference: optimize ALL candidates jointly
        print(f"  Computing oracle reference gain (full pool)...")
        oracle_ref_res = evaluate_selected_group(
            oracle_engine=oracle_engine,
            selected_gaussian_ids=candidate_indices,
            rgb_gt=rgb_gt,
            depth_gt=depth_gt,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
        )
        oracle_ref_gain = oracle_ref_res["delta_q_realized"]
        print(f"  Oracle reference ΔQ (full pool): {oracle_ref_gain:.6e}")

        # Create evaluator
        evaluator = Phase6Evaluator(
            p6_predictor=p6_predictor,
            safety_factor=1.10,
            use_predicted_cost=True,
            device=device,
        )

        # ─── Experiment A: Relative Budget Sweep ───
        rel_fractions = [0.10, 0.20, 0.40, 0.60, 0.80]
        if args.quick:
            rel_fractions = [0.20, 0.50, 0.80]
        rel_budgets = [f * total_cost for f in rel_fractions]

        print(f"\n[4/4] Running Relative Budget Sweep: {rel_fractions}")
        rel_results = run_budget_sweep(
            evaluator=evaluator,
            oracle_engine=oracle_engine,
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            rgb_gt=rgb_gt,
            depth_gt=depth_gt,
            budgets=rel_budgets,
            budget_type="relative",
            current_frame=current_frame,
            seed=seed,
            reject_negative=args.reject_negative,
            oracle_reference_gain=oracle_ref_gain,
            policies=all_policies,
        )

        # ─── Experiment B: Wall-Clock Budget Sweep ───
        wall_budgets = [10.0, 15.0, 20.0, 33.3]
        if args.quick:
            wall_budgets = [10.0, 20.0]

        print(f"\n  Wall-Clock Budget Sweep: {wall_budgets} ms")
        wall_results = run_budget_sweep(
            evaluator=evaluator,
            oracle_engine=oracle_engine,
            candidates=candidates,
            positions=positions,
            all_features=all_features,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            rgb_gt=rgb_gt,
            depth_gt=depth_gt,
            budgets=wall_budgets,
            budget_type="wall_clock",
            current_frame=current_frame,
            seed=seed,
            reject_negative=args.reject_negative,
            oracle_reference_gain=oracle_ref_gain,
            policies=all_policies,
        )

        # ─── Statistical Tests ───
        test_vs_p4 = compute_paired_statistics(
            rel_results, "phase4_learned", "phase6_adaptive",
        )
        test_vs_heur = compute_paired_statistics(
            rel_results, "heuristic", "phase6_adaptive",
        )

        # ─── Non-Additivity Analysis ───
        non_add_data = []
        for r in rel_results + wall_results:
            if r["k_count"] > 1:
                non_add_data.append({
                    "policy": r["policy"],
                    "k_count": r["k_count"],
                    "actual_delta_q": r["actual_delta_q"],
                    "sum_individual_dq": r["sum_individual_dq"],
                    "non_additivity_gap": r["non_additivity_gap"],
                    "gap_relative": abs(r["non_additivity_gap"]) / max(abs(r["sum_individual_dq"]), 1e-10),
                })

        # ─── Print Summary ───
        print(f"\n{'='*85}")
        print(f"  RELATIVE BUDGET SWEEP — SEED {seed}")
        print(f"  ΔQ = ACTUAL JOINT GROUP GAIN (C1 FIX)")
        print(f"{'='*85}")

        pols_show = [p for p in all_policies if p != "no_op"]
        header = f"{'Budget':>10} | " + " | ".join(f"{p[:12]:>12}" for p in pols_show)
        print(header)
        print("-" * len(header))

        for idx, f in enumerate(rel_fractions):
            b = rel_budgets[idx]
            vals = []
            for p in pols_show:
                match = next((r for r in rel_results
                              if abs(r["budget_val"] - b) < 1e-4
                              and r["policy"] == p), None)
                if match:
                    vals.append(f"{match['actual_delta_q']*1e5:>12.3f}")
                else:
                    vals.append(f"{'N/A':>12}")
            print(f"{f*100:8.0f}%  | " + " | ".join(vals))

        print(f"\n  Non-Additivity Check (k>1 groups):")
        if non_add_data:
            for nad in non_add_data[:10]:
                print(f"    {nad['policy']:20s} k={nad['k_count']:2d} | "
                      f"joint={nad['actual_delta_q']:.2e} | "
                      f"sum_ind={nad['sum_individual_dq']:.2e} | "
                      f"gap={nad['non_additivity_gap']:.2e} | "
                      f"gap_rel={nad['gap_relative']:.1%}")
        else:
            print("    No multi-Gaussian groups selected.")

        if "phase6_adaptive" in all_policies:
            print(f"\n  Gate 6C (Decision):")
            gate_pass = test_vs_p4["statistically_significant"] or (
                test_vs_p4["win_rate"] >= 0.5 and test_vs_p4["mean_difference"] >= 0.0
            )
            print(f"    Status: {'✓ PASS' if gate_pass else '✗ FAIL'}")
            print(f"    P6 vs P4: win={test_vs_p4['win_rate']*100:.0f}%, "
                  f"mean_diff={test_vs_p4['mean_difference']:.2e}, "
                  f"p={test_vs_p4['wilcoxon_pval']:.4f}")
            print(f"    P6 vs Heur: win={test_vs_heur['win_rate']*100:.0f}%, "
                  f"mean_diff={test_vs_heur['mean_difference']:.2e}")

        # Collect per-seed results
        seed_artifact = {
            "seed": seed,
            "scene": args.scene,
            "frame": args.frame,
            "n_gaussians": N,
            "n_candidates": len(candidates),
            "oracle_reference_gain": oracle_ref_gain,
            "relative_sweep": rel_results,
            "wall_clock_sweep": wall_results,
            "statistical_tests": {
                "vs_phase4_learned": test_vs_p4,
                "vs_heuristic": test_vs_heur,
            },
            "non_additivity_analysis": non_add_data,
            "critical_fixes": ["C1_joint_optimization", "C2_real_positions", "C3_real_attribution"],
        }
        all_seed_results.append(seed_artifact)

        # Clean up GPU memory
        del pipeline, oracle_engine
        if device == "cuda":
            torch.cuda.empty_cache()

    # ─── Save Combined Artifacts ───
    combined = {
        "benchmark": "Phase 6 RQ5 Budget-Constrained Selection (REFORMED)",
        "seeds": seeds,
        "scene": args.scene,
        "critical_fixes": [
            "C1: actual joint group optimization via evaluate_selected_group",
            "C2: real model.positions from live pipeline",
            "C3: real attribution via render_with_attribution",
        ],
        "per_seed_results": all_seed_results,
    }

    out_file = os.path.join(output_dir, f"selection_benchmark_reformed_{args.scene}.json")
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n[Saved] {out_file}")


if __name__ == "__main__":
    main()
