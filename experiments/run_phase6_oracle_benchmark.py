#!/usr/bin/env python3
"""Phase 6 — Oracle Conditional Greedy Benchmark & Failure Mode Decomposition.

Answers the fundamental question of Phase 6:
  "Is context-aware greedy selection theoretically superior to pointwise selection?"

Evaluates 5 benchmark policies under identical compute budget B with actual joint group optimization:
  1. Static heuristic:      ranking=heuristic(e·m), context=none,     selection=greedy
  2. Phase 4 learned:       ranking=learned f(s_i), context=none,     selection=greedy
  3. Oracle static:         ranking=U*(i|∅),        context=none,     selection=greedy
  4. Oracle conditional:    ranking=U*(i|S_t),       context=dynamic,  selection=greedy
  5. Phase 6 learned:       ranking=f(s_i,N,O,S_t),  context=dynamic,  selection=greedy

Failure Mode Diagnosis:
  - Case A (OracleConditional >> P4, P6 ≈ P4):  Model prediction failure (model needs fix).
  - Case B (OracleConditional ≈ P4):            Hypothesis failure (context-aware greedy itself has no benefit).
  - Case C (OracleConditional > P4, P6 > P4):   Hypothesis confirmed.

Usage:
    python experiments/run_phase6_oracle_benchmark.py --tiny
    python experiments/run_phase6_oracle_benchmark.py --scene tum_fr2_xyz --frame 20 --candidates 20
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.phase6_oracle import ConditionalOracleExperiment, ConditionalOracleConfig
from research.phase6_selection import (
    select_phase6_subset,
    Phase6PolicyName,
    map_candidate_to_active_index,
)
from research.phase6_evaluator import Phase6Evaluator, evaluate_selected_group
from research.phase6_model import FrozenContextPredictor
from research.utility_models import TwoHeadMLP
from research.utility_dataset import FeatureNormalizer
from research.attribution import render_with_attribution, compute_gaussian_statistics
from research.protocol import (
    load_protocol,
    get_resolution,
    get_dataset_config,
    get_oracle_config,
)


def load_sequence(data_path, camera, n_frames, H, W, device):
    """Load and resize TUM RGB-D sequence."""
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
    config = {
        'gaussian': {
            'sh_degree': 0, 'initial_opacity': 0.5,
            'max_gaussians': 30000, 'initial_scale': 0.02,
        },
        'rendering': {
            'tile_size': 16, 'image_width': W, 'image_height': H,
            'use_surface_aware_depth': True, 'attribution_top_k': 4,
        },
        'scheduler': {'gpu_budget_ms': 25.0, 'policy': 'budget_aware'},
        'densification': {
            'max_new_per_frame': 80, 'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }
    return OnlineReconstructionPipeline(config=config, device=device)


class FrozenUtilityPredictorWrapper:
    """Lightweight wrapper for Phase 4 TwoHeadMLP."""
    def __init__(self, checkpoint_path: str, normalizer_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.normalizer = FeatureNormalizer.load_json(normalizer_path)
        self.model = TwoHeadMLP(in_features=11).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state", ckpt))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict_features(self, X_raw: np.ndarray) -> Dict[str, np.ndarray]:
        X_norm = self.normalizer.transform(X_raw)
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dq, dt, u = self.model(X_t)
        return {
            "predicted_delta_q": dq.cpu().numpy(),
            "predicted_delta_t": dt.cpu().numpy(),
            "predicted_utility": u.cpu().numpy(),
        }


def extract_features(model, pipeline, rgb, depth, H, W):
    """Extract canonical 11-dim features."""
    N = model.num_gaussians
    attr_out = render_with_attribution(
        means3D=model.positions,
        cov3D=model.build_covariance(),
        colors=model.get_colors(),
        opacities=model.opacities.squeeze(-1),
        extrinsics=pipeline.current_pose,
        intrinsics=pipeline.intrinsics,
        image_width=W, image_height=H,
        tile_size=16, top_k=4,
    )
    stats = compute_gaussian_statistics(
        rendered_color=attr_out['color'],
        rendered_depth=attr_out['depth'],
        gt_color=rgb, gt_depth=depth,
        contrib_weights=attr_out['contrib_weights'],
        contrib_indices=attr_out['contrib_indices'],
        n_gaussians=N,
    )
    store = getattr(model, 'state_store', None)
    feats = np.zeros((N, 11), dtype=np.float32)
    for i in range(N):
        feats[i, 0] = float(stats['color_error'][i].detach().cpu())
        feats[i, 1] = float(stats['depth_error'][i].detach().cpu())
        feats[i, 2] = float(((stats['color_error'][i] + stats['depth_error'][i]) * stats['influence_mass'][i]).detach().cpu())
        feats[i, 3] = float(stats['pixel_count'][i].detach().cpu())
        feats[i, 4] = float(stats['influence_mass'][i].detach().cpu())
        if store is not None and i < len(store.position_drift):
            feats[i, 5] = float(store.position_drift[i].item())
            feats[i, 6] = float(store.residual_drift_ema[i].item())
            feats[i, 7] = float(store.uncertainty[i].item())
            age = max(1, int(store.ages[i].item()))
            feats[i, 9] = float(store.update_counts[i].item()) / age
            feats[i, 10] = float(store.ages[i].item())
        feats[i, 8] = float(stats['projected_area'][i].detach().cpu())

    return feats, attr_out


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Oracle Conditional Greedy Benchmark")
    parser.add_argument("--tiny", action="store_true", help="Tiny verification mode")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.seed
    n_candidates = 10 if args.tiny else args.candidates
    eval_frame = 10 if args.tiny else args.frame
    scene_name = args.scene

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(
        repo_root, "results", "phase6_context_utility", "oracle_benchmark"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("  PHASE 6 — ORACLE CONDITIONAL GREEDY BENCHMARK & DECOMPOSITION")
    print("=" * 80)
    print(f"  Scene: {scene_name}, Frame: {eval_frame}, Candidates: {n_candidates}, Seed: {seed}")
    print(f"  Device: {device}")
    print()

    # Load protocol & sequence
    protocol = load_protocol()
    oracle_cfg = get_oracle_config(protocol)
    H, W = get_resolution(scene_name, protocol)
    camera = "freiburg2" if "fr2" in scene_name else "freiburg1"
    ds_cfg = get_dataset_config(scene_name, protocol)
    max_frame = eval_frame + 1

    frames, intrinsics = load_sequence(ds_cfg["full_path"], camera, max_frame, H, W, device)

    # Build and warm up pipeline
    torch.manual_seed(seed)
    np.random.seed(seed)
    pipeline = build_pipeline(H, W, device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])

    print(f"[Pipeline] Running {max_frame-1} frames...")
    for t in range(1, max_frame):
        pipeline.process_frame(frames[t]['rgb'], frames[t]['depth'], frames[t]['pose'])

    rgb_gt = frames[eval_frame]['rgb']
    depth_gt = frames[eval_frame]['depth']
    model = pipeline.gaussian_model
    N = model.num_gaussians
    print(f"  Model has {N} Gaussians at frame {eval_frame}")

    # Extract features & attribution
    all_features, attr_out = extract_features(model, pipeline, rgb_gt, depth_gt, H, W)
    contrib_indices = attr_out['contrib_indices']
    contrib_weights = attr_out['contrib_weights']

    # Sample visible candidates
    valid_attr = contrib_weights > 0.01
    visible = contrib_indices[valid_attr].unique().tolist() if valid_attr.any() else list(range(N))
    visible = [int(i) for i in visible if 0 <= int(i) < N]
    rng = np.random.default_rng(seed)
    cand_ids = rng.choice(visible, size=min(n_candidates, len(visible)), replace=False).tolist()

    # Setup Oracle & Evaluator
    cond_oracle = ConditionalOracleExperiment(
        pipeline=pipeline,
        config=ConditionalOracleConfig(n_opt_steps=oracle_cfg['n_opt_steps']),
        oracle_config={'n_opt_steps': oracle_cfg['n_opt_steps'], 'seed': seed},
    )

    # Measure single-Gaussian oracle gains for each candidate
    print(f"\n[Oracle] Measuring single-Gaussian baselines for {len(cand_ids)} candidates...")
    cand_records = []
    for cid in cand_ids:
        m = cond_oracle.measure_conditional_utility(
            candidate_idx=cid,
            context_indices=[],
            rgb_gt=rgb_gt, depth_gt=depth_gt,
            contrib_indices=contrib_indices, contrib_weights=contrib_weights,
        )
        cand_records.append({
            "candidate_id": cid,
            "gaussian_id": cid,
            "frame": eval_frame,
            "measured_trial_cost_ms": max(m["delta_t_conditional_ms"], 1.0),
            "predicted_delta_t": max(m["delta_t_conditional_ms"], 1.0),
            "oracle_delta_q": m["delta_q_conditional"],
            "oracle_delta_t": m["delta_t_conditional_ms"],
            "oracle_utility_joint_global": m["utility_conditional"],
            "rgb_error": float(all_features[cid, 0]),
            "depth_error": float(all_features[cid, 1]),
            "influence_mass": float(all_features[cid, 4]),
            "predicted_importance": float((all_features[cid, 0] + all_features[cid, 1]) * all_features[cid, 4]),
        })

    # Load Phase 4 predictor if available
    p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", f"two_head_mlp_seed_{seed}.pt")
    p4_norm = os.path.join(repo_root, "results", "learned_utility", "feature_normalizer.json")
    p4_predictor = None
    if os.path.exists(p4_ckpt) and os.path.exists(p4_norm):
        p4_predictor = FrozenUtilityPredictorWrapper(p4_ckpt, p4_norm, device=device)
        p4_preds = p4_predictor.predict_features(all_features[cand_ids])
        for idx, c in enumerate(cand_records):
            c["predicted_utility"] = float(p4_preds["predicted_utility"][idx])
            c["predicted_delta_q"] = float(p4_preds["predicted_delta_q"][idx])
    else:
        for c in cand_records:
            c["predicted_utility"] = c["predicted_importance"]

    # Load Phase 6 predictor if available
    p6_ckpt = os.path.join(repo_root, "results", "phase6_context_utility", "checkpoints", f"context_mlp_V11_seed_{seed}.pt")
    p6_norm = os.path.join(repo_root, "results", "phase6_context_utility", f"normalization_V11.json")
    p6_predictor = None
    if os.path.exists(p6_ckpt) and os.path.exists(p6_norm):
        p6_predictor = FrozenContextPredictor(p6_ckpt, p6_norm, device=device)

    # Initialize Phase 6 Evaluator
    evaluator = Phase6Evaluator(
        p6_predictor=p6_predictor,
        safety_factor=1.10,
        use_predicted_cost=False,
        device=device,
    )

    # Budget levels
    total_cost = sum(c["measured_trial_cost_ms"] for c in cand_records)
    budgets = [0.20, 0.40, 0.60] if not args.tiny else [0.30, 0.60]

    policies = [
        ("Static Heuristic", "error_influence"),
        ("Phase 4 Learned", "phase4_learned"),
        ("Oracle Static", "oracle_reference"),
        ("Oracle Conditional", "oracle_conditional"),
    ]
    if p6_predictor is not None:
        policies.append(("Phase 6 Adaptive", "phase6_adaptive"))

    print(f"\n[Benchmark] Running 5-Policy Decomposition across budgets {budgets}...")
    benchmark_results = {}

    for b_frac in budgets:
        b_val = total_cost * b_frac
        b_str = f"{int(b_frac*100)}%"
        print(f"\n{'─'*60}")
        print(f"  BUDGET: {b_str} ({b_val:.1f} ms)")
        print(f"{'─'*60}")

        benchmark_results[b_str] = {}

        for pol_label, pol_key in policies:
            t_pol_start = time.perf_counter()
            res = evaluator.evaluate_policy(
                policy=pol_key,
                candidates=cand_records,
                budget=b_val,
                current_frame=eval_frame,
                oracle_engine=cond_oracle.oracle_engine,
                rgb_gt=rgb_gt,
                depth_gt=depth_gt,
                contrib_indices=contrib_indices,
                contrib_weights=contrib_weights,
                positions=model.positions,
                all_features=all_features,
                seed=seed,
                budget_type="relative",
                budget_pct_str=b_str,
            )
            t_pol = time.perf_counter() - t_pol_start

            dq_realized = res.get("actual_delta_q", res.get("delta_q_realized", 0.0))
            n_sel = res.get("k_count", res.get("n_selected", 0))
            print(f"  {pol_label:<22s} | K={n_sel:2d} | ΔQ_realized={dq_realized*1e5:6.2f}e-5 | {t_pol:.1f}s")

            benchmark_results[b_str][pol_label] = {
                "policy_key": pol_key,
                "delta_q_realized": float(dq_realized),
                "n_selected": int(n_sel),
                "scheduled_cost_ms": float(res["scheduled_cost_ms"]),
                "actual_cost_ms": float(res["actual_cost_ms"]),
                "budget_violation_ms": float(res["budget_violation_ms"]),
                "evaluation_time_s": float(t_pol),
            }

    # Decomposition Analysis
    print("\n" + "=" * 80)
    print("  DECOMPOSITION ANALYSIS & FAILURE MODE DIAGNOSIS")
    print("=" * 80)

    diagnoses = {}
    for b_str in benchmark_results:
        b_data = benchmark_results[b_str]
        q_p4 = b_data.get("Phase 4 Learned", {}).get("delta_q_realized", 0.0)
        q_cond_oracle = b_data.get("Oracle Conditional", {}).get("delta_q_realized", 0.0)
        q_static_oracle = b_data.get("Oracle Static", {}).get("delta_q_realized", 0.0)
        q_p6 = b_data.get("Phase 6 Adaptive", {}).get("delta_q_realized", q_p4)

        oracle_conditional_lift = q_cond_oracle - q_p4
        static_oracle_lift = q_static_oracle - q_p4
        context_advantage = q_cond_oracle - q_static_oracle

        # Case determination
        if oracle_conditional_lift > 1e-6 and abs(q_p6 - q_p4) < 0.5 * oracle_conditional_lift:
            diagnosis = "Case A: Model Prediction Failure (OracleConditional >> P4, but P6 ≈ P4)"
        elif oracle_conditional_lift <= 1e-6:
            diagnosis = "Case B: Hypothesis Failure (OracleConditional ≈ P4; context-aware greedy has no benefit)"
        else:
            diagnosis = "Case C: Hypothesis Confirmed (OracleConditional > P4 and P6 > P4)"

        diagnoses[b_str] = {
            "q_p4": q_p4,
            "q_static_oracle": q_static_oracle,
            "q_cond_oracle": q_cond_oracle,
            "q_p6": q_p6,
            "oracle_conditional_lift": oracle_conditional_lift,
            "context_advantage": context_advantage,
            "diagnosis": diagnosis,
        }

        print(f"\nBudget {b_str}:")
        print(f"  P4 Learned:         ΔQ = {q_p4*1e5:6.2f}e-5")
        print(f"  Oracle Static:      ΔQ = {q_static_oracle*1e5:6.2f}e-5")
        print(f"  Oracle Conditional: ΔQ = {q_cond_oracle*1e5:6.2f}e-5")
        if "Phase 6 Adaptive" in b_data:
            print(f"  Phase 6 Adaptive:   ΔQ = {q_p6*1e5:6.2f}e-5")
        print(f"  Context Advantage (OracleCond - OracleStatic): {context_advantage*1e5:+.2f}e-5")
        print(f"  Diagnosis: {diagnosis}")

    # Save artifact
    output_path = os.path.join(output_dir, f"oracle_benchmark_decomposition_seed_{seed}.json")
    output_data = {
        "scene": scene_name,
        "frame": eval_frame,
        "seed": seed,
        "n_candidates": n_candidates,
        "benchmark_results": benchmark_results,
        "diagnoses": diagnoses,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved] Artifact saved to: {output_path}")


if __name__ == "__main__":
    main()
