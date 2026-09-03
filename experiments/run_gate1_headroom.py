#!/usr/bin/env python3
"""Gate 1 & Headroom Verification (Points XXI, LXIV–LXVIII).

Validates:
  1. Measurability of Marginal Utility: Var(U*) > 0, long-tailed distribution vs flat.
  2. Prevalence of Negative Utility: U_i^* in Real (counterfactual degradation).
  3. Realized Headroom on Real TUM RGB-D: H = Delta Q(Oracle-TopK) - Delta Q(Random).
  4. Oracle Selection Efficiency (OSE@K) and Absolute Selection Regret (R_K).
  5. Empirical Group Additivity Ratio: R_add(S) = Delta Q(S) / sum Delta Q_i.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation


def load_tum_sequence(data_path: str, n_frames: int = 25, H: int = 120, W: int = 160, device: str = 'cuda'):
    """Load TUM fr1/desk sequence frames scaled to balanced resolution."""
    dataset = TUMDataset(data_path, max_frames=n_frames, camera='freiburg1')
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
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)      # (1, 1, H, W)
        
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


def run_gate1_and_headroom():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== GATE 1 & HEADROOM VERIFICATION [Device: {device}] ===")
    
    # Setup data
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, 'datasets', 'TUM', 'rgbd_dataset_freiburg1_desk')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"TUM dataset not found at {data_path}")
        
    H, W = 120, 160
    n_warmup = 15
    print(f">> Loading TUM fr1/desk sequence ({n_warmup + 1} frames at {W}x{H})...")
    frames, intrinsics = load_tum_sequence(data_path, n_frames=n_warmup + 1, H=H, W=W, device=device)
    
    # Initialize pipeline
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
            'max_new_per_frame': 100,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    
    # Initialize on frame 0
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose']
    )
    print(f">> Initialized map with {pipeline.gaussian_model.num_gaussians} Gaussians.")
    
    # Warmup reconstruction to allow scene geometry & appearance to form
    print(f">> Running {n_warmup} warmup frames...")
    for t in range(1, n_warmup):
        m = pipeline.process_frame(
            rgb=frames[t]['rgb'],
            depth=frames[t]['depth'],
            gt_pose=frames[t]['pose']
        )
    print(f">> Warmup complete. Map now contains {pipeline.gaussian_model.num_gaussians} Gaussians.")
    
    # Evaluate at frame n_warmup
    eval_frame = frames[n_warmup]
    rgb_eval = eval_frame['rgb']
    depth_eval = eval_frame['depth']
    
    # Run Oracle Utility Experiment
    n_samples = 60
    oracle_engine = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=n_samples,
        n_opt_steps=5,
        w_rgb=0.70,
        w_depth=0.30,
        seed=42,
        min_influence_pixels=25
    )
    
    print(f"\n>> Running Oracle Counterfactual Interventions ({n_samples} stratified candidates)...")
    results = oracle_engine.run_oracle_experiment(
        rgb=rgb_eval,
        depth=depth_eval,
        population_type=SamplingPopulation.GEOMETRY_STRATIFIED
    )
    
    visible_results = [r for r in results if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f">> Evaluated {len(visible_results)} visible interventions.")
    
    # --- 1. Gate 1 Analysis: Variance & Negative Utility ---
    u_stars = [r['oracle_utility_joint'] for r in visible_results]
    delta_qs = [r['delta_quality_local'] for r in visible_results]
    delta_psnrs = [r['delta_psnr_local'] for r in visible_results]
    delta_depths = [r['delta_depth_gain_local'] for r in visible_results]
    
    u_var = float(np.var(u_stars))
    u_mean = float(np.mean(u_stars))
    u_std = float(np.std(u_stars))
    u_min = float(np.min(u_stars))
    u_max = float(np.max(u_stars))
    
    n_neg = sum(1 for u in u_stars if u < -1e-5)
    pct_neg = (n_neg / len(u_stars)) * 100.0 if u_stars else 0.0
    
    gate1_passed = (u_std > 1e-5 and len(u_stars) >= 15 and (u_max - u_min) > 1e-4)
    print(f"\n=== GATE 1 METRICS ===")
    print(f"   Variance Var(U*):     {u_var:.8f} (Std: {u_std:.6f}) [{'PASSED' if gate1_passed else 'FAILED'}]")
    print(f"   Range [Min, Max]:    [{u_min:+.4f}, {u_max:+.4f}]")
    print(f"   Mean ± Std:          {u_mean:+.4f} ± {u_std:.4f}")
    print(f"   Negative Utility:    {n_neg}/{len(u_stars)} ({pct_neg:.1f}% of interventions degraded quality)")
    
    # Stratified breakdown
    strata_stats = {}
    for r in visible_results:
        st = r.get('geometry_stratum', 'unknown')
        if st not in strata_stats:
            strata_stats[st] = []
        strata_stats[st].append(r['oracle_utility_joint'])
        
    print(f"\n>> Utility by Geometry Stratum:")
    for st, vals in strata_stats.items():
        print(f"   - {st:<20}: Mean U* = {np.mean(vals):+.4f} | Std = {np.std(vals):.4f} | N = {len(vals)}")
        
    # --- 2. Headroom & Policy Comparison ---
    K = max(1, int(len(visible_results) * 0.20))  # Top 20%
    print(f"\n>> Evaluating Realized Headroom at K = {K} Gaussians...")
    
    # Candidate indices
    cand_indices = [r['gaussian_id'] for r in visible_results]
    u_by_id = {r['gaussian_id']: r['oracle_utility_joint'] for r in visible_results}
    imp_by_id = {r['gaussian_id']: r['predicted_importance'] for r in visible_results}
    err_by_id = {r['gaussian_id']: r['features']['rgb_error'] + r['features']['depth_error'] for r in visible_results}
    
    # Selection sets
    s_oracle = sorted(cand_indices, key=lambda idx: u_by_id[idx], reverse=True)[:K]
    s_heur = sorted(cand_indices, key=lambda idx: imp_by_id[idx], reverse=True)[:K]
    s_err = sorted(cand_indices, key=lambda idx: err_by_id[idx], reverse=True)[:K]
    
    # Evaluate realized joint subset optimization
    def evaluate_subset_gain(subset_indices):
        snap = oracle_engine.snapshot_state()
        try:
            # Full image mask for global realized gain
            full_mask = torch.ones(H, W, dtype=torch.bool, device=device)
            res = oracle_engine.optimize_gaussian_group(
                subset_indices, n_steps=5, rgb=rgb_eval, depth=depth_eval, influence_mask=full_mask
            )
            return res['delta_psnr_global'], res['delta_depth_gain_global'], res['delta_quality_global'], res['measured_trial_cost_ms']
        finally:
            oracle_engine.restore_state(snap)
            
    psnr_ora, depth_ora, q_ora, cost_ora = evaluate_subset_gain(s_oracle)
    psnr_heur, depth_heur, q_heur, cost_heur = evaluate_subset_gain(s_heur)
    psnr_err, depth_err, q_err, cost_err = evaluate_subset_gain(s_err)
    
    # Random baseline (average of 5 random subsets)
    rand_qs = []
    rand_psnrs = []
    rand_depths = []
    np.random.seed(42)
    for _ in range(5):
        perm = np.random.permutation(len(cand_indices))
        s_rand = [cand_indices[p] for p in perm[:K]]
        r_p, r_d, r_q, _ = evaluate_subset_gain(s_rand)
        rand_qs.append(r_q)
        rand_psnrs.append(r_p)
        rand_depths.append(r_d)
        
    q_rand = float(np.mean(rand_qs))
    psnr_rand = float(np.mean(rand_psnrs))
    depth_rand = float(np.mean(rand_depths))
    
    headroom = q_ora - q_rand
    headroom_psnr = psnr_ora - psnr_rand
    
    # Oracle Selection Efficiency & Regret
    ose_heur = float(q_heur / (q_ora + 1e-8)) if q_ora > 0 else 0.0
    ose_err = float(q_err / (q_ora + 1e-8)) if q_ora > 0 else 0.0
    ose_rand = float(q_rand / (q_ora + 1e-8)) if q_ora > 0 else 0.0
    
    regret_heur = q_ora - q_heur
    regret_err = q_ora - q_err
    regret_rand = q_ora - q_rand
    
    print(f"\n=== HEADROOM & SELECTION EFFICIENCY (K = {K}) ===")
    print(f"   Headroom H (Joint ΔQ): {headroom:+.6f}  [{'CONFIRMED ROOM (>0)' if headroom > 0 else 'SATURATED (<=0)'}]")
    print(f"   Headroom ΔPSNR:        {headroom_psnr:+.4f} dB")
    print(f"   Policy Realized Gains:")
    print(f"     - Oracle-Optimal:    ΔQ = {q_ora:+.6f} | ΔPSNR = {psnr_ora:+.4f} dB | OSE = 1.000 | Regret = 0.000")
    print(f"     - Heuristic (Ours):  ΔQ = {q_heur:+.6f} | ΔPSNR = {psnr_heur:+.4f} dB | OSE = {ose_heur:.3f} | Regret = {regret_heur:+.6f}")
    print(f"     - Error-Only:        ΔQ = {q_err:+.6f} | ΔPSNR = {psnr_err:+.4f} dB | OSE = {ose_err:.3f} | Regret = {regret_err:+.6f}")
    print(f"     - Random (Baseline): ΔQ = {q_rand:+.6f} | ΔPSNR = {psnr_rand:+.4f} dB | OSE = {ose_rand:.3f} | Regret = {regret_rand:+.6f}")
    
    # --- 3. Group Additivity Ratio (Points XI & XII) ---
    print(f"\n>> Evaluating Group Additivity R_add(S) for group sizes [4, 16]...")
    group_res = oracle_engine.evaluate_group_interaction(
        rgb=rgb_eval,
        depth=depth_eval,
        candidate_indices=cand_indices[:24],
        group_sizes=[1, 4, 16],
        n_groups_per_size=3
    )
    
    r_add_4 = group_res.get('group_size_4', {}).get('additivity_ratio_mean', float('nan'))
    r_add_16 = group_res.get('group_size_16', {}).get('additivity_ratio_mean', float('nan'))
    print(f"   R_add (g=4):  {r_add_4:.4f}")
    print(f"   R_add (g=16): {r_add_16:.4f}")
    
    # --- 4. Export Artifacts ---
    save_dir = os.path.join(repo_root, 'results', 'gate1_headroom')
    os.makedirs(save_dir, exist_ok=True)
    report_file = os.path.join(save_dir, 'gate1_headroom_report.md')
    json_file = os.path.join(save_dir, 'gate1_summary.json')
    
    summary_data = {
        'gate1_passed': bool(gate1_passed),
        'n_visible_candidates': len(visible_results),
        'utility_variance': u_var,
        'utility_mean': u_mean,
        'utility_std': u_std,
        'negative_utility_fraction': pct_neg,
        'headroom_joint': headroom,
        'headroom_psnr_db': headroom_psnr,
        'selection_efficiency': {
            'oracle': 1.0,
            'heuristic': ose_heur,
            'error_only': ose_err,
            'random': ose_rand,
        },
        'regret': {
            'heuristic': regret_heur,
            'error_only': regret_err,
            'random': regret_rand,
        },
        'group_additivity': {
            'r_add_4': r_add_4,
            'r_add_16': r_add_16,
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
        
    lines = [
        "# Gate 1 & Headroom Verification Report",
        "",
        "## 1. Gate 1: Measurability & Variance of Marginal Utility",
        "",
        f"- **Status:** {'PASSED ✅' if gate1_passed else 'FAILED ❌'}",
        f"- **Sample Size:** {len(visible_results)} visible interventions",
        f"- **Utility Variance:** $\\text{{Var}}(U^\\star) = {u_var:.6f}$",
        f"- **Utility Range:** $[{u_min:+.4f}, {u_max:+.4f}]$ (Mean: ${u_mean:+.4f} \\pm {u_std:.4f}$)",
        f"- **Negative Utility Proportion:** {pct_neg:.1f}% ({n_neg}/{len(u_stars)} candidates degraded quality upon intervention)",
        "",
        "## 2. Optimization Headroom ($H$) & Policy Selection at $K = " + str(K) + "$",
        "",
        f"- **Headroom $H$ (Joint Gain):** **{headroom:+.6f}** ({'Strictly Positive ✅' if headroom > 0 else 'Degenerate ❌'})",
        f"- **Headroom $\\Delta$PSNR:** **{headroom_psnr:+.4f} dB**",
        "",
        "| Policy | Realized $\\Delta Q$ | Realized $\\Delta$PSNR | Oracle Selection Efficiency ($OSE$) ↑ | Selection Regret ($R_K$) ↓ |",
        "|:---|:---:|:---:|:---:|:---:|",
        f"| **Oracle Upper Bound ($S^\\star_K$)** | **{q_ora:+.6f}** | **{psnr_ora:+.4f} dB** | **1.000** | **0.0000** |",
        f"| **Heuristic (Pre-fusion Norm)** | {q_heur:+.6f} | {psnr_heur:+.4f} dB | **{ose_heur:.3f}** | {regret_heur:+.6f} |",
        f"| **Error-Only Top-$K$** | {q_err:+.6f} | {psnr_err:+.4f} dB | {ose_err:.3f} | {regret_err:+.6f} |",
        f"| **Random Baseline** | {q_rand:+.6f} | {psnr_rand:+.4f} dB | {ose_rand:.3f} | {regret_rand:+.6f} |",
        "",
        "## 3. Empirical Group Additivity ($R_{add}$)",
        "",
        f"- **Group Size $g=4$:** $R_{{add}} = {r_add_4:.4f}$",
        f"- **Group Size $g=16$:** $R_{{add}} = {r_add_16:.4f}$",
        f"- *Interpretation:* $R_{{add}} < 1.0$ quantitatively confirms diminishing marginal returns / occlusion overlap in concurrent Gaussian optimization.",
        ""
    ]
    
    with open(report_file, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"\n[Generated Report] Successfully saved to {report_file}")


if __name__ == '__main__':
    run_gate1_and_headroom()
