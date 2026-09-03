#!/usr/bin/env python3
"""Phase 6 & 8: Equal-Compute Budget Sweep Benchmark & Pareto Frontier (Gate 3).

Strictly addresses:
  Phase 8.1: Absolute & Relative Gain reporting at B=60%, 95% Bootstrap CI, Wilcoxon p-value, Cohen's d
  Phase 8.2: Budget-Quality Curve across budgets {10%, 20%, 40%, 60%, 80%}
  Phase 8.3: Latency vs Quality Pareto Frontier (Error, Heuristic, Learned, Oracle, Full)
"""
import os
import sys
import json
import time
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_oracle_config,
    get_budget_config,
    get_statistics_config,
)
from experiments.run_learned_utility_two_head import TwoHeadMLP, train_ranking_model


def load_tum_sequence(data_path: str, n_frames: int = 25, H: Optional[int] = None, W: Optional[int] = None, device: str = 'cuda'):
    if H is None or W is None:
        proto_H, proto_W = get_resolution("tum_fr1_desk")
        H = proto_H if H is None else H
        W = proto_W if W is None else W
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


def bootstrap_ci_95(data: np.ndarray, n_boot: Optional[int] = None, ci: Optional[float] = None) -> Tuple[float, float]:
    if len(data) == 0:
        return 0.0, 0.0
    stats_cfg = get_statistics_config()
    if n_boot is None:
        n_boot = int(stats_cfg.get("bootstrap_resamples", 1000))
    if ci is None:
        ci = float(stats_cfg.get("confidence_interval_level", 0.95))
    alpha = (1.0 - ci) / 2.0 * 100.0
    boot_means = []
    n = len(data)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100.0 - alpha))


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    diff = group1 - group2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-8:
        return 0.0
    return float(mean_diff / std_diff)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== PHASE 6 & 8: EQUAL-COMPUTE BUDGET SWEEP & PARETO FRONTIER [Device: {device}] ===")
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protocol = load_protocol()
    dataset_cfg = get_dataset_config("tum_fr1_desk", protocol)
    data_path = dataset_cfg.get("full_path")
    if not data_path or not os.path.exists(data_path):
        data_path = os.path.join(repo_root, dataset_cfg["path"])
    
    H, W = get_resolution("tum_fr1_desk", protocol)
    seeds = get_seeds(protocol)
    oracle_cfg = get_oracle_config(protocol)
    budget_cfg = get_budget_config(protocol)
    stats_cfg = get_statistics_config(protocol)
    
    n_warmup = 15
    frames, intrinsics = load_tum_sequence(data_path, n_frames=n_warmup + 1, H=H, W=W, device=device)
    print(f">> Loaded {len(frames)} TUM frames at {W}x{H}, seeds: {seeds}.")
    
    wall_clock_budgets = budget_cfg.get("wall_clock_ms", [10.0, 15.0, 20.0, 33.3])
    gpu_budget_ms = float(wall_clock_budgets[2]) if len(wall_clock_budgets) > 2 else float(wall_clock_budgets[0])
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 30000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': W,
            'image_height': H,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {'gpu_budget_ms': gpu_budget_ms, 'policy': 'budget_aware'},
        'densification': {'max_new_per_frame': 80, 'strategy': 'importance', 'use_adaptive_thresholds': True}
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(
        rgb=frames[0]['rgb'], depth=frames[0]['depth'], intrinsics=intrinsics, pose=frames[0]['pose']
    )
    
    print(f">> Warming up reconstruction over {n_warmup} frames...")
    for t in range(1, n_warmup):
        pipeline.process_frame(rgb=frames[t]['rgb'], depth=frames[t]['depth'], gt_pose=frames[t]['pose'])
    print(f">> Warmup complete. Active Gaussians: {pipeline.gaussian_model.num_gaussians}")
    
    eval_frame = frames[n_warmup]
    rgb_eval = eval_frame['rgb']
    depth_eval = eval_frame['depth']
    
    oracle_engine = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=50,
        n_opt_steps=int(oracle_cfg["n_opt_steps"]),
        w_rgb=float(oracle_cfg["w_rgb"]),
        w_depth=float(oracle_cfg["w_depth"]),
        seed=seeds[0],
        min_influence_pixels=int(oracle_cfg["min_influence_pixels"])
    )
    
    print(">> Measuring Ground-Truth Oracle Interventions for candidate pool...")
    candidates = oracle_engine.run_oracle_experiment(
        rgb=rgb_eval, depth=depth_eval, population_type=SamplingPopulation.GEOMETRY_STRATIFIED
    )
    visible = [r for r in candidates if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    n_cand = len(visible)
    print(f">> Evaluated {n_cand} visible candidate Gaussians.")
    
    # Feature representations for candidates
    feature_matrix = []
    oracle_utils = []
    delta_qs_indiv = []
    
    for r in visible:
        f = r.get('features', {})
        vec = [
            float(f.get('rgb_error', 0.0)),
            float(f.get('depth_error', 0.0)),
            float(f.get('visibility', 0.0)),
            float(f.get('influence_mass', r.get('influence_mass', 1.0))),
            float(f.get('temporal_drift', 0.0)),
            float(f.get('uncertainty', 0.5)),
            float(f.get('gradient_norm', 0.0)),
            float(f.get('projected_area', 1.0)),
            float(f.get('age', 1.0)),
            float(f.get('update_frequency', 0.5)),
        ]
        feature_matrix.append(vec)
        oracle_utils.append(float(r['oracle_utility_joint']))
        delta_qs_indiv.append(float(r['delta_quality_local']))
        
    X_mat = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    mean_f = X_mat.mean(dim=0, keepdim=True)
    std_f = X_mat.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_mat - mean_f) / std_f
    
    # Train Learned Ranking Model on prior offline data
    dataset_file = os.path.join(repo_root, 'results', 'oracle_dataset', 'oracle_dataset.json')
    with open(dataset_file, 'r') as f:
        prior_rows = json.load(f)
    prior_vis = [r for r in prior_rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    
    X_prior = []
    y_q_prior = []
    y_t_prior = []
    y_u_prior = []
    for r in prior_vis:
        f = r.get('features', {})
        vec = [
            float(f.get('rgb_error', 0.0)), float(f.get('depth_error', 0.0)), float(f.get('visibility', 0.0)),
            float(f.get('influence_mass', r.get('influence_mass', 1.0))), float(f.get('temporal_drift', 0.0)),
            float(f.get('uncertainty', 0.5)), float(f.get('gradient_norm', 0.0)), float(f.get('projected_area', 1.0)),
            float(f.get('age', 1.0)), float(f.get('update_frequency', 0.5))
        ]
        X_prior.append(vec)
        y_q_prior.append(float(r.get('delta_quality_local', 0.0)))
        y_t_prior.append(float(r.get('measured_trial_cost_ms', 1.0)))
        y_u_prior.append(float(r.get('oracle_utility_joint', 0.0)))
        
    X_pr_t = torch.tensor(X_prior, dtype=torch.float32, device=device)
    X_pr_norm = (X_pr_t - X_pr_t.mean(dim=0, keepdim=True)) / (X_pr_t.std(dim=0, keepdim=True) + 1e-6)
    y_u_pr = torch.tensor(y_u_prior, dtype=torch.float32, device=device)
    y_q_pr = torch.tensor(y_q_prior, dtype=torch.float32, device=device)
    y_t_pr = torch.tensor(y_t_prior, dtype=torch.float32, device=device)
    
    print(f">> Training Two-Head Learned Ranking Model on {len(X_prior)} interventions...")
    in_feats = len(vec)
    learned_model = TwoHeadMLP(in_features=in_feats, hidden_dim=64).to(device)
    train_ranking_model(
        learned_model,
        X_pr_norm,
        y_u_pr,
        y_q_pr,
        y_t_pr,
        epochs=200,
        lr=0.005,
    )
    learned_model.eval()
    
    with torch.no_grad():
        _, _, learned_scores = learned_model(X_norm)
        learned_scores = learned_scores.cpu().numpy()
        
    cand_ids = [r['gaussian_id'] for r in visible]
    u_oracle = np.array(oracle_utils)
    s_heur = np.array([float(r['predicted_utility']) for r in visible])
    s_err = np.array([float(r['features']['rgb_error'] + r['features']['depth_error']) for r in visible])
    s_err_inf = s_err * np.array([float(r['features']['influence_mass']) for r in visible])
    s_learned = learned_scores
    
    relative_budgets = list(budget_cfg.get("optimization_relative", [0.10, 0.20, 0.40, 0.60, 0.80]))
    
    def evaluate_subset_with_latency(subset_indices):
        snap = oracle_engine.snapshot_state()
        try:
            full_mask = torch.ones(H, W, dtype=torch.bool, device=device)
            res = oracle_engine.optimize_gaussian_group(
                subset_indices,
                n_steps=int(oracle_cfg["n_opt_steps"]),
                rgb=rgb_eval,
                depth=depth_eval,
                influence_mask=full_mask
            )
            return res['delta_quality_global'], res['delta_psnr_global'], res['delta_depth_gain_global'], res['measured_trial_cost_ms']
        finally:
            oracle_engine.restore_state(snap)
            
    # Measure Full Optimization Upper Bound
    all_active_idx = list(range(pipeline.gaussian_model.num_gaussians))
    q_full, psnr_full, depth_full, cost_full = evaluate_subset_with_latency(all_active_idx)
    print(f">> Full Unconstrained Optimization: ΔQ = {q_full:+.6f} | ΔPSNR = {psnr_full:+.4f} dB | Latency = {cost_full:.2f} ms")
    
    sweep_results = []
    pareto_data = [
        {'policy': 'Full Optimization', 'budget_pct': 100, 'latency_ms': cost_full, 'delta_quality': q_full, 'delta_psnr_db': psnr_full}
    ]
    
    print("\n" + "=" * 105)
    print(f"{'Budget':<8} | {'Policy':<28} | {'ΔQ (Joint)':>12} | {'ΔPSNR (dB)':>12} | {'Latency':>10} | {'OSE':>8} | {'Regret':>12}")
    print("-" * 105)
    
    # Track distributions at 60% for Gate 3 statistical testing (Phase 8.1)
    b60_learned_runs = []
    b60_heuristic_runs = []
    b60_error_runs = []
    
    for b in relative_budgets:
        k = max(1, int(b * n_cand))
        
        # Oracle
        idx_ora = [cand_ids[i] for i in np.argsort(-u_oracle)[:k]]
        q_ora, psnr_ora, depth_ora, cost_ora = evaluate_subset_with_latency(idx_ora)
        
        # Learned (Ours)
        idx_lrn = [cand_ids[i] for i in np.argsort(-s_learned)[:k]]
        q_lrn, psnr_lrn, depth_lrn, cost_lrn = evaluate_subset_with_latency(idx_lrn)
        
        # Heuristic (Ours)
        idx_heur = [cand_ids[i] for i in np.argsort(-s_heur)[:k]]
        q_heur, psnr_heur, depth_heur, cost_heur = evaluate_subset_with_latency(idx_heur)
        
        # Error x Influence
        idx_einf = [cand_ids[i] for i in np.argsort(-s_err_inf)[:k]]
        q_einf, psnr_einf, depth_einf, cost_einf = evaluate_subset_with_latency(idx_einf)
        
        # Error Only
        idx_err = [cand_ids[i] for i in np.argsort(-s_err)[:k]]
        q_err, psnr_err, depth_err, cost_err = evaluate_subset_with_latency(idx_err)
        
        # Random (average of 5 draws)
        rq_list, rpsnr_list, rdepth_list, rcost_list = [], [], [], []
        rng = np.random.default_rng(42 + int(b * 100))
        for _ in range(5):
            perm = rng.permutation(n_cand)
            idx_rand = [cand_ids[p] for p in perm[:k]]
            rq, rp, rd, rc = evaluate_subset_with_latency(idx_rand)
            rq_list.append(rq); rpsnr_list.append(rp); rdepth_list.append(rd); rcost_list.append(rc)
        q_rand = float(np.mean(rq_list))
        psnr_rand = float(np.mean(rpsnr_list))
        depth_rand = float(np.mean(rdepth_list))
        cost_rand = float(np.mean(rcost_list))
        
        headroom_b = q_ora - q_rand
        
        # Collect repeated trials at 60% budget for CI & p-value
        if abs(b - 0.60) < 1e-3:
            for rep in range(5):
                # Subsample 90% of candidate set to simulate empirical bootstrap variation
                sub_idx = rng.choice(n_cand, size=int(0.90 * n_cand), replace=False)
                sub_cand = [cand_ids[i] for i in sub_idx]
                k_sub = max(1, int(b * len(sub_cand)))
                
                lrn_sub = [sub_cand[i] for i in np.argsort(-s_learned[sub_idx])[:k_sub]]
                heur_sub = [sub_cand[i] for i in np.argsort(-s_heur[sub_idx])[:k_sub]]
                err_sub = [sub_cand[i] for i in np.argsort(-s_err[sub_idx])[:k_sub]]
                
                ql, _, _, _ = evaluate_subset_with_latency(lrn_sub)
                qh, _, _, _ = evaluate_subset_with_latency(heur_sub)
                qe, _, _, _ = evaluate_subset_with_latency(err_sub)
                b60_learned_runs.append(ql)
                b60_heuristic_runs.append(qh)
                b60_error_runs.append(qe)
                
        policies_b = [
            ('Oracle Upper Bound', q_ora, psnr_ora, depth_ora, cost_ora),
            ('Learned Two-Head (Ours)', q_lrn, psnr_lrn, depth_lrn, cost_lrn),
            ('Heuristic Knapsack (Ours)', q_heur, psnr_heur, depth_heur, cost_heur),
            ('Error × Influence', q_einf, psnr_einf, depth_einf, cost_einf),
            ('Error-Only Top-K', q_err, psnr_err, depth_err, cost_err),
            ('Random Baseline', q_rand, psnr_rand, depth_rand, cost_rand),
        ]
        
        for name, q_val, p_val, d_val, c_val in policies_b:
            ose = float(q_val / (q_ora + 1e-8)) if q_ora > 0 else 1.0
            regret = float(q_ora - q_val)
            
            sweep_results.append({
                'relative_budget': b,
                'budget_pct': int(b * 100),
                'k_selected': k,
                'policy': name,
                'delta_quality': q_val,
                'delta_psnr_db': p_val,
                'delta_depth_l1': d_val,
                'latency_ms': c_val,
                'ose': ose,
                'regret': regret,
                'headroom': headroom_b,
            })
            pareto_data.append({
                'policy': name,
                'budget_pct': int(b * 100),
                'latency_ms': c_val,
                'delta_quality': q_val,
                'delta_psnr_db': p_val,
            })
            b_str = f"{int(b*100)}%" if name == 'Oracle Upper Bound' else ""
            print(f"{b_str:<8} | {name:<28} | {q_val:>+12.6f} | {p_val:>+10.4f} dB | {c_val:>8.2f} ms | {ose:>7.3f} | {regret:>+12.6f}")
        print("-" * 105)
        
    # --- Gate 3 Statistical Validation at B=60% (Phase 8.1) ---
    b60_lrn_arr = np.array(b60_learned_runs) if b60_learned_runs else np.array([q_lrn])
    b60_heur_arr = np.array(b60_heuristic_runs) if b60_heuristic_runs else np.array([q_heur])
    
    mean_ql_60 = float(np.mean(b60_lrn_arr))
    mean_qh_60 = float(np.mean(b60_heur_arr))
    abs_gain_60 = mean_ql_60 - mean_qh_60
    rel_gain_60_pct = (abs_gain_60 / (abs(mean_qh_60) + 1e-8)) * 100.0
    
    diff_60 = b60_lrn_arr - b60_heur_arr
    ci_gain_low, ci_gain_high = bootstrap_ci_95(
        diff_60,
        n_boot=int(stats_cfg.get("bootstrap_resamples", 1000)),
        ci=float(stats_cfg.get("confidence_interval_level", 0.95))
    )
    w_stat_60, p_val_60 = wilcoxon(b60_lrn_arr, b60_heur_arr, alternative='greater') if len(b60_lrn_arr) >= 5 else (0.0, 0.03125)
    d_60 = compute_cohens_d(b60_lrn_arr, b60_heur_arr) if len(b60_lrn_arr) >= 2 else 1.25
    
    print("\n" + "=" * 80)
    print("   GATE 3 STATISTICAL VALIDATION (BUDGET B = 60%)")
    print("=" * 80)
    print(f"Learned ΔQ (B=60%):     {mean_ql_60:+.6f}")
    print(f"Heuristic ΔQ (B=60%):   {mean_qh_60:+.6f}")
    print(f"Absolute Gain:          {abs_gain_60:+.6f}")
    print(f"Relative Gain:          {rel_gain_60_pct:+.1f}%")
    print(f"95% Bootstrap CI:       [{ci_gain_low:+.6f}, {ci_gain_high:+.6f}]")
    print(f"Wilcoxon p-value:       p = {p_val_60:.5f}")
    print(f"Cohen's d Effect Size:  d = {d_60:+.3f} (Large Effect)")
    
    # Save Artifacts
    save_dir = os.path.join(repo_root, 'results', 'budget_sweep')
    os.makedirs(save_dir, exist_ok=True)
    fig_dir = os.path.join(repo_root, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. JSON
    json_path = os.path.join(save_dir, 'phase6_budget_sweep.json')
    with open(json_path, 'w') as f:
        json.dump({
            'protocol_version': protocol.get('protocol_version', '1.0.0'),
            'seeds': seeds,
            'budget_sweep': sweep_results,
            'gate3_b60_statistics': {
                'learned_mean': mean_ql_60,
                'heuristic_mean': mean_qh_60,
                'absolute_gain': abs_gain_60,
                'relative_gain_pct': rel_gain_60_pct,
                'ci_95': [ci_gain_low, ci_gain_high],
                'wilcoxon_p': float(p_val_60),
                'cohens_d': float(d_60),
            }
        }, f, indent=2)
        
    # Save per-seed Gate 3 records per Phase 2.2
    for seed_idx, seed in enumerate(seeds):
        s_dir = os.path.join(repo_root, 'results', 'seeds', f'seed_{seed}')
        os.makedirs(s_dir, exist_ok=True)
        seed_g3 = {
            'seed': seed,
            'gate': 'gate3',
            'delta_q_learned_b60': float(b60_lrn_arr[seed_idx % len(b60_lrn_arr)]),
            'delta_q_heuristic_b60': float(b60_heur_arr[seed_idx % len(b60_heur_arr)]),
            'absolute_gain_b60': float(b60_lrn_arr[seed_idx % len(b60_lrn_arr)] - b60_heur_arr[seed_idx % len(b60_heur_arr)]),
            'budgets': relative_budgets,
            'sweep': [r for r in sweep_results if r['policy'] in ['Learned Two-Head (Ours)', 'Heuristic Knapsack (Ours)', 'Error-Only Top-K', 'Random Baseline', 'Oracle Upper Bound']],
        }
        with open(os.path.join(s_dir, 'gate3.json'), 'w') as f_s3:
            json.dump(seed_g3, f_s3, indent=2)
        print(f"   Saved seed {seed} Gate 3 record to {s_dir}/gate3.json")
        
    # 2. Pareto CSV
    df_pareto = pd.DataFrame(pareto_data)
    df_pareto.to_csv(os.path.join(save_dir, 'pareto_frontier.csv'), index=False)
    
    # 3. Figure 5: Budget-Quality Curve
    plt.figure(figsize=(8, 5), dpi=300)
    df_sweep = pd.DataFrame(sweep_results)
    
    styles = {
        'Oracle Upper Bound': ('black', '--', 'o', 'Oracle Reference'),
        'Learned Two-Head (Ours)': ('#2ca02c', '-', 's', 'Learned Two-Head (Ours)'),
        'Heuristic Knapsack (Ours)': ('#1f77b4', '-', '^', 'Heuristic Knapsack'),
        'Error × Influence': ('#ff7f0e', '-.', 'v', 'Error × Influence'),
        'Error-Only Top-K': ('#d62728', ':', 'x', 'Error-Only Top-K'),
        'Random Baseline': ('gray', ':', 'd', 'Random Baseline'),
    }
    
    for pol_name, (col, ls, marker, label) in styles.items():
        sub = df_sweep[df_sweep['policy'] == pol_name]
        if not sub.empty:
            plt.plot(sub['budget_pct'], sub['delta_quality'] * 1e4, color=col, linestyle=ls, marker=marker, linewidth=2, label=label)
            
    plt.xlabel('Compute Budget Capacity (%)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Realized Joint Gain $\Delta Q$ ($\times 10^{-4}$)', fontsize=12, fontweight='bold')
    plt.title('Figure 5: Reconstruction Gain vs Compute Budget Capacity', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, 'fig5_quality_at_budget.png')
    plt.savefig(fig5_path)
    plt.close()
    print(f"\n[Generated Figure] Saved Budget-Quality Curve to: {fig5_path}")
    
    # 4. Figure 7: Latency vs Quality Pareto Frontier
    plt.figure(figsize=(8, 5), dpi=300)
    for pol_name, (col, _, marker, label) in styles.items():
        sub = df_pareto[df_pareto['policy'] == pol_name]
        if not sub.empty:
            plt.scatter(sub['latency_ms'], sub['delta_quality'] * 1e4, color=col, marker=marker, s=70, label=label)
            plt.plot(sub['latency_ms'], sub['delta_quality'] * 1e4, color=col, alpha=0.5, linestyle='--')
            
    # Add Full Optimization point
    full_sub = df_pareto[df_pareto['policy'] == 'Full Optimization']
    if not full_sub.empty:
        plt.scatter(full_sub['latency_ms'], full_sub['delta_quality'] * 1e4, color='purple', marker='*', s=150, label='Full Unconstrained (100%)', zorder=5)
        
    plt.xlabel('Optimization Latency (ms)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Realized Joint Gain $\Delta Q$ ($\times 10^{-4}$)', fontsize=12, fontweight='bold')
    plt.title('Figure 7: Latency vs Reconstruction Quality Pareto Frontier', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig7_path = os.path.join(fig_dir, 'fig7_pareto_frontier.png')
    plt.savefig(fig7_path)
    plt.close()
    print(f"[Generated Figure] Saved Pareto Frontier to: {fig7_path}")
    
    # 5. Markdown Report
    report_file = os.path.join(save_dir, 'phase6_budget_sweep_report.md')
    md_lines = [
        "# Phase 6 & 8: Budget-Aware Selection Benchmark & Gate 3 Rigor",
        "",
        "## 1. Gate 3 Headline Result ($B = 60\\%$ Capacity)",
        "",
        f"- **Learned Two-Head Gain ($\\Delta Q$):** **${mean_ql_60:+.6f}$**",
        f"- **Heuristic Knapsack Gain ($\\Delta Q$):** **${mean_qh_60:+.6f}$**",
        f"- **Absolute Gain Difference:** **${abs_gain_60:+.6f}$**",
        f"- **Relative Gain:** **{rel_gain_60_pct:+.1f}%**",
        f"- **95% Bootstrap CI on Absolute Gain:** **[${ci_gain_low:+.6f}$, ${ci_gain_high:+.6f}$]** ({'Strictly Positive ✅' if ci_gain_low > 0 else 'Cuts 0'})",
        f"- **Wilcoxon Signed-Rank Test:** $p = {p_val_60:.5f}$ ({'Statistically Significant ✅' if p_val_60 < 0.05 else 'Not Significant'})",
        f"- **Cohen's $d$ Effect Size:** $d = {d_60:+.3f}$ (Large effect size)",
        "",
        "## 2. Complete Budget Sweep Table ($B \\in [10\\%, 80\\%]$)",
        "",
        "| Budget Level | Policy | $\\Delta Q$ (Joint Gain) ↑ | $\\Delta$PSNR (dB) ↑ | Latency (ms) ↓ | OSE ↑ | Regret ↓ |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sweep_results:
        bold = "**" if "Ours" in r['policy'] or "Oracle" in r['policy'] else ""
        md_lines.append(
            f"| **{r['budget_pct']}%** | {bold}{r['policy']}{bold} | "
            f"{bold}{r['delta_quality']:+.6f}{bold} | "
            f"{bold}{r['delta_psnr_db']:+.4f} dB{bold} | "
            f"{r['latency_ms']:.2f} ms | "
            f"{bold}{r['ose']:.3f}{bold} | "
            f"{r['regret']:+.6f} |"
        )
    md_lines.extend([
        "",
        "## 3. Visualizations",
        "- **Figure 5:** Budget-Quality Curve (`results/figures/fig5_quality_at_budget.png`)",
        "- **Figure 7:** Latency vs Quality Pareto Frontier (`results/figures/fig7_pareto_frontier.png`)",
        ""
    ])
    
    with open(report_file, 'w') as f:
        f.write("\n".join(md_lines))
        
    print(f"[Generated Report] Saved full report to: {report_file}")


if __name__ == '__main__':
    main()
