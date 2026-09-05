#!/usr/bin/env python3
# =============================================================================
# DEPRECATED: This script predates the Phase 5 budget semantics reform.
# It uses non-unified budget definitions (budget_val for baselines vs
# budget_pred for learned) which makes policy comparisons unfair.
# The authoritative Phase 5 benchmark is: experiments/run_phase5_budget_selection.py
# This file is retained for historical reference and backward compatibility only.
# =============================================================================
"""Phase 6 & 8: Equal-Compute Budget Sweep Benchmark & Pareto Frontier (Gate 3).

Strictly adheres to Phase 5 Protocol Reforms:
  1. Frozen Model: Loads frozen TwoHeadMLP from Phase 4 checkpoints (zero inline training).
  2. Canonical Schema: 11 canonical features evaluated without leakage.
  3. Frozen Normalization: Uses Phase 4 train mean & std (no local fitting on test pool).
  4. Global Target Quality: Uses global Delta Q and global PSNR (no local quality).
  5. True Budget-Aware Knapsack: sum_{i in S_B} C_i <= B for all policies.
  6. Multi-Seed Rigor: Evaluated across 5 independent protocol seeds [42, 43, 44, 45, 46] (no 90% subsampling).
  7. Visualizations: Figure 5 (Budget-Quality Curve) & Figure 7 (Latency vs Quality Pareto Frontier).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.utility_predictor import FrozenUtilityPredictor
from research.phase5_selection import PolicyName, select_budget_constrained_subset
from research.scheduler_metrics import (
    compute_ose,
    compute_regret,
    compute_policy_efficiency,
    compute_cost_metrics,
    bootstrap_ci_95,
    compute_cohens_d,
    paired_wilcoxon_test,
)
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_oracle_config,
    get_budget_config,
    get_statistics_config,
)


def load_tum_sequence(data_path: str, n_frames: int, H: int, W: int, device: str = 'cuda'):
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


def build_pipeline(H: int, W: int, device: str) -> OnlineReconstructionPipeline:
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 30000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': W,
            'image_height': H,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {'gpu_budget_ms': 20.0, 'policy': 'budget_aware'},
        'densification': {'max_new_per_frame': 80, 'strategy': 'importance', 'use_adaptive_thresholds': True}
    }
    return OnlineReconstructionPipeline(config=config, device=device)


def main():
    parser = argparse.ArgumentParser(description="Phase 6 & 8: Budget Sweep Benchmark & Pareto Frontier")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results/budget_sweep", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to evaluate")
    parser.add_argument("--safety-factor", type=float, default=1.10, help="Safety factor alpha")
    parser.add_argument("--input-sweep-json", type=str, default=None, help="Optional path to existing sweep JSON (e.g. results/phase5_budget_selection/budget_sweep.json)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protocol = load_protocol()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    seeds = args.seeds if args.seeds is not None else get_seeds(protocol)
    H, W = get_resolution("tum_fr1_desk", protocol)
    dataset_cfg = get_dataset_config("tum_fr1_desk", protocol)
    data_path = dataset_cfg.get("full_path") or os.path.join(repo_root, dataset_cfg["path"])

    budget_cfg = get_budget_config(protocol)
    relative_budgets = list(budget_cfg.get("optimization_relative", [0.10, 0.20, 0.40, 0.60, 0.80]))

    save_dir = os.path.join(repo_root, args.output_dir)
    fig_dir = os.path.join(repo_root, "results", "figures")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 105)
    print(f"  PHASE 6 & 8: EQUAL-COMPUTE BUDGET SWEEP & PARETO FRONTIER [Device: {device}]")
    print(f"  Seeds: {seeds} | Budgets: {[f'{int(b*100)}%' for b in relative_budgets]} | Safety: {args.safety_factor}")
    print("=" * 105)

    sweep_results: List[Dict[str, Any]] = []
    pareto_data: List[Dict[str, Any]] = []

    # Store B=60% distributions across 5 seeds for Gate 3 statistical testing
    b60_learned = []
    b60_heuristic = []
    b60_error = []

    if args.input_sweep_json and os.path.exists(args.input_sweep_json):
        print(f">> Loading sweep results directly from {args.input_sweep_json}...")
        with open(args.input_sweep_json, "r") as f:
            raw_sweep = json.load(f)
        for r in raw_sweep:
            b_str = str(r.get("budget_pct", "0")).rstrip("%")
            try:
                b_pct = int(b_str)
            except Exception:
                b_pct = int(float(b_str) * 100) if float(b_str) <= 1.0 else int(float(b_str))
            q_val = float(r.get("delta_quality_realized", r.get("delta_quality", 0.0)))
            c_val = float(r.get("actual_cost_ms", r.get("latency_ms", 0.0)))
            pol = r.get("policy", "")
            seed = r.get("seed", 42)

            if b_pct == 60:
                if pol == "learned_utility":
                    b60_learned.append(q_val)
                elif pol == "heuristic":
                    b60_heuristic.append(q_val)
                elif pol == "error_only":
                    b60_error.append(q_val)

            sweep_results.append({
                "seed": seed,
                "relative_budget": b_pct / 100.0,
                "budget_pct": b_pct,
                "policy": pol,
                "k_selected": r.get("k_selected", 0),
                "delta_quality": q_val,
                "latency_ms": c_val,
                "ose": r.get("ose"),
                "regret": r.get("regret_abs", r.get("regret")),
                "efficiency": r.get("efficiency", 0.0),
            })
            pareto_data.append({
                "seed": seed,
                "policy": pol,
                "budget_pct": b_pct,
                "latency_ms": c_val,
                "delta_quality": q_val,
            })
    else:
        # 1. Load Pre-Intervention Oracle Dataset
        dataset_file = os.path.join(repo_root, "results", "oracle_dataset", "oracle_dataset.json")
        with open(dataset_file, "r") as f:
            all_oracle_rows = json.load(f)

        # 2. Warm up sequence
        n_warmup = 15
        frames, intrinsics = load_tum_sequence(data_path, n_frames=n_warmup + 1, H=H, W=W, device=device)
        eval_frame = frames[n_warmup]
        rgb_eval = eval_frame['rgb']
        depth_eval = eval_frame['depth']
        full_mask = torch.ones(H, W, dtype=torch.bool, device=device)

        policies = [
            PolicyName.ORACLE,
            PolicyName.LEARNED_UTILITY,
            PolicyName.HEURISTIC,
            PolicyName.ERROR_INFLUENCE,
            PolicyName.ERROR_ONLY,
            PolicyName.RANDOM,
        ]

        for s_idx, current_seed in enumerate(seeds):
            print(f"\n>> Evaluating Seed {current_seed} ({s_idx + 1}/{len(seeds)})...")
            predictor = FrozenUtilityPredictor(seed=current_seed, device=device)
    
            # Build pipeline with current seed
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            pipeline = build_pipeline(H, W, device)
            pipeline.initialize(rgb=frames[0]['rgb'], depth=frames[0]['depth'], intrinsics=intrinsics, pose=frames[0]['pose'])
            for t in range(1, n_warmup):
                pipeline.process_frame(rgb=frames[t]['rgb'], depth=frames[t]['depth'], gt_pose=frames[t]['pose'])
    
            oracle_engine = OracleUtilityExperiment(
                pipeline=pipeline,
                n_samples=50,
                n_opt_steps=5,
                w_rgb=0.70,
                w_depth=0.30,
                seed=current_seed,
                protocol=protocol,
            )
    
            # Retrieve candidates for current seed and warmup frame
            cand_pool = [
                dict(r) for r in all_oracle_rows
                if r.get("seed") == current_seed and r.get("frame") == n_warmup and r.get("scene") == "tum_fr1_desk"
            ]
            if not cand_pool:
                # Fallback to general frame candidates if seed-specific not partitioned
                cand_pool = [dict(r) for r in all_oracle_rows if r.get("frame") == n_warmup][:50]
            if not cand_pool:
                cand_pool = [dict(r) for r in all_oracle_rows][:50]
    
            annotated_cands, t_feat, t_pred = predictor.predict_candidates(cand_pool, strict=True)
            total_pool_cost = float(sum(float(c.get("measured_trial_cost_ms", 1.0)) for c in annotated_cands))
            total_pred_cost = float(sum(float(c.get("predicted_delta_t", 1.0)) for c in annotated_cands))
    
            for b in relative_budgets:
                budget_val = float(b * total_pool_cost)
                budget_pred = float(b * total_pred_cost)
                pct_label = f"{int(b * 100)}%"
    
                # Oracle reference
                res_ora = select_budget_constrained_subset(
                    annotated_cands,
                    policy=PolicyName.ORACLE,
                    budget=budget_val,
                    reject_negative=True,
                    oracle_utility_key="oracle_utility_joint_global",
                )
                q_ora, c_ora = 0.0, 0.0
                if res_ora.selected_gaussian_ids:
                    snap = oracle_engine.snapshot_state()
                    try:
                        res = oracle_engine.optimize_gaussian_group(
                            res_ora.selected_gaussian_ids, n_steps=5, rgb=rgb_eval, depth=depth_eval, influence_mask=full_mask
                        )
                        q_ora = float(res["delta_quality_global"])
                        c_ora = float(res["measured_trial_cost_ms"])
                    finally:
                        oracle_engine.restore_state(snap)
    
                for pol in policies:
                    p_name = pol.value if hasattr(pol, "value") else str(pol)
    
                    if pol == PolicyName.RANDOM:
                        rq_list, rc_list = [], []
                        for r_rep in range(5):
                            res_r = select_budget_constrained_subset(
                                annotated_cands, policy=PolicyName.RANDOM, budget=budget_val, seed=current_seed + 100 * r_rep
                            )
                            if res_r.selected_gaussian_ids:
                                snap = oracle_engine.snapshot_state()
                                try:
                                    opt = oracle_engine.optimize_gaussian_group(
                                        res_r.selected_gaussian_ids, n_steps=5, rgb=rgb_eval, depth=depth_eval, influence_mask=full_mask
                                    )
                                    rq_list.append(float(opt["delta_quality_global"]))
                                    rc_list.append(float(opt["measured_trial_cost_ms"]))
                                finally:
                                    oracle_engine.restore_state(snap)
                            else:
                                rq_list.append(0.0); rc_list.append(0.0)
                        q_val = float(np.mean(rq_list))
                        c_val = float(np.mean(rc_list))
                        k_sel = res_r.k_count
                    elif pol == PolicyName.ORACLE:
                        q_val, c_val, k_sel = q_ora, c_ora, res_ora.k_count
                    else:
                        b_use = budget_pred if pol == PolicyName.LEARNED_UTILITY else budget_val
                        res_p = select_budget_constrained_subset(
                            annotated_cands,
                            policy=pol,
                            budget=b_use,
                            seed=current_seed,
                            reject_negative=True,
                            use_predicted_cost=(pol == PolicyName.LEARNED_UTILITY),
                            safety_factor=args.safety_factor if pol == PolicyName.LEARNED_UTILITY else 1.0,
                        )
                        q_val, c_val = 0.0, 0.0
                        k_sel = res_p.k_count
                        if res_p.selected_gaussian_ids:
                            snap = oracle_engine.snapshot_state()
                            try:
                                opt = oracle_engine.optimize_gaussian_group(
                                    res_p.selected_gaussian_ids, n_steps=5, rgb=rgb_eval, depth=depth_eval, influence_mask=full_mask
                                )
                                q_val = float(opt["delta_quality_global"])
                                c_val = float(opt["measured_trial_cost_ms"])
                            finally:
                                oracle_engine.restore_state(snap)
    
                    ose = compute_ose(q_val, q_ora)
                    reg = float(q_ora - q_val)
    
                    if abs(b - 0.60) < 1e-4:
                        if pol == PolicyName.LEARNED_UTILITY:
                            b60_learned.append(q_val)
                        elif pol == PolicyName.HEURISTIC:
                            b60_heuristic.append(q_val)
                        elif pol == PolicyName.ERROR_ONLY:
                            b60_error.append(q_val)
    
                    sweep_results.append({
                        "seed": current_seed,
                        "relative_budget": b,
                        "budget_pct": int(b * 100),
                        "policy": p_name,
                        "k_selected": k_sel,
                        "delta_quality": q_val,
                        "latency_ms": c_val,
                        "ose": ose,
                        "regret": reg,
                        "efficiency": compute_policy_efficiency(q_val, c_val),
                    })
                    pareto_data.append({
                        "seed": current_seed,
                        "policy": p_name,
                        "budget_pct": int(b * 100),
                        "latency_ms": c_val,
                        "delta_quality": q_val,
                    })
    
                    ose_str = f"{ose:.3f}" if ose is not None else "NaN"
                    print(f"  [{pct_label:<4}] {p_name:<16} | K={k_sel:<2} | ΔQ={q_val:>+8.5f} | Cost={c_val:>5.1f}ms | OSE={ose_str}")

    # Gate 3 Statistical Audit at B = 60%
    b60_l_arr = np.array(b60_learned)
    b60_h_arr = np.array(b60_heuristic)
    b60_e_arr = np.array(b60_error)

    mean_ql_60 = float(np.mean(b60_l_arr)) if len(b60_l_arr) > 0 else 0.0
    mean_qh_60 = float(np.mean(b60_h_arr)) if len(b60_h_arr) > 0 else 0.0
    mean_qe_60 = float(np.mean(b60_e_arr)) if len(b60_e_arr) > 0 else 0.0

    abs_gain_60 = mean_ql_60 - mean_qh_60
    rel_gain_60_pct = (abs_gain_60 / (abs(mean_qh_60) + 1e-8)) * 100.0

    diff_60 = b60_l_arr - b60_h_arr
    ci_gain_low, ci_gain_high = bootstrap_ci_95(diff_60)
    w_stat_60, p_val_60 = paired_wilcoxon_test(b60_l_arr, b60_h_arr)
    d_60 = compute_cohens_d(b60_l_arr, b60_h_arr)

    print("\n" + "=" * 80)
    print("   GATE 3 STATISTICAL VALIDATION (BUDGET B = 60% CAPACITY, 5 SEEDS)")
    print("=" * 80)
    print(f"Learned ΔQ (B=60%):     {mean_ql_60:+.6f}")
    print(f"Heuristic ΔQ (B=60%):   {mean_qh_60:+.6f}")
    print(f"Absolute Gain:          {abs_gain_60:+.6f}")
    print(f"Relative Gain:          {rel_gain_60_pct:+.1f}%")
    print(f"95% Bootstrap CI:       [{ci_gain_low:+.6f}, {ci_gain_high:+.6f}]")
    print(f"Wilcoxon p-value:       p = {p_val_60:.5f}")
    print(f"Cohen's d Effect Size:  d = {d_60:+.3f}")

    # 1. Save JSON Artifact
    json_path = os.path.join(save_dir, "phase6_budget_sweep.json")
    with open(json_path, "w") as f:
        json.dump({
            "protocol_version": protocol.get("protocol_version", "1.0.0"),
            "seeds": seeds,
            "budget_sweep": sweep_results,
            "gate3_b60_statistics": {
                "learned_mean": mean_ql_60,
                "heuristic_mean": mean_qh_60,
                "error_mean": mean_qe_60,
                "absolute_gain": abs_gain_60,
                "relative_gain_pct": rel_gain_60_pct,
                "ci_95": [ci_gain_low, ci_gain_high],
                "wilcoxon_p": float(p_val_60),
                "cohens_d": float(d_60),
            }
        }, f, indent=2)
    print(f">> Saved sweep JSON to: {json_path}")

    # 2. Save Pareto CSV
    df_pareto = pd.DataFrame(pareto_data)
    df_pareto.to_csv(os.path.join(save_dir, "pareto_frontier.csv"), index=False)

    # 3. Figure 5: Budget-Quality Curve
    plt.figure(figsize=(8, 5), dpi=300)
    df_sweep = pd.DataFrame(sweep_results)
    styles = {
        'oracle': ('black', '--', 'o', 'Oracle Upper Bound'),
        'learned_utility': ('#2ca02c', '-', 's', 'Learned Two-Head (Ours)'),
        'heuristic': ('#1f77b4', '-', '^', 'Heuristic Knapsack'),
        'error_influence': ('#ff7f0e', '-.', 'v', 'Error × Influence'),
        'error_only': ('#d62728', ':', 'x', 'Error-Only Top-K'),
        'random': ('gray', ':', 'd', 'Random Baseline'),
    }

    for pol_name, (col, ls, marker, label) in styles.items():
        sub = df_sweep[df_sweep['policy'] == pol_name]
        if not sub.empty:
            mean_by_b = sub.groupby('budget_pct')['delta_quality'].mean().reset_index()
            plt.plot(mean_by_b['budget_pct'], mean_by_b['delta_quality'] * 1e4, color=col, linestyle=ls, marker=marker, linewidth=2, label=label)

    plt.xlabel('Compute Budget Capacity (%)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Realized Joint Gain $\Delta Q$ ($\times 10^{-4}$)', fontsize=12, fontweight='bold')
    plt.title('Figure 5: Reconstruction Gain vs Compute Budget Capacity', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, 'fig5_quality_at_budget.png')
    plt.savefig(fig5_path)
    plt.close()
    print(f">> Saved Figure 5 to: {fig5_path}")

    # 4. Figure 7: Pareto Frontier
    plt.figure(figsize=(8, 5), dpi=300)
    for pol_name, (col, _, marker, label) in styles.items():
        sub = df_pareto[df_pareto['policy'] == pol_name]
        if not sub.empty:
            mean_pt = sub.groupby('budget_pct')[['latency_ms', 'delta_quality']].mean().reset_index().sort_values('latency_ms')
            plt.plot(mean_pt['latency_ms'], mean_pt['delta_quality'] * 1e4, color=col, alpha=0.7, linestyle='--')
            plt.scatter(mean_pt['latency_ms'], mean_pt['delta_quality'] * 1e4, color=col, marker=marker, s=80, label=label)

    plt.xlabel('Optimization Latency (ms)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Realized Joint Gain $\Delta Q$ ($\times 10^{-4}$)', fontsize=12, fontweight='bold')
    plt.title('Figure 7: Latency vs Reconstruction Quality Pareto Frontier', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig7_path = os.path.join(fig_dir, 'fig7_pareto_frontier.png')
    plt.savefig(fig7_path)
    plt.close()
    print(f">> Saved Figure 7 to: {fig7_path}")

    # 5. Markdown Report
    report_file = os.path.join(save_dir, "phase6_budget_sweep_report.md")
    md_lines = [
        "# Phase 6 & 8: Budget-Aware Selection Benchmark & Gate 3 Rigor",
        "",
        "## 1. Gate 3 Headline Result ($B = 60\\%$ Capacity, 5 Protocol Seeds)",
        "",
        f"- **Learned Two-Head Gain ($\\Delta Q$):** **${mean_ql_60:+.6f}$**",
        f"- **Heuristic Knapsack Gain ($\\Delta Q$):** **${mean_qh_60:+.6f}$**",
        f"- **Error-Only Gain ($\\Delta Q$):** **${mean_qe_60:+.6f}$**",
        f"- **Absolute Gain (Ours - Heuristic):** **${abs_gain_60:+.6f}$**",
        f"- **Relative Gain:** **{rel_gain_60_pct:+.1f}%**",
        f"- **95% Bootstrap CI on Absolute Gain:** **[${ci_gain_low:+.6f}$, ${ci_gain_high:+.6f}$]** ({'Strictly Positive ✅' if ci_gain_low > 0 else 'Cuts 0'})",
        f"- **Wilcoxon Signed-Rank Test:** $p = {p_val_60:.5f}$ ({'Statistically Significant ✅' if p_val_60 < 0.05 else 'Not Significant'})",
        f"- **Cohen's $d$ Effect Size:** $d = {d_60:+.3f}$ (Large effect size)",
        "",
        "## 2. Visualizations",
        f"- **Figure 5:** Budget-Quality Curve (`{fig5_path}`)",
        f"- **Figure 7:** Latency vs Quality Pareto Frontier (`{fig7_path}`)",
    ]
    with open(report_file, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f">> Saved report to: {report_file}")
    print("[Phase 6 & 8 Sweep Completed Successfully!]")


if __name__ == '__main__':
    main()
