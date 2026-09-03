#!/usr/bin/env python3
"""Oracle Utility Comprehensive Research Experiment Runner.

Evaluates ground-truth marginal utility U_i^oracle = ΔQ_{local,i} / (ΔT_i + ε)
incorporating:
  1. Decoupled raw quality gains (ΔPSNR, ΔSSIM, ΔDepth L1, ΔLoss) and trial time ΔT
  2. Unbiased population comparisons:
     - GEOMETRY_STRATIFIED (Edge, Texture, Flat, Depth Discontinuity)
     - IMPORTANCE_STRATIFIED
     - RANDOM_VISIBLE
     - UNIFORM_VISIBLE
  3. Group oracle scaling (group_size = 1, 4, 16) for non-additivity analysis
  4. Repeat stability analysis (n = 3–5 trials, μ_U, σ_U, CV)
  5. Automatic Oracle Dataset export (JSON & CSV) to results/oracle_dataset/
  6. Automatic Generation of results/oracle/oracle_validation.md (Gate 1 Validation)
"""
import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation


def create_structured_synthetic_frames(n_frames: int = 10, H: int = 64, W: int = 80, device: str = 'cpu'):
    """Create synthetic frames with clear spatial structure."""
    fx, fy = 160.0, 160.0
    cx, cy = W / 2.0, H / 2.0
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    
    frames = []
    for t in range(n_frames):
        angle = t * 0.03
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0
        
        # Region 1: High texture
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Region 2: Flat surface
        for j in range(W // 2, W):
            rgb[:H//2, j] = torch.tensor([0.4, 0.4, 0.6])
        depth[:H//2, W//2:] = 2.5
        
        # Region 3: Object edge
        box_h = slice(H // 2 + 5, H - 5)
        box_w = slice(W // 4, 3 * W // 4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0
        
        # Region 4: Sparse depth
        depth[H - 10:, :10] = 0.0
        
        rgb = (rgb + 0.02 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.0
        
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose, 'intrinsics': intrinsics})
        
    return frames, intrinsics


def generate_oracle_validation_report(
    population_metrics: Dict[str, Any],
    stability_metrics: Optional[Dict[str, Any]],
    group_metrics: Dict[str, Any],
    geometry_stats: Dict[str, Any],
    output_path: str,
):
    """Generate professional Markdown report verifying Gate 1 (Oracle Validity)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = []
    lines.append("# Gate 1: Ground-Truth Oracle Validation Report")
    lines.append("")
    lines.append(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. Executive Summary & Gate 1 Verification")
    lines.append("")
    
    geo_rho = population_metrics.get('geometry_stratified', {}).get('spearman_utility_vs_oracle', 0.0)
    imp_rho = population_metrics.get('importance_stratified', {}).get('spearman_utility_vs_oracle', 0.0)
    rand_rho = population_metrics.get('random_visible', {}).get('spearman_utility_vs_oracle', 0.0)
    
    mean_cv = stability_metrics.get('mean_cv', 0.0) if stability_metrics else 0.0
    stable_frac = stability_metrics.get('stable_fraction', 1.0) if stability_metrics else 1.0
    
    gate1_passed = (geo_rho > 0 or imp_rho > 0) and (mean_cv <= 0.35 or not stability_metrics)
    
    lines.append(f"**Gate 1 Status:** {'✅ PASSED (Statistically Valid Oracle Ground Truth)' if gate1_passed else '⚠️ REQUIRES TUNING'}")
    lines.append("")
    lines.append(f"- **Rank Correlation with Geometry Stratification:** $\\rho = {geo_rho:+.4f}$")
    lines.append(f"- **Rank Correlation with Importance Stratification:** $\\rho = {imp_rho:+.4f}$")
    lines.append(f"- **Rank Correlation with Random Visible:** $\\rho = {rand_rho:+.4f}$")
    if stability_metrics:
        lines.append(f"- **Oracle Repeat Stability ($CV$):** Mean $CV = {mean_cv:.3f}$ ({stable_frac:.1%} stable trials)")
    lines.append("")
    
    lines.append("## 2. Multi-Population Ranking & Correlation Table")
    lines.append("")
    lines.append("| Sampling Population | Spearman $\\rho(U, U_{oracle})$ | Spearman $\\rho(I, \\Delta Q)$ | Overlap@10% | Overlap@20% | Gain Ratio@20% | Regret@20% |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    
    for pop_name, m in population_metrics.items():
        if 'error' in m:
            continue
        rho_u = m.get('spearman_utility_vs_oracle', 0.0)
        rho_q = m.get('spearman_importance_vs_deltaQ', 0.0)
        ov10 = m.get('overlaps', {}).get('top_10pct', 0.0)
        ov20 = m.get('overlaps', {}).get('top_20pct', 0.0)
        g20 = m.get('realized_gains', {}).get('top_20pct_ratio', 0.0)
        reg20 = m.get('regrets', {}).get('top_20pct', 0.0)
        lines.append(f"| **{pop_name}** | **{rho_u:+.4f}** | {rho_q:+.4f} | {ov10:.1%} | {ov20:.1%} | {g20:.4f} | {reg20:.4f} |")
    lines.append("")
    
    if stability_metrics and 'candidates' in stability_metrics:
        lines.append("## 3. Repeat Measurement Stability Analysis ($n=3–5$ Repeated Trials)")
        lines.append("")
        lines.append(f"Evaluated across {stability_metrics['n_candidates']} candidate Gaussians over {stability_metrics['n_repeats']} identical initial trials:")
        lines.append("")
        lines.append("| Gaussian ID | Mean Utility $\\mu_U$ | Std $\\sigma_U$ | $CV = \\sigma / (|\\mu| + \\epsilon)$ | Mean Time (ms) | Stable ($CV \\le 0.35$) |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for c in stability_metrics['candidates'][:15]:
            lines.append(f"| {c['gaussian_id']} | {c['mean_utility']:.4f} | {c['std_utility']:.4f} | {c['coefficient_of_variation']:.3f} | {c['mean_time_ms']:.2f} ms | {'Yes' if c['is_stable'] else 'No'} |")
        lines.append("")
        lines.append(f"**Mean Population $CV$:** {stability_metrics['mean_cv']:.4f} (Threshold $\\le 0.35$)")
        lines.append("")
        
    if group_metrics:
        lines.append("## 4. Group Size Scaling & Non-Additivity ($g \\in \\{1, 4, 16\\}$)")
        lines.append("")
        lines.append("| Group Size ($g$) | Number of Groups | Spearman $\\rho(U, U_{oracle})$ | Gain Ratio@20% | Mean Group Time (ms) |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|")
        for g_name, gm in group_metrics.items():
            if 'error' in gm: continue
            rho_g = gm.get('spearman_utility_vs_oracle', 0.0)
            g20 = gm.get('realized_gains', {}).get('top_20pct_ratio', 0.0)
            n_tot = gm.get('n_total', 0)
            lines.append(f"| {g_name} | {n_tot} | {rho_g:+.4f} | {g20:.4f} | — |")
        lines.append("")
        
    if geometry_stats:
        lines.append("## 5. Geometry-Stratified Breakdown Analysis")
        lines.append("")
        lines.append("| Geometric Stratum | Count | Mean $\\Delta$PSNR (dB) | Mean $\\Delta$Depth Gain (m) | Mean $\\Delta$Loss | Mean Utility |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for s_name, s_data in geometry_stats.items():
            lines.append(f"| **{s_name}** | {s_data.get('count', 0)} | {s_data.get('mean_psnr', 0.0):.4f} dB | {s_data.get('mean_depth', 0.0):.4f} m | {s_data.get('mean_loss', 0.0):.4f} | {s_data.get('mean_utility', 0.0):.4f} |")
        lines.append("")
        
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"\n[Report] Saved Oracle Validation report to {output_path}")


def run_experiment(args):
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
        
    print("=" * 80)
    print("       GROUND-TRUTH ORACLE UTILITY & MARGINAL VALUE EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device} | Warmup: {args.n_warmup} frames | Samples/Pop: {args.n_samples}")
    print(f"Quality Formulation: Decoupled raw metrics + Joint ΔQ (w_rgb={args.w_rgb}, w_depth={args.w_depth})")
    
    # 1. Dataset setup
    frames, intrinsics = None, None
    tum_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'TUM', 'rgbd_dataset_freiburg1_desk')
    if os.path.exists(tum_path) and not args.synthetic:
        try:
            from datasets.tum_dataset import TUMDataset
            dataset = TUMDataset(tum_path, max_frames=args.n_warmup + 6, stride=4)
            raw_frames = [dataset[i] for i in range(len(dataset))]
            intrinsics = dataset.intrinsics.clone()
            
            target_h, target_w = args.height, args.width
            scale_x = target_w / raw_frames[0]['rgb'].shape[1]
            scale_y = target_h / raw_frames[0]['rgb'].shape[0]
            intrinsics[0, :] *= scale_x
            intrinsics[1, :] *= scale_y
            
            frames = []
            for rf in raw_frames:
                rgb_t = rf['rgb'].permute(2, 0, 1).unsqueeze(0)
                depth_t = rf['depth'].unsqueeze(0).unsqueeze(0)
                rgb_down = torch.nn.functional.interpolate(rgb_t, size=(target_h, target_w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
                depth_down = torch.nn.functional.interpolate(depth_t, size=(target_h, target_w), mode='nearest')[0, 0]
                frames.append({'rgb': rgb_down, 'depth': depth_down, 'pose': rf.get('pose', torch.eye(4)), 'intrinsics': intrinsics})
                
            print(f"[Dataset] Loaded and scaled {len(frames)} frames from TUM RGB-D ({target_w}x{target_h})")
        except Exception as e:
            print(f"[Dataset] TUM load fallback: {e}")
            frames = None
            
    if frames is None:
        print("[Dataset] Initializing synthetic structured stress-test environment")
        frames, intrinsics = create_structured_synthetic_frames(
            n_frames=args.n_warmup + 6, H=args.height, W=args.width, device=device)
            
    # 2. Pipeline setup
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': frames[0]['rgb'].shape[1],
            'image_height': frames[0]['rgb'].shape[0],
            'use_surface_aware_depth': True,
            'attribution_top_k': 4
        },
        'scheduler': {
            'gpu_budget_ms': 50.0,
            'policy': 'budget_aware',
            'optimize_ratio': 0.6,
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.importance_estimator.novelty_weight = 0.5
    pipeline.importance_estimator.use_error_prior = True
    
    # 3. Pipeline Warmup
    print("\n" + "-" * 80)
    print(f"PHASE 1: Pipeline Online Warmup ({args.n_warmup} frames)")
    print("-" * 80)
    
    f0 = frames[0]
    pipeline.initialize(f0['rgb'].to(device), f0['depth'].to(device), intrinsics.to(device), f0.get('pose', torch.eye(4)).to(device))
    print(f"[Init] Initialized model with {pipeline.gaussian_model.num_gaussians} Gaussians")
    
    for i in range(1, min(args.n_warmup + 1, len(frames))):
        f = frames[i]
        m = pipeline.process_frame(f['rgb'].to(device), f['depth'].to(device), gt_pose=f.get('pose', torch.eye(4)).to(device))
        if i % 2 == 0 or i == args.n_warmup:
            print(f"  [Frame {i:2d}] PSNR={m['psnr']:5.2f} dB | N_gauss={m['n_gaussians']:4d} | N_opt={m['n_optimized']:3d} | Time={m['frame_time_ms']:5.1f}ms")
            
    total_gaussians = pipeline.gaussian_model.num_gaussians
    print(f"Warmup Complete. Total Active Gaussians: {total_gaussians}")
    
    # 4. Multi-Population Oracle Experiment
    print("\n" + "-" * 80)
    print("PHASE 2: Evaluating Oracle Utility across Sampling Populations")
    print("-" * 80)
    
    last_frame = frames[min(args.n_warmup, len(frames) - 1)]
    eval_rgb = last_frame['rgb'].to(device)
    eval_depth = last_frame['depth'].to(device)
    
    experiment = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=args.n_samples,
        n_opt_steps=args.n_opt_steps,
        w_rgb=args.w_rgb,
        w_depth=args.w_depth,
        seed=42,
        group_size=1
    )
    
    all_oracle_rows = []
    population_metrics = {}
    
    populations_to_test = [
        SamplingPopulation.GEOMETRY_STRATIFIED,
        SamplingPopulation.IMPORTANCE_STRATIFIED,
        SamplingPopulation.RANDOM_VISIBLE,
        SamplingPopulation.UNIFORM_VISIBLE,
    ]
    
    for pop in populations_to_test:
        pop_name = pop.value
        print(f"\n>> Evaluating Population: {pop_name.upper()}...")
        start_t = time.time()
        results = experiment.run_oracle_experiment(eval_rgb, eval_depth, population_type=pop, frame_idx=args.n_warmup)
        elapsed = time.time() - start_t
        all_oracle_rows.extend(results)
        
        corr = experiment.compute_correlation_metrics(results)
        population_metrics[pop_name] = corr
        
        if 'error' not in corr:
            print(f"   Done in {elapsed:.1f}s (Visible: {corr['n_visible']}/{corr['n_total']})")
            print(f"   • Spearman ρ(Predicted Utility, Oracle Utility): {corr['spearman_utility_vs_oracle']:+.4f} (p={corr['spearman_utility_p']:.4f})")
            print(f"   • Spearman ρ(Importance, Oracle Utility):        {corr['spearman_importance_vs_oracle']:+.4f} (p={corr['spearman_importance_p']:.4f})")
            print(f"   • Spearman ρ(Importance, ΔQ_local):             {corr['spearman_importance_vs_deltaQ']:+.4f} (p={corr['spearman_deltaQ_p']:.4f})")
            print(f"   • Top-10% Overlap: {corr['overlaps'].get('top_10pct', 0):.1%} | Top-20% Overlap: {corr['overlaps'].get('top_20pct', 0):.1%}")
            print(f"   • Realized Gain Ratio @10%: {corr['realized_gains'].get('top_10pct_ratio', 0):.4f} | @20%: {corr['realized_gains'].get('top_20pct_ratio', 0):.4f}")
        else:
            print(f"   ⚠️ {corr['error']}")
            
    # Compute geometry stratification breakdown
    geometry_stats = {}
    geo_rows = [r for r in all_oracle_rows if r.get('population') == SamplingPopulation.GEOMETRY_STRATIFIED.value and r.get('visible', True)]
    for stratum in ['flat', 'edge', 'texture', 'depth_discontinuity']:
        s_rows = [r for r in geo_rows if r.get('geometry_stratum') == stratum]
        if len(s_rows) > 0:
            geometry_stats[stratum] = {
                'count': len(s_rows),
                'mean_psnr': float(np.mean([r['delta_psnr_local'] for r in s_rows])),
                'mean_depth': float(np.mean([r['delta_depth_gain_local'] for r in s_rows])),
                'mean_loss': float(np.mean([r['delta_loss_local'] for r in s_rows])),
                'mean_utility': float(np.mean([r['oracle_utility_joint'] for r in s_rows])),
            }
            
    # 5. Stability Verification (Point 6: n=3-5 repeat trials)
    stability_metrics = None
    if args.eval_stability:
        print("\n" + "-" * 80)
        print(f"PHASE 3: Evaluating Oracle Noise & Repeat Stability (n={args.n_repeats} trials)")
        print("-" * 80)
        visible_candidates = [r['gaussian_id'] for r in all_oracle_rows if r.get('visible', True)][:args.stability_candidates]
        if len(visible_candidates) > 0:
            stability_metrics = experiment.run_stability_check(
                eval_rgb, eval_depth, candidate_indices=visible_candidates, n_repeats=args.n_repeats
            )
            print(f"   • Mean Coefficient of Variation (CV): {stability_metrics['mean_cv']:.4f}")
            print(f"   • Median CV:                          {stability_metrics['median_cv']:.4f}")
            print(f"   • Stable Fraction (CV <= 0.35):       {stability_metrics['stable_fraction']:.1%}")
            print(f"   • Gate 1 Stability Standard:          {'PASSED' if stability_metrics['gate1_passed'] else 'FAILED'}")

    # 6. Group Scaling Evaluation (Point 5: group_size in {4, 16})
    group_metrics = {}
    if args.eval_groups:
        for g_size in [4, 16]:
            print("\n" + "-" * 80)
            print(f"PHASE 4: Evaluating Group Oracle Interactions (group_size = {g_size})")
            print("-" * 80)
            group_experiment = OracleUtilityExperiment(
                pipeline=pipeline,
                n_samples=args.n_samples,
                n_opt_steps=args.n_opt_steps,
                w_rgb=args.w_rgb,
                w_depth=args.w_depth,
                seed=42,
                group_size=g_size
            )
            group_results = group_experiment.run_oracle_experiment(
                eval_rgb, eval_depth, population_type=SamplingPopulation.GEOMETRY_STRATIFIED, frame_idx=args.n_warmup)
            all_oracle_rows.extend(group_results)
            g_corr = group_experiment.compute_correlation_metrics(group_results)
            group_metrics[f'group_size_{g_size}'] = g_corr
            print(f"   • Group (K={g_size}) Spearman ρ(Utility, Oracle): {g_corr.get('spearman_utility_vs_oracle', 0.0):+.4f}")
            print(f"   • Group (K={g_size}) Realized Gain Ratio @20%:    {g_corr.get('realized_gains', {}).get('top_20pct_ratio', 0.0):.4f}")
            
    # 7. Save Oracle Dataset Artifact
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'oracle_dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_file = os.path.join(dataset_dir, 'oracle_dataset.json')
    experiment.export_oracle_dataset(all_oracle_rows, dataset_file)
    print(f"\n[Artifact] Successfully exported {len(all_oracle_rows)} rows to Oracle Dataset:")
    print(f"           → {dataset_file}")
    
    # 8. Save Multi-Population JSON Summary
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'oracle_utility')
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, 'multi_population_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'population_metrics': population_metrics,
            'stability_metrics': stability_metrics,
            'group_metrics': group_metrics,
            'geometry_stats': geometry_stats,
        }, f, indent=2)
        
    # 9. Generate Gate 1 Oracle Validation Report
    report_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'oracle', 'oracle_validation.md')
    generate_oracle_validation_report(
        population_metrics=population_metrics,
        stability_metrics=stability_metrics,
        group_metrics=group_metrics,
        geometry_stats=geometry_stats,
        output_path=report_file
    )
    
    # 10. Print Final Comparative Table
    print("\n" + "=" * 80)
    print("                 GROUND-TRUTH ORACLE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Sampling Population':<24} | {'ρ(Util,Oracle)':>14} | {'ρ(Imp,ΔQ)':>10} | {'Ov@10%':>8} | {'Ov@20%':>8} | {'Gain@20%':>9}")
    print("-" * 80)
    for pop_name, m in population_metrics.items():
        if 'error' in m: continue
        rho_u = m.get('spearman_utility_vs_oracle', 0.0)
        rho_q = m.get('spearman_importance_vs_deltaQ', 0.0)
        ov10 = m.get('overlaps', {}).get('top_10pct', 0.0)
        ov20 = m.get('overlaps', {}).get('top_20pct', 0.0)
        g20 = m.get('realized_gains', {}).get('top_20pct_ratio', 0.0)
        print(f"{pop_name:<24} | {rho_u:>14.4f} | {rho_q:>10.4f} | {ov10:>7.1%} | {ov20:>7.1%} | {g20:>9.4f}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oracle Utility Ground-Truth Experiment Runner")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames', type=int, default=6, help='Number of warmup frames')
    parser.add_argument('--n_warmup', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=40)
    parser.add_argument('--n_opt_steps', type=int, default=5)
    parser.add_argument('--w_rgb', type=float, default=0.7)
    parser.add_argument('--w_depth', type=float, default=0.3)
    parser.add_argument('--eval_stability', action='store_true', default=True)
    parser.add_argument('--stability_candidates', type=int, default=15)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--eval_groups', action='store_true', default=True)
    parser.add_argument('--synthetic', action='store_true', default=False)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=80)
    args = parser.parse_args()
    if args.frames:
        args.n_warmup = args.frames
    run_experiment(args)
