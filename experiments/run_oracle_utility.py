#!/usr/bin/env python3
"""Oracle Utility Comprehensive Research Experiment Runner.

Evaluates ground-truth marginal utility U_i^oracle = ΔQ_{local,i} / (Cost_i + ε)
incorporating:
  1. Combined RGB-D local quality improvement (w_rgb=0.7, w_depth=0.3)
  2. Unbiased population comparisons (IMPORTANCE_STRATIFIED, RANDOM_VISIBLE, UNIFORM_VISIBLE)
  3. Group oracle scaling (group_size = 1, 4)
  4. Precise trial cost vs modeled marginal cost separation
  5. Automatic Oracle Dataset export to results/oracle_dataset/ for Learned Utility training
"""
import os
import sys
import json
import time
import argparse
import torch
import numpy as np

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


def run_experiment(args):
    device = args.device
    print("=" * 80)
    print("       ORACLE UTILITY & MARGINAL VALUE EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"Device: {device} | Warmup: {args.n_warmup} frames | Samples/Pop: {args.n_samples}")
    print(f"Quality Formulation: ΔQ = {args.w_rgb:.2f} · ΔPSNR + {args.w_depth:.2f} · (10 · ΔDepthGain)")
    
    # 1. Dataset setup
    frames, intrinsics = None, None
    tum_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'TUM', 'rgbd_dataset_freiburg1_desk')
    if os.path.exists(tum_path) and not args.synthetic:
        try:
            from datasets.tum_dataset import TUMDataset
            dataset = TUMDataset(tum_path, max_frames=args.n_warmup + 4, stride=5)
            raw_frames = [dataset[i] for i in range(len(dataset))]
            intrinsics = dataset.intrinsics.clone()
            
            # Downsample frames to target resolution for fast CPU execution
            target_h, target_w = args.height, args.width
            scale_x = target_w / raw_frames[0]['rgb'].shape[1]
            scale_y = target_h / raw_frames[0]['rgb'].shape[0]
            intrinsics[0, :] *= scale_x
            intrinsics[1, :] *= scale_y
            
            frames = []
            for rf in raw_frames:
                rgb_t = rf['rgb'].permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                depth_t = rf['depth'].unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
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
            n_frames=args.n_warmup + 4, H=args.height, W=args.width, device=device)
            
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
            
    # 5. Group Oracle Scaling Evaluation (group_size = 4)
    if args.eval_groups:
        print("\n" + "-" * 80)
        print("PHASE 3: Evaluating Group Oracle Interactions (group_size = 4)")
        print("-" * 80)
        group_experiment = OracleUtilityExperiment(
            pipeline=pipeline,
            n_samples=args.n_samples,
            n_opt_steps=args.n_opt_steps,
            w_rgb=args.w_rgb,
            w_depth=args.w_depth,
            seed=42,
            group_size=4
        )
        group_results = group_experiment.run_oracle_experiment(
            eval_rgb, eval_depth, population_type=SamplingPopulation.IMPORTANCE_STRATIFIED, frame_idx=args.n_warmup)
        all_oracle_rows.extend(group_results)
        group_corr = group_experiment.compute_correlation_metrics(group_results)
        population_metrics['group_size_4'] = group_corr
        print(f"   • Group (K=4) Spearman ρ(Utility, Oracle): {group_corr.get('spearman_utility_vs_oracle', 0.0):+.4f}")
        print(f"   • Group (K=4) Realized Gain Ratio @20%:    {group_corr.get('realized_gains', {}).get('top_20pct_ratio', 0.0):.4f}")
        
    # 6. Save Oracle Dataset Artifact
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'oracle_dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_file = os.path.join(dataset_dir, 'oracle_dataset.json')
    experiment.export_oracle_dataset(all_oracle_rows, dataset_file)
    print(f"\n[Artifact] Successfully exported {len(all_oracle_rows)} rows to Oracle Dataset:")
    print(f"           → {dataset_file}")
    
    # 7. Save Summary Metrics
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'oracle_utility')
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, 'multi_population_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(population_metrics, f, indent=2)
    print(f"[Metrics] Saved multi-population evaluation summary to {metrics_file}")
    
    # 8. Print Comparative Table
    print("\n" + "=" * 80)
    print("                 MULTI-POPULATION ORACLE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Sampling Population':<24} | {'ρ(Util,Oracle)':>14} | {'ρ(Imp,ΔQ)':>10} | {'Ov@10%':>8} | {'Ov@20%':>8} | {'Gain@20%':>9}")
    print("-" * 80)
    for pop_name, m in population_metrics.items():
        if 'error' in m:
            continue
        rho_u = m['spearman_utility_vs_oracle']
        rho_q = m['spearman_importance_vs_deltaQ']
        ov10 = m.get('overlaps', {}).get('top_10pct', 0)
        ov20 = m.get('overlaps', {}).get('top_20pct', 0)
        g20 = m.get('realized_gains', {}).get('top_20pct_ratio', 0)
        print(f"{pop_name:<24} | {rho_u:>14.4f} | {rho_q:>10.4f} | {ov10:>7.1%} | {ov20:>7.1%} | {g20:>9.4f}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Oracle Utility Multi-Population Experiment Runner")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--frames', type=int, default=8, help='Number of warmup frames')
    parser.add_argument('--n_warmup', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=60)
    parser.add_argument('--n_opt_steps', type=int, default=10)
    parser.add_argument('--w_rgb', type=float, default=0.7)
    parser.add_argument('--w_depth', type=float, default=0.3)
    parser.add_argument('--eval_groups', action='store_true', default=True)
    parser.add_argument('--synthetic', action='store_true', default=False)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=80)
    args = parser.parse_args()
    if args.frames:
        args.n_warmup = args.frames
    run_experiment(args)
