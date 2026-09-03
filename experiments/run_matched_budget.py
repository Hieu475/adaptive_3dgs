#!/usr/bin/env python3
"""Primary Research Experiment — Rigorous Matched-Budget Benchmark on Real 3DGS.

Executes:
    1. Benchmark A (Open-Loop Calibrated):
       Relative Budgets: B_{rel} ∈ {10%, 20%, 40%, 60%, 80%} of Full Reference Optimization Cost
       Evaluation strictly with frozen calibrated K on held-out evaluation frames.
    2. Benchmark B (Closed-Loop Systems):
       Real-time deadline adherence under 10, 15, 20, 30 ms budgets.
    3. Reproducibility & Provenance Bundling (Section XXXII).

Metrics:
    - PSNR (dB) [Mean & 95% Bootstrap CI], SSIM, Depth L1 (m) [Mean & 95% CI]
    - Latency: p50, p95, p99, Latency Jitter (ms)
    - Budget Utilization (T/B) and Budget Violation Rate (%)
    - Gain Efficiency: GE@B = (Q(B) - Q_{random}(B)) / (Q_{oracle}(B) - Q_{random}(B) + ε)
"""
import sys
import os
import torch
import numpy as np
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.matched_budget_benchmark import MatchedBudgetBenchmark, SchedulerMetrics
from research.schema import (
    ExperimentMetadata,
    QualityMetrics,
    LatencyMetrics,
    MemoryMetrics,
    SelectionMetrics,
    ExperimentMetrics,
    ExperimentResult,
)
from research.reproducibility import (
    get_git_commit,
    get_hardware_info,
    set_seed,
    save_experiment_bundle,
    DatasetManifest,
)


def create_benchmark_frames(n_frames: int = 15, H: int = 64, W: int = 80):
    """Generate structured synthetic benchmark frames."""
    fx, fy = 160.0, 160.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    frames = []
    
    for t in range(n_frames):
        angle = t * 0.03
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0
        
        # Texture patch
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 8 + j // 8) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.9, 0.2, 0.1])
                else:
                    rgb[i, j] = torch.tensor([0.1, 0.8, 0.2])
        depth[:H//2, :W//2] = 2.0
        
        # Foreground box
        box_h = slice(H // 2 + 5, H - 5)
        box_w = slice(W // 4, 3 * W // 4)
        rgb[box_h, box_w] = torch.tensor([0.7, 0.3, 0.5])
        depth[box_h, box_w] = 1.0
        
        rgb = (rgb + 0.02 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.1
        
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
        
    return frames, intrinsics


def real_pipeline_factory(config_overrides, device):
    """Instantiate OnlineReconstructionPipeline with budget overrides."""
    budget_ms = config_overrides.get('scheduler', {}).get('gpu_budget_ms', 20.0)
    policy = config_overrides.get('scheduler', {}).get('policy', 'budget_aware')
    top_k = config_overrides.get('scheduler', {}).get('top_k', None)
    
    base_config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': 80,
            'image_height': 64,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4
        },
        'scheduler': {
            'gpu_budget_ms': budget_ms,
            'policy': policy,
            'top_k': top_k,
        },
        'densification': {
            'max_new_per_frame': 60,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    pipeline = OnlineReconstructionPipeline(config=base_config, device=device)
    pipeline.importance_estimator.novelty_weight = 0.5
    pipeline.importance_estimator.use_error_prior = True
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Matched-Budget Primary Benchmark")
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--frames', type=int, default=None, help='Alias for n_frames')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--synthetic', action='store_true', default=False)
    args = parser.parse_args()
    if args.frames is not None:
        args.n_frames = args.frames
        
    set_seed(args.seed)
    
    print("=" * 85)
    print("      FINAL MATCHED-BUDGET SCIENTIFIC BENCHMARK (SECTIONS X–XVIII)")
    print("=" * 85)
    print(f"Device: {args.device} | Total Frames: {args.n_frames} | Seed: {args.seed}")
    
    # Dataset setup
    frames, intrinsics = None, None
    tum_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'TUM', 'rgbd_dataset_freiburg1_desk')
    dataset_name = "Synthetic"
    scene_name = "benchmark_box"
    
    if os.path.exists(tum_path) and not args.synthetic:
        try:
            from datasets.tum_dataset import TUMDataset
            dataset = TUMDataset(tum_path, max_frames=args.n_frames + 4, stride=3)
            raw_frames = [dataset[i] for i in range(len(dataset))]
            intrinsics = dataset.intrinsics.clone()
            
            target_h, target_w = 64, 80
            scale_x = target_w / raw_frames[0]['rgb'].shape[1]
            scale_y = target_h / raw_frames[0]['rgb'].shape[0]
            intrinsics[0, :] *= scale_x
            intrinsics[1, :] *= scale_y
            
            frames = []
            for rf in raw_frames[:args.n_frames]:
                rgb_t = rf['rgb'].permute(2, 0, 1).unsqueeze(0)
                depth_t = rf['depth'].unsqueeze(0).unsqueeze(0)
                rgb_down = torch.nn.functional.interpolate(rgb_t, size=(target_h, target_w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
                depth_down = torch.nn.functional.interpolate(depth_t, size=(target_h, target_w), mode='nearest')[0, 0]
                frames.append({'rgb': rgb_down, 'depth': depth_down, 'pose': rf.get('pose', torch.eye(4)), 'intrinsics': intrinsics})
                
            dataset_name = "TUM-RGBD"
            scene_name = "freiburg1_desk"
            print(f"[Dataset] Loaded {len(frames)} frames from TUM RGB-D ({target_w}x{target_h})")
        except Exception as e:
            print(f"[Dataset] TUM load fallback: {e}")
            frames = None
            
    if frames is None:
        print("[Dataset] Using synthetic structured benchmark frames")
        frames, intrinsics = create_benchmark_frames(n_frames=args.n_frames)
    
    # 1. Dataset Manifest (Section XXXIII)
    manifest = DatasetManifest(
        dataset=dataset_name,
        sequence=scene_name,
        frames=list(range(len(frames))),
        rgb_resolution=(frames[0]['rgb'].shape[0], frames[0]['rgb'].shape[1]),
        depth_scale=5000.0 if dataset_name == "TUM-RGBD" else 1.0,
        pose_source="ground_truth",
    )
    
    relative_budgets = [0.10, 0.20, 0.40, 0.60, 0.80]
    benchmark = MatchedBudgetBenchmark(
        relative_budgets=relative_budgets, device=args.device, seed=args.seed
    )
    
    start_time = time.time()
    
    # Run Benchmark A: Open-Loop Calibrated
    open_loop_results, meta = benchmark.run_benchmark_a_open_loop(
        real_pipeline_factory, frames, intrinsics, calib_fraction=0.35
    )
    
    # Run Benchmark B: Closed-Loop Systems
    eval_frames = frames[meta['n_calib_frames']:]
    closed_loop_results = benchmark.run_benchmark_b_closed_loop(
        real_pipeline_factory, eval_frames, intrinsics, target_budgets_ms=[10.0, 15.0, 20.0, 30.0]
    )
    
    total_time = time.time() - start_time
    print(f"\nComplete Matched-Budget Evaluation finished in {total_time:.1f}s")
    
    table_md = benchmark.format_markdown_table(open_loop_results, meta)
    print("\n" + table_md)
    
    # Save artifacts in results/matched_budget/
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'matched_budget')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, 'results_table.md')
    with open(save_path, 'w') as f:
        f.write(table_md)
        f.write("\n")
        
    json_path = os.path.join(save_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(open_loop_results, f, indent=2)
        
    # Export Pareto curve data
    pareto_csv = os.path.join(save_dir, 'pareto_quality_vs_compute.csv')
    with open(pareto_csv, 'w') as f:
        f.write("relative_budget,budget_ms,policy,measured_compute_ms,p50_ms,p95_ms,jitter,utilization,violation_rate,psnr,psnr_ci_low,psnr_ci_high,depth_l1,gain_efficiency\n")
        for r in open_loop_results:
            ci = r.get('psnr_ci95', (r['avg_psnr'], r['avg_psnr']))
            f.write(
                f"{r.get('relative_budget', 1.0)},{r.get('budget_ms', 0.0):.2f},"
                f"{r['policy_name']},{r['measured_compute_ms']:.3f},{r['p50_ms']:.3f},"
                f"{r['p95_ms']:.3f},{r['jitter']:.3f},{r.get('budget_utilization', 0.0)*100:.1f}%,"
                f"{r.get('violation_rate', 0.0):.1f}%,{r['avg_psnr']:.3f},{ci[0]:.3f},{ci[1]:.3f},"
                f"{r['avg_depth_l1']:.4f},{r.get('gain_efficiency', 0.0):.4f}\n"
            )
            
    # Save Immutable-ish Run Bundle (Section XXXII)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"run_{time_tag}_seed{args.seed}")
    
    ours_eval = next((r for r in open_loop_results if r['policy_name'] == 'ours' and r.get('relative_budget') == 0.40), open_loop_results[0])
    ci = ours_eval.get('psnr_ci95', (ours_eval['avg_psnr'], ours_eval['avg_psnr']))
    
    meta_obj = ExperimentMetadata(
        name="matched_budget_benchmark",
        version="1.0.0",
        git_commit=get_git_commit(),
        timestamp=datetime.now().isoformat(),
        seed=args.seed,
        dataset=dataset_name,
        scene=scene_name,
        frame_range=[0, len(frames)],
        hardware=get_hardware_info(),
        config={'relative_budgets': relative_budgets, 'device': args.device},
    )
    metrics_obj = ExperimentMetrics(
        quality=QualityMetrics(
            psnr_mean=ours_eval['avg_psnr'],
            psnr_std=ours_eval.get('psnr_std', 0.0),
            psnr_ci95=ci,
            depth_l1_mean=ours_eval['avg_depth_l1'],
            depth_l1_std=ours_eval.get('depth_l1_std', 0.0),
        ),
        latency=LatencyMetrics(
            mean_ms=ours_eval['measured_compute_ms'],
            p50_ms=ours_eval['p50_ms'],
            p95_ms=ours_eval['p95_ms'],
            jitter_ms=ours_eval['jitter'],
            violation_rate=ours_eval['violation_rate'],
        ),
        selection=SelectionMetrics(
            gain_efficiency=ours_eval.get('gain_efficiency', 0.0),
        ),
    )
    bundle_result = ExperimentResult(experiment=meta_obj, metrics=metrics_obj)
    save_experiment_bundle(bundle_result, run_dir, markdown_report=table_md)
    manifest.save(os.path.join(run_dir, "dataset_manifest.json"))
    
    print(f"\n[Artifacts] Successfully updated:")
    print(f"  - {save_path}")
    print(f"  - {json_path}")
    print(f"  - {pareto_csv}")
    print(f"  - Bundle: {run_dir}")


if __name__ == "__main__":
    main()
