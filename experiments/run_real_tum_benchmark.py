#!/usr/bin/env python3
"""R37/R38 — Real Dataset Benchmark on TUM RGB-D (fr1/desk) with Multi-Seed Evaluation.

Evaluates Policies across multiple random seeds (seeds = [42, 43, 44]):
  - Full (Upper Bound)
  - Random
  - Error-Only
  - Error × Influence
  - Top-K Importance
  - Ours (Budget-Aware Knapsack)

Outputs:
  - results/real_tum/tum_fr1_desk_results.json
  - results/real_tum/tum_fr1_desk_summary.md
  - Reports Mean ± Std for PSNR, Depth L1, and Compute Latency.
"""
import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.scheduler import OptimizationPolicy


def load_tum_frames(data_path: str, max_frames: int = 10, H: int = 48, W: int = 64, device: str = 'cpu'):
    """Load TUM fr1/desk frames and scale to benchmark resolution."""
    dataset = TUMDataset(data_path, max_frames=max_frames, camera='freiburg1')
    frames = []
    
    # Scale intrinsics to (W, H)
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H
    
    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0, dataset.cx * scale_x],
        [0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0, 0, 1.0]
    ], dtype=torch.float32, device=device)
    
    for i in range(len(dataset)):
        item = dataset[i]
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, H, W)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
        
        # Resize
        rgb_scaled = torch.nn.functional.interpolate(rgb, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(depth, size=(H, W), mode='nearest').squeeze(0).squeeze(0)
        
        frames.append({
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device)
        })
        
    return frames, intrinsics


def run_tum_policy_seed(policy_name: str, seed: int, frames, intrinsics, budget_ms: float = 8.0, device: str = 'cpu'):
    """Run a single policy on TUM frames with specified seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    is_full = (policy_name == 'full')
    ratio = 1.0 if is_full else 0.40
    
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
            'gpu_budget_ms': budget_ms,
            'policy': policy_name,
            'optimize_ratio': ratio,
        },
        'densification': {
            'max_new_per_frame': 60,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])
    
    psnrs, depth_l1s, opt_times = [], [], []
    
    for f in frames[1:]:
        res = pipeline.process_frame(f['rgb'], f['depth'], gt_pose=f['pose'])
        psnrs.append(res['psnr'])
        depth_l1s.append(res['depth_l1'])
        opt_times.append(res['opt_time_ms'])
        
    return {
        'avg_psnr': float(np.mean(psnrs)),
        'avg_depth_l1': float(np.mean(depth_l1s)),
        'avg_opt_ms': float(np.mean(opt_times)),
        'final_gaussians': pipeline.gaussian_model.num_gaussians
    }


def run_real_tum_benchmark(data_path: str, n_frames: int = 8, seeds=[42, 43, 44], device: str = 'cpu'):
    print("=" * 85)
    print("      R37/R38: REAL DATASET BENCHMARK — TUM RGB-D (fr1/desk)")
    print("=" * 85)
    print(f"Device: {device} | Frames: {n_frames} | Seeds: {seeds}\n")
    
    frames, intrinsics = load_tum_frames(data_path, max_frames=n_frames, H=48, W=64, device=device)
    print(f"Loaded {len(frames)} TUM fr1/desk frames at 64x48 resolution.\n")
    
    policies = ['full', 'random', 'error_only', 'error_influence', 'top_k', 'ours']
    results_by_policy = {}
    
    for pol in policies:
        print(f">> Evaluating Policy: {pol:<16} across {len(seeds)} seeds...")
        seed_results = []
        for s in seeds:
            res = run_tum_policy_seed(pol, s, frames, intrinsics, budget_ms=8.0, device=device)
            seed_results.append(res)
            
        psnr_vals = [r['avg_psnr'] for r in seed_results]
        depth_vals = [r['avg_depth_l1'] for r in seed_results]
        opt_vals = [r['avg_opt_ms'] for r in seed_results]
        
        results_by_policy[pol] = {
            'psnr_mean': float(np.mean(psnr_vals)),
            'psnr_std': float(np.std(psnr_vals)),
            'depth_mean': float(np.mean(depth_vals)),
            'depth_std': float(np.std(depth_vals)),
            'opt_ms_mean': float(np.mean(opt_vals)),
            'opt_ms_std': float(np.std(opt_vals)),
            'seed_runs': seed_results
        }
        
    print("\n" + "=" * 85)
    print("                 TUM RGB-D (fr1/desk) MULTI-SEED SUMMARY")
    print("=" * 85)
    print(f"{'Policy':<18} | {'PSNR (dB) ↑':<18} | {'Depth L1 (m) ↓':<18} | {'Opt Time (ms) ↓':<18}")
    print("-" * 85)
    
    for pol in policies:
        d = results_by_policy[pol]
        psnr_str = f"{d['psnr_mean']:.2f} ± {d['psnr_std']:.2f}"
        depth_str = f"{d['depth_mean']:.4f} ± {d['depth_std']:.4f}"
        opt_str = f"{d['opt_ms_mean']:.1f} ± {d['opt_ms_std']:.1f}"
        print(f"{pol:<18} | {psnr_str:<18} | {depth_str:<18} | {opt_str:<18}")
    print("=" * 85 + "\n")
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'real_tum')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'tum_fr1_desk_results.json'), 'w') as f:
        json.dump(results_by_policy, f, indent=2)
        
    with open(os.path.join(save_dir, 'tum_fr1_desk_summary.md'), 'w') as f:
        f.write("# Real TUM RGB-D (fr1/desk) Multi-Seed Benchmark Summary\n\n")
        f.write(f"Evaluated across {len(seeds)} random seeds ({seeds}) on real sensor data with noisy depth & camera motion.\n\n")
        f.write("| Policy | PSNR (dB) ↑ | Depth L1 (m) ↓ | Opt Time (ms) ↓ |\n")
        f.write("|:---:|:---:|:---:|:---:|\n")
        for pol in policies:
            d = results_by_policy[pol]
            f.write(f"| **{pol}** | {d['psnr_mean']:.2f} ± {d['psnr_std']:.2f} | {d['depth_mean']:.4f} ± {d['depth_std']:.4f} | {d['opt_ms_mean']:.1f} ± {d['opt_ms_std']:.1f} ms |\n")
        f.write("\n")
        
    print(f"Artifacts saved to:")
    print(f"  - {os.path.join(save_dir, 'tum_fr1_desk_results.json')}")
    print(f"  - {os.path.join(save_dir, 'tum_fr1_desk_summary.md')}")


def main():
    parser = argparse.ArgumentParser(description="TUM Real RGB-D Benchmark")
    parser.add_argument('--data_path', type=str, default='datasets/TUM/rgbd_dataset_freiburg1_desk')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    run_real_tum_benchmark(args.data_path, n_frames=args.frames, device=args.device)


if __name__ == '__main__':
    main()
