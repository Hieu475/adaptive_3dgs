"""Benchmark full pipeline performance.

Measures end-to-end pipeline timing including:
- Initialization
- Per-frame processing
- Memory usage over time

Usage:
    python benchmarks/benchmark_pipeline.py
    python benchmarks/benchmark_pipeline.py --n_frames 100 --device cuda
"""
import sys
import os
import time
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def benchmark_pipeline(n_frames: int = 30, device: str = 'cpu', H: int = 120, W: int = 160):
    """Run synthetic pipeline benchmark."""
    from research.pipeline import OnlineReconstructionPipeline
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 50000},
        'rendering': {'tile_size': 16, 'image_width': W, 'image_height': H},
        'losses': {'weight_color': 1.0, 'weight_depth': 0.5, 'weight_normal': 0.1, 'weight_regularization': 0.01},
        'importance': {'depth_error': 1.0, 'color_error': 1.0, 'normal_error': 0.5, 'visibility': 0.1, 'temporal': 0.5, 'screen_space': 0.2},
        'scheduler': {'gpu_budget_ms': 16.6, 'tier_thresholds': [0.8, 0.5, 0.2], 'optimize_every_n_frames': 5},
        'densification': {'max_new_per_frame': 200, 'error_threshold_color': 0.1, 'error_threshold_depth': 0.05, 'transmission_threshold': 0.5},
        'training': {'learning_rate': {'position': 1.6e-4, 'scale': 5e-3, 'rotation': 1e-3, 'opacity': 5e-2, 'sh': 2.5e-3}},
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    intrinsics = torch.tensor([[100., 0., W/2.], [0., 100., H/2.], [0., 0., 1.]])
    
    # Generate synthetic data
    print(f"Generating {n_frames} synthetic frames ({H}x{W})...")
    
    # Initialize
    rgb0 = torch.rand(H, W, 3)
    depth0 = torch.rand(H, W) * 3 + 1
    
    t0 = time.perf_counter()
    pipeline.initialize(rgb0, depth0, intrinsics)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Initialization: {init_time:.1f} ms, {pipeline.gaussian_model.num_gaussians} Gaussians")
    
    # Process frames
    frame_times = []
    gaussian_counts = []
    
    print(f"\nProcessing {n_frames} frames...")
    for i in range(1, n_frames + 1):
        rgb = torch.rand(H, W, 3)
        depth = torch.rand(H, W) * 3 + 1
        pose = torch.eye(4)
        pose[0, 3] = 0.01 * i
        
        t0 = time.perf_counter()
        metrics = pipeline.process_frame(rgb, depth, gt_pose=pose)
        frame_time = (time.perf_counter() - t0) * 1000
        
        frame_times.append(frame_time)
        gaussian_counts.append(metrics['n_gaussians'])
        
        if i % 10 == 0 or i == n_frames:
            print(f"  Frame {i:4d}: {frame_time:.1f} ms | "
                  f"Gaussians: {metrics['n_gaussians']} | "
                  f"PSNR: {metrics['psnr']:.1f} dB")
    
    # Summary
    frame_times = np.array(frame_times)
    summary = {
        'device': device,
        'n_frames': n_frames,
        'image_size': f'{W}x{H}',
        'init_time_ms': float(init_time),
        'mean_frame_ms': float(frame_times.mean()),
        'std_frame_ms': float(frame_times.std()),
        'p50_frame_ms': float(np.percentile(frame_times, 50)),
        'p95_frame_ms': float(np.percentile(frame_times, 95)),
        'avg_fps': float(1000.0 / frame_times.mean()),
        'final_gaussians': int(gaussian_counts[-1]),
        'peak_gaussians': int(max(gaussian_counts)),
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        summary['gpu_mem_mb'] = float(torch.cuda.max_memory_allocated() / 1024**2)
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_frames', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='artifacts/pipeline_benchmark.json')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PIPELINE BENCHMARK")
    print("="*60)
    
    summary = benchmark_pipeline(n_frames=args.n_frames, device=args.device)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.2f}")
        else:
            print(f"  {k:25s}: {v}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")
