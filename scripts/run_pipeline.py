#!/usr/bin/env python3
"""Main entry point to run the Adaptive 3DGS reconstruction pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/replica.yaml --data_path datasets/Replica/office0
    python scripts/run_pipeline.py --config configs/tum_rgbd.yaml --data_path datasets/TUM/rgbd_dataset_freiburg1_desk
    python scripts/run_pipeline.py --synthetic  # Run with synthetic data for testing
"""
import argparse
import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import numpy as np

from research.pipeline import OnlineReconstructionPipeline


def create_synthetic_dataset(n_frames: int = 50, H: int = 120, W: int = 160):
    """Create synthetic RGB-D sequence for testing pipeline.
    
    Generates a simple scene with a colored plane and moving camera.
    """
    intrinsics = torch.tensor([
        [100.0, 0.0, W / 2.0],
        [0.0, 100.0, H / 2.0],
        [0.0, 0.0, 1.0],
    ])
    
    frames = []
    for i in range(n_frames):
        # Simple gradient color image
        u = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
        v = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
        rgb = torch.stack([
            u,  # Red channel varies horizontally
            v,  # Green channel varies vertically
            torch.full((H, W), 0.3 + 0.02 * i),  # Blue changes with frame
        ], dim=-1).clamp(0, 1)
        
        # Depth: plane at z = 2 + slight perturbation
        depth = torch.full((H, W), 2.0) + 0.1 * torch.sin(u * 6.28) + 0.05 * torch.randn(H, W)
        depth = depth.clamp(min=0.1)
        
        # Camera moves slowly along x-axis
        pose = torch.eye(4)
        pose[0, 3] = 0.01 * i  # Small translation
        
        frames.append({
            'rgb': rgb,
            'depth': depth,
            'pose': pose,
            'intrinsics': intrinsics,
        })
    
    return frames, intrinsics


def run_on_dataset(pipeline, dataset, max_frames=None, output_dir=None):
    """Run pipeline on a real dataset."""
    n_frames = min(len(dataset), max_frames or len(dataset))
    
    print(f"\n{'='*60}")
    print(f"Running pipeline on {n_frames} frames")
    print(f"{'='*60}\n")
    
    for i in range(n_frames):
        item = dataset[i]
        
        if i == 0:
            pipeline.initialize(
                rgb=item['rgb'],
                depth=item['depth'],
                intrinsics=item['intrinsics'],
                pose=item.get('pose'),
            )
            continue
        
        metrics = pipeline.process_frame(
            rgb=item['rgb'],
            depth=item['depth'],
            gt_pose=item.get('pose'),
        )
        
        if i % 10 == 0 or i == n_frames - 1:
            print(f"  Frame {i:4d}/{n_frames}: "
                  f"PSNR={metrics['psnr']:.2f} dB | "
                  f"Depth L1={metrics['depth_l1']:.4f} | "
                  f"Gaussians={metrics['n_gaussians']:6d} | "
                  f"Optimized={metrics['n_optimized']:5d} | "
                  f"FPS={metrics['fps']:.1f}")
    
    return pipeline.get_metrics_summary()


def run_synthetic(pipeline, n_frames=50, output_dir=None):
    """Run pipeline on synthetic data."""
    frames, intrinsics = create_synthetic_dataset(n_frames=n_frames)
    
    print(f"\n{'='*60}")
    print(f"Running pipeline on {n_frames} synthetic frames")
    print(f"Image size: {frames[0]['rgb'].shape[0]}x{frames[0]['rgb'].shape[1]}")
    print(f"{'='*60}\n")
    
    # Initialize
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )
    
    # Process frames
    for i in range(1, n_frames):
        metrics = pipeline.process_frame(
            rgb=frames[i]['rgb'],
            depth=frames[i]['depth'],
            gt_pose=frames[i]['pose'],
        )
        
        if i % 10 == 0 or i == n_frames - 1:
            print(f"  Frame {i:4d}/{n_frames}: "
                  f"PSNR={metrics['psnr']:.2f} dB | "
                  f"Depth L1={metrics['depth_l1']:.4f} | "
                  f"Gaussians={metrics['n_gaussians']:6d} | "
                  f"Optimized={metrics['n_optimized']:5d} | "
                  f"FPS={metrics['fps']:.1f}")
    
    return pipeline.get_metrics_summary()


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive 3DGS Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, choices=['replica', 'tum'], default='replica')
    parser.add_argument("--synthetic", action='store_true', help="Run on synthetic data")
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--output_dir", type=str, default='artifacts')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    else:
        config = None
        print(f"Config {args.config} not found, using defaults")
    
    # Create pipeline
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run
    if args.synthetic or args.data_path is None:
        summary = run_synthetic(pipeline, n_frames=args.n_frames, output_dir=output_dir)
    else:
        # Load dataset
        if args.dataset_type == 'replica':
            from datasets.replica_dataset import ReplicaDataset
            dataset = ReplicaDataset(args.data_path, max_frames=args.max_frames)
        elif args.dataset_type == 'tum':
            from datasets.tum_dataset import TUMDataset
            dataset = TUMDataset(args.data_path, max_frames=args.max_frames)
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")
        
        summary = run_on_dataset(pipeline, dataset, max_frames=args.max_frames, output_dir=output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("BASELINE METRICS SUMMARY")
    print(f"{'='*60}")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:25s}: {val:.4f}")
        else:
            print(f"  {key:25s}: {val}")
    
    # Save results
    results_path = output_dir / 'baseline_metrics.json'
    # Convert numpy types for JSON serialization
    json_summary = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in summary.items()}
    with open(results_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save per-frame metrics
    frames_path = output_dir / 'per_frame_metrics.json'
    with open(frames_path, 'w') as f:
        json.dump(pipeline.metrics_history, f, indent=2)
    print(f"Per-frame metrics saved to {frames_path}")


if __name__ == "__main__":
    main()
