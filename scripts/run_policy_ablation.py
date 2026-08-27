"""Script to run full Milestone R4 Policy Ablation Benchmark.

Usage:
    python scripts/run_policy_ablation.py [--device cpu|cuda] [--frames 5]
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_policies import (
    run_full_policy_ablation_matrix,
    format_benchmark_table,
)


def generate_synthetic_benchmark_dataset(n_frames: int = 5, H: int = 64, W: int = 64):
    """Generate synthetic RGB-D sequence with textured objects and camera motion."""
    torch.manual_seed(42)
    intrinsics = torch.tensor([
        [60.0, 0.0, float(W // 2)],
        [0.0, 60.0, float(H // 2)],
        [0.0, 0.0, 1.0],
    ])
    
    frames = []
    # Grid coordinates
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    
    for i in range(n_frames):
        # Base textured pattern (checkerboard + gradients)
        pattern = ((x * 4).sin() * (y * 4).cos()).clamp(-1, 1) * 0.5 + 0.5
        color = torch.stack([
            pattern,
            (pattern * 1.5).clamp(0, 1),
            (1.0 - pattern),
        ], dim=-1)
        
        # Ground truth depth with surface discontinuity
        depth = torch.full((H, W), 2.0)
        # Center object closer to camera (depth=1.5)
        center_mask = (x**2 + y**2) < 0.25
        depth[center_mask] = 1.5
        depth += torch.randn(H, W) * 0.01  # small sensor noise
        
        # Camera trajectory (orbiting / panning)
        pose = torch.eye(4)
        pose[0, 3] = (i - n_frames // 2) * 0.04
        pose[1, 3] = (i * 0.02)
        
        frames.append({
            'rgb': color.float(),
            'depth': depth.float(),
            'pose': pose.float(),
        })
        
    return frames, intrinsics


def main():
    parser = argparse.ArgumentParser(description="Milestone R4 Policy Ablation Benchmark")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=4, help='Number of frames to evaluate')
    args = parser.parse_args()

    print(f"Generating {args.frames} synthetic RGB-D frames for policy ablation...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames)

    print("\nRunning Milestone R4 Optimization Policy Ablation Matrix...")
    print("Policies: Full (100%), Binary (RTG-SLAM), Random (10%-75%), Top-K Imp (10%-75%), Budget-Aware (2-16ms)")
    
    ablation = run_full_policy_ablation_matrix(
        frames, intrinsics, ratios=[0.10, 0.25, 0.50, 0.75], device=args.device
    )

    table = format_benchmark_table(ablation)
    print("\n" + "=" * 80)
    print("           MILESTONE R4: OPTIMIZATION POLICY ABLATION RESULTS")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
