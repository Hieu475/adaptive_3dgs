"""Script to run full Milestone R5 Densification Ablation Benchmark.

Usage:
    python scripts/run_densification_ablation.py [--device cpu|cuda] [--frames 4]
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_densification import (
    run_full_densification_ablation,
    format_densification_table,
)
from scripts.run_policy_ablation import generate_synthetic_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="Milestone R5 Densification Ablation Benchmark")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=3, help='Number of frames to evaluate')
    args = parser.parse_args()

    print(f"Generating {args.frames} synthetic RGB-D frames for densification ablation...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames)

    print("\nRunning Milestone R5 Densification Strategy & Adaptive Threshold Ablation...")
    ablation = run_full_densification_ablation(frames, intrinsics, device=args.device)

    table = format_densification_table(ablation)
    print("\n" + "=" * 80)
    print("           MILESTONE R5: ADAPTIVE DENSIFICATION ABLATION RESULTS")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
