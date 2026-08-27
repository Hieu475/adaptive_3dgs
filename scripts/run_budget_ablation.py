"""Script to run full Milestone R6 Latency Budget Benchmark.

Usage:
    python scripts/run_budget_ablation.py [--device cpu|cuda] [--frames 4]
"""
import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_budgets import (
    run_full_budget_matrix,
    format_budget_table,
)
from scripts.run_policy_ablation import generate_synthetic_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="Milestone R6 Latency Budget Benchmark")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=3, help='Number of frames to evaluate')
    args = parser.parse_args()

    print(f"Generating {args.frames} synthetic RGB-D frames for budget ablation...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames)

    print("\nRunning Milestone R6 Latency Budget & Closed-Loop Controller Benchmark...")
    print("Budgets: 2ms, 4ms, 8ms, 16ms, Unconstrained (∞)")
    
    ablation = run_full_budget_matrix(
        frames, intrinsics, budgets=[2.0, 4.0, 8.0, 16.0, None], device=args.device
    )

    table = format_budget_table(ablation)
    print("\n" + "=" * 80)
    print("           MILESTONE R6: LATENCY BUDGET ABLATION RESULTS")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
