"""R13: Budget Controller Sweep & Latency Violation Analysis.

Evaluates hypothesis H4: Closed-loop budget scheduler maintains real-time framerate stability
under strict latency bounds (B ∈ {2ms, 4ms, 8ms, 16ms, Unconstrained}).

Measures:
    - Mean Latency (ms), Std Latency (ms / jitter)
    - P50, P95, P99 Percentile Latency
    - Budget Violation Rate (%)
    - Throughput: Mean FPS and Min FPS (5th percentile)
"""
import os
import sys
import json
import csv
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_budgets import run_full_budget_matrix, format_budget_table
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="R13 Budget Controller Sweep")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/budgets/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R13] Generating {args.frames} frames with seed={args.seed}...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)

    print("Running Latency Budget Spectrum: 2ms, 4ms, 8ms, 16ms, Unconstrained...")
    ablation = run_full_budget_matrix(
        frames=frames,
        intrinsics=intrinsics,
        budgets=[2.0, 4.0, 8.0, 16.0, None],
        device=args.device,
    )

    # Save JSON results
    out_path = os.path.join(args.output_dir, 'budget_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(ablation, f, indent=4)
    print(f"Saved results to {out_path}")

    # Print table
    print("\n" + "=" * 90)
    print("                    R13: CLOSED-LOOP BUDGET CONTROLLER BENCHMARK")
    print("=" * 90)
    print(format_budget_table(ablation))
    print("=" * 90 + "\n")

    # Save F7: Budget vs Latency CSV
    f7_path = 'results/figures/f7_budget_vs_latency.csv'
    with open(f7_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['budget_label', 'budget_ms', 'mean_latency_ms', 'p95_latency_ms', 'std_latency_ms', 'mean_fps', 'min_fps', 'violation_rate', 'psnr'])
        for exp in ablation['experiments']:
            b_val = exp['budget_ms'] if exp['budget_ms'] is not None else 999.0
            writer.writerow([
                exp['budget_label'],
                b_val,
                f"{exp['mean_frame_time_ms']:.2f}",
                f"{exp['p95_frame_time_ms']:.2f}",
                f"{exp['std_frame_time_ms']:.2f}",
                f"{exp['avg_fps']:.2f}",
                f"{exp['min_fps']:.2f}",
                f"{exp['budget_violation_rate']:.4f}",
                f"{exp['avg_psnr']:.4f}",
            ])
    print(f"Saved F7 budget curve data to {f7_path}")


if __name__ == '__main__':
    main()
