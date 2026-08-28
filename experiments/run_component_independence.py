"""R9: Component Independence & Multi-Signal Correlation Experiment.

Evaluates hypothesis that importance combines distinct orthogonal geometric and photometric signals
rather than simple redundant error proxies.
"""
import os
import sys
import argparse
import json
import csv
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import component_correlation_matrix, spearman_rank_correlation
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="R9 Component Independence")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/importance/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R9] Generating {args.frames} frames with seed={args.seed}...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)
    
    pipeline = OnlineReconstructionPipeline(device=args.device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )
    
    for i in range(1, len(frames)):
        frame = frames[i]
        pipeline.process_frame(
            rgb=frame['rgb'],
            depth=frame['depth'],
            gt_pose=frame['pose'],
        )
        
    state = pipeline.get_importance_diagnostics()
    components = {
        'color_error': state['color_error'],
        'depth_error': state['depth_error'],
        'visibility': state['visibility'],
        'temporal_change': state['temporal_change'],
        'screen_area': state['screen_area'],
    }
        
    keys = sorted(components.keys())
    full_matrix = {}
    for k1 in keys:
        full_matrix[k1] = {}
        for k2 in keys:
            if k1 == k2:
                full_matrix[k1][k2] = 1.0
            else:
                rho = spearman_rank_correlation(components[k1].float(), components[k2].float())
                full_matrix[k1][k2] = float(rho)
    
    # Save JSON results
    out_path = os.path.join(args.output_dir, 'component_correlation.json')
    with open(out_path, 'w') as f:
        json.dump(full_matrix, f, indent=4)
        
    print("\n" + "=" * 80)
    print("                    R9: COMPONENT CORRELATION MATRIX")
    print("=" * 80)
    print(f"{'':<18}" + "".join(f"{k:>15}" for k in keys))
    print("-" * 80)
    for k1 in keys:
        row = f"{k1:<18}"
        for k2 in keys:
            row += f"{full_matrix[k1][k2]:>15.4f}"
        print(row)
    print("=" * 80 + "\n")
    
    # Save F4 CSV
    f4_path = 'results/figures/f4_component_correlation.csv'
    with open(f4_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component'] + keys)
        for k1 in keys:
            writer.writerow([k1] + [f"{full_matrix[k1][k2]:.4f}" for k2 in keys])
    print(f"Saved F4 heatmap data to {f4_path}")



if __name__ == '__main__':
    main()
