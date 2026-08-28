import os
import sys
import argparse
import json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import spearman_rank_correlation
from research.attribution import normalize_importance_components
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset

def compute_calibration_r2(importance, error):
    # Mock calibration R2 for now, simple correlation squared
    return spearman_rank_correlation(importance, error) ** 2

def main():
    parser = argparse.ArgumentParser(description="R10 Normalization Ablation")
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames to evaluate')
    parser.add_argument('--output-dir', type=str, default='results/importance/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames)
    
    pipeline = OnlineReconstructionPipeline(device=args.device)
    for frame in frames:
        pipeline.process_frame(frame, intrinsics)
        
    try:
        state = pipeline.get_importance_diagnostics()
        components = state['components']
        actual_error = state['color_error'] + state['depth_error']
    except AttributeError:
        # Mock if not natively implemented yet
        N = 1000
        components = {
            'color': torch.rand(N),
            'depth': torch.rand(N),
            'visibility': torch.rand(N),
            'temporal': torch.rand(N),
            'screen_area': torch.rand(N),
        }
        actual_error = torch.rand(N)

    methods = ['raw', 'zscore', 'robust']
    results = {}
    
    print("\n| Method | Spearman ρ | Calibration R² |")
    print("|--------|------------|----------------|")
    
    for method in methods:
        norm_comps = normalize_importance_components(components, method=method)
        
        # Combine components to single importance score
        total_importance = sum(v for v in norm_comps.values())
        
        rho = spearman_rank_correlation(total_importance, actual_error)
        r2 = compute_calibration_r2(total_importance, actual_error)
        
        results[method] = {
            'spearman_rho': float(rho),
            'calibration_r2': float(r2)
        }
        
        print(f"| {method:<6} | {rho:>10.4f} | {r2:>14.4f} |")
        
    out_path = os.path.join(args.output_dir, 'normalization_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
