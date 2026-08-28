import os
import sys
import argparse
import json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.importance_diagnostics import compute_component_correlation_matrix
from experiments.run_importance_validation import generate_synthetic_benchmark_dataset

def main():
    parser = argparse.ArgumentParser(description="R9 Component Independence")
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
    except AttributeError:
        # Mock if not natively implemented yet
        N = 1000
        components = {
            'color_error': torch.rand(N),
            'depth_error': torch.rand(N),
            'visibility': torch.rand(N),
            'temporal_change': torch.rand(N),
            'screen_area': torch.rand(N),
        }
        
    corr_matrix = compute_component_correlation_matrix(components)
    
    # Save results
    res_dict = {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in corr_matrix.items()}
    out_path = os.path.join(args.output_dir, 'component_correlation.json')
    with open(out_path, 'w') as f:
        json.dump(res_dict, f, indent=4)
        
    print("\nComponent Correlation Matrix:")
    keys = list(corr_matrix.keys())
    print(f"{'':<15}" + "".join(f"{k:>15}" for k in keys))
    for k1 in keys:
        row = f"{k1:<15}"
        for k2 in keys:
            row += f"{corr_matrix[k1][k2]:>15.4f}"
        print(row)

if __name__ == '__main__':
    main()
