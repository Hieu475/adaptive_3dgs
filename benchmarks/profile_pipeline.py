import os
import sys
import json
import argparse
import torch
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline

def generate_synthetic_benchmark_dataset(n_frames=5, H=64, W=64, seed=42):
    torch.manual_seed(seed)
    intrinsics = torch.tensor([[50.0, 0, W/2], [0, 50.0, H/2], [0, 0, 1.0]])
    frames = []
    for i in range(n_frames):
        rgb = torch.rand(H, W, 3) * 0.8 + 0.1
        depth = torch.full((H, W), 2.0) + torch.rand(H, W) * 0.5
        pose = torch.eye(4)
        pose[0, 3] = i * 0.02
        frames.append({'rgb': rgb, 'depth': depth, 'pose': pose})
    return frames, intrinsics

def profile_pipeline_loop(pipeline, frames_data):
    times = {
        'Tracking': [],
        'Attribution': [],
        'Depth Rendering': [],
        'Importance Estimation': [],
        'Densification': [],
        'Scheduling': [],
        'Optimization': [],
        'Pruning': [],
        'Total Frame': []
    }
    
    for frame in frames_data:
        t_start = time.time()
        
        t0 = time.time()
        if hasattr(pipeline, 'track'):
            pipeline.track(frame['pose'])
        times['Tracking'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'render_with_attribution'):
            pipeline.render_with_attribution(frame['pose'])
        times['Attribution'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'render_depth'):
            pipeline.render_depth(frame['pose'])
        times['Depth Rendering'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'estimate_importance'):
            pipeline.estimate_importance()
        times['Importance Estimation'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'densify'):
            pipeline.densify()
        times['Densification'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'schedule'):
            pipeline.schedule()
        times['Scheduling'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'optimize'):
            pipeline.optimize(frame['rgb'], frame['depth'])
        times['Optimization'].append((time.time() - t0) * 1000)
        
        t0 = time.time()
        if hasattr(pipeline, 'prune'):
            pipeline.prune()
        times['Pruning'].append((time.time() - t0) * 1000)
        
        times['Total Frame'].append((time.time() - t_start) * 1000)
        
    return times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames', type=int, default=10)
    args = parser.parse_args()

    os.makedirs('results/figures/', exist_ok=True)
    
    frames, intrinsics = generate_synthetic_benchmark_dataset(args.frames)
    
    pipeline = OnlineReconstructionPipeline(intrinsics, args.device)
    
    print("Profiling pipeline stages...")
    times = profile_pipeline_loop(pipeline, frames)
    
    breakdown = {}
    print("\n| Stage | Mean (ms) | Std (ms) | % of Total |")
    print("|---|---|---|---|")
    total_mean = np.mean(times['Total Frame']) if times['Total Frame'] else 1.0
    
    for stage, t_list in times.items():
        if stage == 'Total Frame': continue
        mean_t = np.mean(t_list)
        std_t = np.std(t_list)
        pct = (mean_t / total_mean) * 100 if total_mean > 0 else 0
        breakdown[stage] = {'mean': mean_t, 'std': std_t, 'pct': pct}
        print(f"| {stage} | {mean_t:.2f} | {std_t:.2f} | {pct:.1f}% |")
        
    breakdown['Total Frame'] = {'mean': total_mean, 'std': np.std(times['Total Frame']), 'pct': 100.0}
    
    with open('results/figures/profile_breakdown.json', 'w') as f:
        json.dump(breakdown, f, indent=4)
        
    print(f"\nResults saved to results/figures/profile_breakdown.json")

if __name__ == '__main__':
    main()
