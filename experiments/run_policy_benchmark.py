import os
import sys
import json
import argparse
import torch
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.pipeline import OnlineReconstructionPipeline
from research.benchmark_policies import run_full_policy_ablation_matrix, format_benchmark_table

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/policies/')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)
    
    frames, intrinsics = generate_synthetic_benchmark_dataset(args.frames, seed=args.seed)
    
    policies = [
        "full",
        "random@10%", "random@25%", "random@50%", "random@75%",
        "top_k@10%", "top_k@25%", "top_k@50%", "top_k@75%",
        "binary",
        "budget_aware@2ms", "budget_aware@4ms", "budget_aware@8ms", "budget_aware@16ms"
    ]
    
    print("Running Policy Benchmark...")
    results = run_full_policy_ablation_matrix(policies, frames, intrinsics, args.device, seed=args.seed)
    
    for p, res in results.items():
        with open(os.path.join(args.output_dir, f"{p.replace('@', '_')}.json"), 'w') as f:
            json.dump(res, f, indent=4)
            
    print("\n" + format_benchmark_table(results))
    
    fig_data = []
    for p, res in results.items():
        opt_pct = res.get('mean_optimized_pct', 0.0)
        psnr = res.get('mean_psnr', 0.0)
        fig_data.append({'policy': p, 'optimized_gaussian_pct': opt_pct, 'psnr': psnr})
        
    with open('results/figures/quality_vs_compute.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'optimized_gaussian_pct', 'psnr'])
        writer.writeheader()
        writer.writerows(fig_data)
        
    print(f"Results saved to {args.output_dir}")
    print(f"Figure data saved to results/figures/quality_vs_compute.csv")

if __name__ == '__main__':
    main()
