import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.benchmark_budgets import run_full_budget_matrix, format_budget_table

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
    args = parser.parse_args()

    os.makedirs('results/budgets/', exist_ok=True)
    
    frames, intrinsics = generate_synthetic_benchmark_dataset(args.frames, seed=args.seed)
    
    budgets = [2, 4, 8, 16, None]
    
    print("Running Budget Sweep...")
    # NOTE: budget_ms refers to the full frame time budget (Option A)
    # The pipeline is expected to adapt scheduling to keep the total processing time per frame under this budget.
    results = run_full_budget_matrix(budgets, frames, intrinsics, args.device, seed=args.seed)
    
    for b, res in results.items():
        name = f"{b}ms" if b is not None else "unconstrained"
        with open(os.path.join('results/budgets/', f"{name}.json"), 'w') as f:
            json.dump(res, f, indent=4)
            
    print("\n" + format_budget_table(results))
    print(f"Results saved to results/budgets/")

if __name__ == '__main__':
    main()
