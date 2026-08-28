"""R8: Importance Validation & R15: Error Coverage@K Benchmark.

Evaluates hypothesis H1: Does continuous importance accurately predict reconstruction error?
Measures:
    - Spearman rank correlation: ρ(I, E_color), ρ(I, E_depth), ρ(I, E_combined)
    - Error Capture: Coverage@K across [5%, 10%, 25%, 50%, 75%, 100%]
      for Importance vs Random vs Binary selection policies.
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
from research.importance_diagnostics import (
    spearman_rank_correlation,
    pearson_correlation,
    compute_full_diagnostics,
)


def generate_synthetic_benchmark_dataset(n_frames: int = 20, H: int = 64, W: int = 64, seed: int = 42):
    """Generate synthetic RGB-D sequence with textured objects and camera motion."""
    torch.manual_seed(seed)
    intrinsics = torch.tensor([
        [60.0, 0.0, float(W // 2)],
        [0.0, 60.0, float(H // 2)],
        [0.0, 0.0, 1.0],
    ])
    
    frames = []
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    
    for i in range(n_frames):
        pattern = ((x * 4).sin() * (y * 4).cos()).clamp(-1, 1) * 0.5 + 0.5
        color = torch.stack([
            pattern,
            (pattern * 1.5).clamp(0, 1),
            (1.0 - pattern),
        ], dim=-1)
        
        depth = torch.full((H, W), 2.0)
        center_mask = (x**2 + y**2) < 0.25
        depth[center_mask] = 1.5
        depth += torch.randn(H, W) * 0.01
        
        pose = torch.eye(4)
        pose[0, 3] = (i - n_frames // 2) * 0.04
        pose[1, 3] = i * 0.02
        
        frames.append({
            'rgb': color.float(),
            'depth': depth.float(),
            'pose': pose.float(),
        })
        
    return frames, intrinsics


def compute_coverage_at_k(order_tensor: torch.Tensor, errors: torch.Tensor, k_percentages=[5, 10, 25, 50, 75, 100]):
    """Coverage@K = sum(E_i for top K% Gaussians) / sum(E_i)."""
    total_error = errors.sum().item()
    if total_error < 1e-8:
        return {k: 1.0 for k in k_percentages}
    
    sorted_idx = torch.argsort(order_tensor, descending=True)
    sorted_errors = errors[sorted_idx]
    N = len(order_tensor)
    
    result = {}
    for k in k_percentages:
        n_select = max(1, int(N * k / 100.0))
        captured = sorted_errors[:n_select].sum().item()
        result[k] = captured / total_error
    return result


def main():
    parser = argparse.ArgumentParser(description="R8 Importance Validation & R15 Coverage@K")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=20, help='Number of frames to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/importance/', help='Output directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results/figures/', exist_ok=True)

    print(f"[R8] Generating {args.frames} frames with seed={args.seed}...")
    frames, intrinsics = generate_synthetic_benchmark_dataset(n_frames=args.frames, seed=args.seed)
    
    pipeline = OnlineReconstructionPipeline(device=args.device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )
    
    checkpoint_records = []
    
    for i in range(1, len(frames)):
        frame = frames[i]
        pipeline.process_frame(
            rgb=frame['rgb'],
            depth=frame['depth'],
            gt_pose=frame['pose'],
        )
        
        # Checkpoint every 5 frames
        if (i + 1) % 5 == 0 or (i + 1) == len(frames):
            state = pipeline.get_importance_diagnostics()
            importance = state['importance']
            color_error = state['color_error']
            depth_error = state['depth_error']
            combined_error = color_error + depth_error
            confidence = state['confidence']
            
            # 1. Spearman Rank Correlations
            spearman_col = spearman_rank_correlation(importance, color_error)
            spearman_dep = spearman_rank_correlation(importance, depth_error)
            spearman_com = spearman_rank_correlation(importance, combined_error)
            
            # 2. Coverage@K (Importance vs Random vs Binary)
            cov_importance = compute_coverage_at_k(importance, combined_error)
            random_ranking = torch.rand_like(importance)
            cov_random = compute_coverage_at_k(random_ranking, combined_error)
            # Binary: lower confidence = higher priority to optimize
            binary_ranking = 1.0 - confidence
            cov_binary = compute_coverage_at_k(binary_ranking, combined_error)
            
            rec = {
                'frame': i + 1,
                'n_gaussians': pipeline.gaussian_model.num_gaussians,
                'spearman_color': float(spearman_col),
                'spearman_depth': float(spearman_dep),
                'spearman_combined': float(spearman_com),
                'coverage_importance': {str(k): float(v) for k, v in cov_importance.items()},
                'coverage_random': {str(k): float(v) for k, v in cov_random.items()},
                'coverage_binary': {str(k): float(v) for k, v in cov_binary.items()},
            }
            checkpoint_records.append(rec)
            
            out_file = os.path.join(args.output_dir, f"frame_{i+1:03d}.json")
            with open(out_file, 'w') as f:
                json.dump(rec, f, indent=4)
                
            print(f"  Frame {i+1:03d}: Spearman ρ(I, E_comb)={spearman_com:.4f}, Coverage@10%={cov_importance[10]:.1%}")

    # Final State Summary & Figures Data
    if checkpoint_records:
        spearmans = [r['spearman_combined'] for r in checkpoint_records]
        summary = {
            'seed': args.seed,
            'total_frames': args.frames,
            'mean_spearman_combined': float(np.mean(spearmans)),
            'std_spearman_combined': float(np.std(spearmans)),
            'p5_spearman': float(np.percentile(spearmans, 5)),
            'p50_spearman': float(np.percentile(spearmans, 50)),
            'p95_spearman': float(np.percentile(spearmans, 95)),
            'final_coverage_importance': checkpoint_records[-1]['coverage_importance'],
            'final_coverage_random': checkpoint_records[-1]['coverage_random'],
            'final_coverage_binary': checkpoint_records[-1]['coverage_binary'],
        }
        
        sum_path = os.path.join(args.output_dir, 'summary.json')
        with open(sum_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print("\n" + "=" * 60)
        print("          R8/R15 IMPORTANCE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Mean Spearman ρ(I, E):  {summary['mean_spearman_combined']:.4f} ± {summary['std_spearman_combined']:.4f}")
        print(f"P50 Spearman ρ(I, E):   {summary['p50_spearman']:.4f} (P5: {summary['p5_spearman']:.4f}, P95: {summary['p95_spearman']:.4f})")
        print("-" * 60)
        print("Coverage@K Error Capture Comparison (Final Frame):")
        print(f"{'K (%)':<10}{'Importance':<15}{'Random':<15}{'Binary':<15}")
        for k in [5, 10, 25, 50, 75, 100]:
            imp_val = summary['final_coverage_importance'][str(k)]
            rnd_val = summary['final_coverage_random'][str(k)]
            bin_val = summary['final_coverage_binary'][str(k)]
            print(f"{k:<10}{imp_val:<15.1%}{rnd_val:<15.1%}{bin_val:<15.1%}")
        print("=" * 60 + "\n")
        
        # Save F1: Importance vs Error Scatter CSV
        final_state = pipeline.get_importance_diagnostics()
        f1_path = 'results/figures/f1_importance_vs_error.csv'
        with open(f1_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['gaussian_id', 'importance', 'color_error', 'depth_error', 'combined_error', 'tier'])
            imp_arr = final_state['importance'].cpu().numpy()
            col_arr = final_state['color_error'].cpu().numpy()
            dep_arr = final_state['depth_error'].cpu().numpy()
            com_arr = (final_state['color_error'] + final_state['depth_error']).cpu().numpy()
            tier_arr = final_state['tiers'].cpu().numpy()
            for g_id in range(len(imp_arr)):
                writer.writerow([g_id, f"{imp_arr[g_id]:.6f}", f"{col_arr[g_id]:.6f}", f"{dep_arr[g_id]:.6f}", f"{com_arr[g_id]:.6f}", int(tier_arr[g_id])])
        print(f"Saved F1 data to {f1_path}")
        
        # Save F2: Coverage@K CSV
        f2_path = 'results/figures/f2_coverage_at_k.csv'
        with open(f2_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['k_percent', 'importance', 'random', 'binary'])
            for k in [5, 10, 25, 50, 75, 100]:
                writer.writerow([
                    k,
                    summary['final_coverage_importance'][str(k)],
                    summary['final_coverage_random'][str(k)],
                    summary['final_coverage_binary'][str(k)],
                ])
        print(f"Saved F2 data to {f2_path}")


if __name__ == '__main__':
    main()
