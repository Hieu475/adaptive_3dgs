"""
Failure Analysis Diagnostic Runner for Adaptive 3DGS.

Executes comprehensive failure mode analysis on challenging geometric
and photometric situations:
  1. FLAT_SURFACE
  2. OBJECT_EDGE
  3. HIGH_TEXTURE
  4. SPARSE_DEPTH
  5. VIEWPOINT_CHANGE
"""
import sys
import os
import torch
import numpy as np
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.pipeline import OnlineReconstructionPipeline
from research.failure_analysis import FailureCaseAnalyzer, format_failure_analysis_report
from research.attribution import render_with_attribution


def create_stress_test_frames(n_frames: int = 10, H: int = 64, W: int = 80):
    """Create frames with distinct failure case conditions."""
    fx, fy = 160.0, 160.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    frames = []
    
    for t in range(n_frames):
        # Frame 7 has a sudden large viewpoint jump to trigger VIEWPOINT_CHANGE
        angle = t * 0.03 if t != 7 else t * 0.03 + 0.35
        pose = torch.eye(4)
        pose[0, 0] = np.cos(angle); pose[0, 2] = np.sin(angle)
        pose[2, 0] = -np.sin(angle); pose[2, 2] = np.cos(angle)
        pose[0, 3] = 0.02 * t if t != 7 else 0.02 * t + 0.25
        
        rgb = torch.zeros(H, W, 3)
        depth = torch.ones(H, W) * 3.0
        
        # 1. HIGH_TEXTURE: Fine checkerboard in top-left
        for i in range(H // 2):
            for j in range(W // 2):
                if (i // 4 + j // 4) % 2 == 0:
                    rgb[i, j] = torch.tensor([0.95, 0.1, 0.05])
                else:
                    rgb[i, j] = torch.tensor([0.05, 0.85, 0.15])
        depth[:H//2, :W//2] = 2.0
        
        # 2. FLAT_SURFACE: Uniform smooth low-frequency gradient in top-right
        for j in range(W // 2, W):
            rgb[:H//2, j] = torch.tensor([0.45, 0.45, 0.55])
        depth[:H//2, W//2:] = 2.5
        
        # 3. OBJECT_EDGE: Sharp depth discontinuity in center-bottom
        box_h = slice(H // 2 + 5, H - 5)
        box_w = slice(W // 4, 3 * W // 4)
        rgb[box_h, box_w] = torch.tensor([0.8, 0.2, 0.6])
        depth[box_h, box_w] = 1.1
        
        # 4. SPARSE_DEPTH: Invalid depth patch in bottom-left
        depth[H - 12:, :12] = 0.0
        
        # Noise
        rgb = (rgb + 0.02 * torch.randn_like(rgb)).clamp(0, 1)
        depth = depth + 0.01 * torch.randn_like(depth)
        depth[depth <= 0] = 0.0
        
        frames.append({
            'rgb': rgb,
            'depth': depth,
            'pose': pose
        })
        
    return frames, intrinsics


def main():
    parser = argparse.ArgumentParser(description="Run Failure Analysis Diagnostic Suite")
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--frames', type=int, default=None, help='Alias for n_frames')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    if args.frames is not None:
        args.n_frames = args.frames
    
    print("=" * 72)
    print("      FAILURE CASE DIAGNOSTIC & EDGE IMPORTANCE ANALYSIS")
    print("=" * 72)
    
    frames, intrinsics = create_stress_test_frames(n_frames=args.n_frames)
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 20000, 'initial_scale': 0.02},
        'rendering': {
            'tile_size': 16,
            'image_width': 80,
            'image_height': 64,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4
        },
        'scheduler': {
            'gpu_budget_ms': 50.0,
            'policy': 'budget_aware',
            'optimize_ratio': 0.6,
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=args.device)
    analyzer = FailureCaseAnalyzer()
    
    f0 = frames[0]
    pipeline.initialize(f0['rgb'], f0['depth'], intrinsics, f0['pose'])
    
    analysis_reports = []
    
    for i in range(1, len(frames)):
        f = frames[i]
        prev_f = frames[i - 1]
        
        pipeline.process_frame(f['rgb'], f['depth'], gt_pose=f['pose'])
        
        # Render with attribution to get pixel-to-gaussian contributions
        H, W = f['rgb'].shape[:2]
        attr = render_with_attribution(
            means3D=pipeline.gaussian_model.positions,
            cov3D=pipeline.gaussian_model.build_covariance(),
            colors=pipeline.gaussian_model.get_colors(),
            opacities=pipeline.gaussian_model.opacities.squeeze(-1),
            extrinsics=f['pose'],
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
            top_k=4
        )
        
        diag = pipeline.get_importance_diagnostics()
        importance = diag['importance']
        
        # Run failure case analysis
        analysis = analyzer.analyze_frame(
            rgb=f['rgb'],
            depth=f['depth'],
            rendered_color=attr['color'],
            rendered_depth=attr['depth'],
            importance=importance,
            visibility_mask=None,
            contrib_indices=attr['contrib_indices'],
            contrib_weights=attr['contrib_weights'],
            n_gaussians=pipeline.gaussian_model.num_gaussians,
            current_pose=f['pose'],
            prev_pose=prev_f['pose']
        )
        
        report_str = format_failure_analysis_report(analysis)
        analysis_reports.append(analysis)
        
        if i in (1, len(frames) - 2, len(frames) - 1):
            print(f"\n[Frame {i}] Diagnostic Output:")
            print(report_str)
            
    # Aggregate Region-Labeled Summary Table
    print("\n" + "=" * 80)
    print("                 MULTI-FRAME REGION-LABELED FAILURE SUMMARY")
    print("=" * 80)
    print(f"{'Region / Failure Mode':<22} | {'Mean Pixels':>11} | {'Mean PSNR':>10} | {'Mean Imp':>9} | {'Mean Severity':>13}")
    print("-" * 80)
    for category in ['FLAT_SURFACE', 'OBJECT_EDGE', 'HIGH_TEXTURE', 'SPARSE_DEPTH', 'VIEWPOINT_CHANGE']:
        pix_list = [rep[category]['affected_pixels'] for rep in analysis_reports if category in rep]
        psnr_list = [rep[category]['quality_in_region'] for rep in analysis_reports if category in rep and rep[category]['affected_pixels'] > 0]
        imp_list = [rep[category]['importance_in_region'] for rep in analysis_reports if category in rep and rep[category]['affected_pixels'] > 0]
        sev_list = [rep[category]['failure_severity'] for rep in analysis_reports if category in rep]
        
        m_pix = np.mean(pix_list) if pix_list else 0.0
        m_psnr = f"{np.mean(psnr_list):.2f} dB" if psnr_list else "N/A"
        m_imp = f"{np.mean(imp_list):.4f}" if imp_list else "0.0000"
        m_sev = f"{np.mean(sev_list):.2f}" if sev_list else "0.00"
        print(f"{category:<22} | {m_pix:>11.0f} | {m_psnr:>10} | {m_imp:>9} | {m_sev:>13}")
    print("=" * 80)

    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'failure_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    json_path = os.path.join(save_dir, 'failure_analysis_summary.json')
    with open(json_path, 'w') as f:
        json.dump(analysis_reports, f, indent=2, default=str)
        
    print(f"\nDetailed diagnostics saved to {json_path}")


if __name__ == "__main__":
    main()
