#!/usr/bin/env python3
"""Multi-Frame Ground-Truth Oracle Dataset Generator on Real TUM RGB-D.

Samples counterfactual interventions across multiple frames (t = 15, 20, 25, 30)
to provide diverse viewpoints, geometric strata, and reconstruction stages
with strictly unclamped marginal utilities U_i^* in Real.
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
from research.protocol import load_protocol, get_dataset_config, get_resolution


def load_tum_sequence(data_path: str, n_frames: int = 35, H: int = 240, W: int = 320, device: str = 'cuda'):
    dataset = TUMDataset(data_path, max_frames=n_frames, camera='freiburg1')
    frames = []
    
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H
    
    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0, dataset.cx * scale_x],
        [0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0, 0, 1.0]
    ], dtype=torch.float32, device=device)
    
    for i in range(min(n_frames, len(dataset))):
        item = dataset[i]
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)
        
        rgb_scaled = torch.nn.functional.interpolate(
            rgb, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0)
        
        frames.append({
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device)
        })
        
    return frames, intrinsics


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== GENERATING MULTI-FRAME ORACLE DATASET [Device: {device}] ===")
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protocol = load_protocol()
    dataset_cfg = protocol["datasets"]["tum_fr1_desk"]
    data_path = os.path.join(repo_root, dataset_cfg["path"])
    
    H = dataset_cfg["image_height"]
    W = dataset_cfg["image_width"]
    eval_frames_idx = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    max_frames = max(eval_frames_idx) + 1
    
    frames, intrinsics = load_tum_sequence(data_path, n_frames=max_frames, H=H, W=W, device=device)
    print(f">> Loaded {len(frames)} TUM frames at {W}x{H}.")
    
    config = {
        'gaussian': {
            'sh_degree': 0,
            'initial_opacity': 0.5,
            'max_gaussians': 30000,
            'initial_scale': 0.02,
        },
        'rendering': {
            'tile_size': 16,
            'image_width': W,
            'image_height': H,
            'use_surface_aware_depth': True,
            'attribution_top_k': 4,
        },
        'scheduler': {
            'gpu_budget_ms': 25.0,
            'policy': 'budget_aware',
        },
        'densification': {
            'max_new_per_frame': 80,
            'strategy': 'importance',
            'use_adaptive_thresholds': True,
        }
    }
    
    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose']
    )
    
    all_rows = []
    
    for t in range(1, max_frames):
        pipeline.process_frame(
            rgb=frames[t]['rgb'],
            depth=frames[t]['depth'],
            gt_pose=frames[t]['pose']
        )
        
        if t in eval_frames_idx:
            print(f"\n>> Sampling interventions at Frame {t} (Active Gaussians: {pipeline.gaussian_model.num_gaussians})...")
            oracle = OracleUtilityExperiment(
                pipeline=pipeline,
                n_samples=40,
                n_opt_steps=5,
                w_rgb=0.70,
                w_depth=0.30,
                seed=42 + t,
                min_influence_pixels=25
            )
            
            frame_res = oracle.run_oracle_experiment(
                rgb=frames[t]['rgb'],
                depth=frames[t]['depth'],
                population_type=SamplingPopulation.GEOMETRY_STRATIFIED,
                frame_idx=t
            )
            vis = [r for r in frame_res if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
            for r in frame_res:
                r['split'] = 'train' if t <= 40 else 'validation'
            print(f"   Collected {len(vis)} visible interventions from frame {t} (split: {'train' if t <= 40 else 'val'}).")
            all_rows.extend(frame_res)
            
    # Save dataset
    save_dir = os.path.join(repo_root, 'results', 'oracle_dataset')
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, 'oracle_dataset.json')
    
    with open(out_file, 'w') as f:
        json.dump(all_rows, f, indent=2)
        
    vis_total = [r for r in all_rows if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    print(f"\n[Artifact] Successfully generated Oracle Dataset with {len(all_rows)} total rows ({len(vis_total)} visible).")
    print(f"           Saved to: {out_file}")


if __name__ == '__main__':
    main()
