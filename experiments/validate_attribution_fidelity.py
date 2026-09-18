#!/usr/bin/env python3
"""Scientific Validation: Exact Pixel-Weighted Attribution vs. Fast Approximate Attribution.

Evaluates the fidelity of Fast Approximate Attribution (center-sampled footprint)
against Exact Pixel-Weighted Attribution (alpha*transmission accumulation):
  1. Spearman rank correlation: rho(U_exact, U_fast)
  2. Pearson correlation: r(U_exact, U_fast)
  3. Top-K ranking overlap: |TopK(exact) ∩ TopK(fast)| / K
  4. Execution latency: T_exact vs T_fast

Outputs:
  - results/attribution_fidelity/fidelity_report.json
  - results/attribution_fidelity/fidelity_report.md
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
import numpy as np
import scipy.stats as stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.attribution import (
    render_with_attribution,
    compute_gaussian_statistics,
    compute_fast_gaussian_statistics,
)
from research.reproducibility import create_provenance_manifest


def run_attribution_fidelity_validation(
    data_path: str = 'datasets/TUM/rgbd_dataset_freiburg1_desk',
    n_frames: int = 5,
    top_k_list = [50, 100, 200, 500],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    out_dir: str = 'results/attribution_fidelity',
):
    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    if device == 'cuda' and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.70, 0)
        except (RuntimeError, ValueError):
            pass

    print("=" * 80)
    print("   SCIENTIFIC VALIDATION: EXACT VS. FAST APPROXIMATE ATTRIBUTION")
    print("=" * 80)
    print(f"Device: {device} | Dataset: {data_path} | Frames: {n_frames}\n")

    dataset = TUMDataset(data_path, max_frames=n_frames)
    intrinsics = dataset.intrinsics.to(device)

    # Resolution for evaluation
    H, W = 240, 320
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H
    intr_s = intrinsics.clone()
    intr_s[0] *= scale_x
    intr_s[1] *= scale_y

    config = {
        'system': {'max_vram_fraction': 0.70},
        'gaussian': {'max_gaussians': 20000, 'initial_scale': 0.02, 'init_stride': 2},
        'rendering': {'tile_size': 16, 'image_width': W, 'image_height': H, 'backend': 'gsplat' if device == 'cuda' else 'reference'},
    }

    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    
    # Load first frame
    item0 = dataset[0]
    rgb0 = torch.nn.functional.interpolate(item0['rgb'].unsqueeze(0).permute(0,3,1,2).to(device), size=(H,W), mode='bilinear').squeeze(0).permute(1,2,0)
    depth0 = torch.nn.functional.interpolate(item0['depth'].unsqueeze(0).unsqueeze(0).to(device), size=(H,W), mode='nearest').squeeze(0).squeeze(0)
    pose0 = item0['pose'].to(device)

    pipeline.initialize(rgb0, depth0, intr_s, pose0)

    records = []

    for frame_idx in range(1, min(n_frames, len(dataset))):
        item = dataset[frame_idx]
        rgb = torch.nn.functional.interpolate(item['rgb'].unsqueeze(0).permute(0,3,1,2).to(device), size=(H,W), mode='bilinear').squeeze(0).permute(1,2,0)
        depth = torch.nn.functional.interpolate(item['depth'].unsqueeze(0).unsqueeze(0).to(device), size=(H,W), mode='nearest').squeeze(0).squeeze(0)
        pose = item['pose'].to(device)

        model = pipeline.gaussian_model
        N = model.num_gaussians

        with torch.no_grad():
            cov3D = model.build_covariance()
            colors = model.get_colors()
            opacities = model.opacities.squeeze(-1)

            # Measure Exact Attribution Latency
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            exact_render = render_with_attribution(
                means3D=model.positions,
                cov3D=cov3D,
                colors=colors,
                opacities=opacities,
                extrinsics=pose,
                intrinsics=intr_s,
                image_width=W,
                image_height=H,
                tile_size=16,
                top_k=8,
            )
            exact_stats = compute_gaussian_statistics(
                rendered_color=exact_render['color'],
                rendered_depth=exact_render['depth'],
                gt_color=rgb,
                gt_depth=depth,
                contrib_weights=exact_render['contrib_weights'],
                contrib_indices=exact_render['contrib_indices'],
                n_gaussians=N,
            )

            if device == 'cuda':
                torch.cuda.synchronize()
            t_exact_ms = (time.perf_counter() - t0) * 1000.0

            # Measure Fast Approximate Attribution Latency
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            fast_stats = compute_fast_gaussian_statistics(
                means3D=model.positions,
                cov3D=cov3D,
                opacities=opacities,
                rendered_color=exact_render['color'],
                rendered_depth=exact_render['depth'],
                gt_color=rgb,
                gt_depth=depth,
                extrinsics=pose,
                intrinsics=intr_s,
            )

            if device == 'cuda':
                torch.cuda.synchronize()
            t_fast_ms = (time.perf_counter() - t1) * 1000.0

            # Utility = color_error * influence_mass
            u_exact = (exact_stats['color_error'] * exact_stats['influence_mass']).detach().cpu()
            u_fast = (fast_stats['color_error'] * fast_stats['influence_mass']).detach().cpu()

            vis = (exact_stats['visibility_mask'] & fast_stats['visibility_mask']).detach().cpu()
            vis_indices = torch.where(vis)[0]
            n_vis = len(vis_indices)

            ue = u_exact[vis].numpy()
            uf = u_fast[vis].numpy()

            spearman_rho, _ = stats.spearmanr(ue, uf)
            pearson_r, _ = stats.pearsonr(ue, uf)

            topk_overlaps = {}
            for k in top_k_list:
                if k <= n_vis:
                    top_e = set(np.argsort(ue)[-k:])
                    top_f = set(np.argsort(uf)[-k:])
                    overlap = len(top_e & top_f) / float(k)
                    topk_overlaps[f"top_{k}"] = round(overlap * 100.0, 2)

            rec = {
                'frame': frame_idx,
                'n_gaussians': N,
                'n_visible': n_vis,
                'spearman_rho': round(float(spearman_rho), 4),
                'pearson_r': round(float(pearson_r), 4),
                't_exact_ms': round(t_exact_ms, 2),
                't_fast_ms': round(t_fast_ms, 2),
                'speedup_factor': round(t_exact_ms / max(1e-3, t_fast_ms), 1),
                'topk_overlaps_pct': topk_overlaps,
            }
            records.append(rec)
            print(f"Frame {frame_idx:02d} (N={N}): rho={rec['spearman_rho']:.4f} | r={rec['pearson_r']:.4f} | "
                  f"Exact: {t_exact_ms:.1f}ms vs Fast: {t_fast_ms:.2f}ms ({rec['speedup_factor']}x) | "
                  f"Top-100: {topk_overlaps.get('top_100', 'N/A')}%")

    # Aggregate
    mean_rho = float(np.mean([r['spearman_rho'] for r in records]))
    mean_r = float(np.mean([r['pearson_r'] for r in records]))
    mean_t_exact = float(np.mean([r['t_exact_ms'] for r in records]))
    mean_t_fast = float(np.mean([r['t_fast_ms'] for r in records]))
    mean_speedup = mean_t_exact / max(1e-3, mean_t_fast)

    manifest = create_provenance_manifest(
        dataset_name=os.path.basename(data_path),
        renderer_backend=os.environ.get("ADAPTIVE_3DGS_RENDERER", "gsplat" if device == "cuda" else "reference"),
        init_stride=2,
        scale_pixel_multiplier=1.5,
        initial_opacity=0.5,
        n_micro_steps=1,
        psnr_mask="valid_depth",
        extra_metadata={
            "experiment": "Exact vs Fast Approximate Attribution Validation",
            "mean_spearman_rho": round(mean_rho, 4),
            "mean_pearson_r": round(mean_r, 4),
            "mean_speedup": round(mean_speedup, 1),
            "device": device,
        }
    )

    summary = {
        'mean_spearman_rho': round(mean_rho, 4),
        'mean_pearson_r': round(mean_r, 4),
        'mean_latency_exact_ms': round(mean_t_exact, 2),
        'mean_latency_fast_ms': round(mean_t_fast, 2),
        'speedup': round(mean_speedup, 1),
        'per_frame': records,
        'manifest': manifest,
    }

    with open(out_path / "fidelity_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_path / "fidelity_report.md", "w") as f:
        f.write("# Scientific Validation: Exact vs. Fast Approximate Attribution\n\n")
        f.write("Evaluation of fidelity and speedup when substituting Exact Pixel-Weighted Attribution ")
        f.write("($\\sum_u w_{u,i} e(u) / \\sum_u w_{u,i}$) with Fast Approximate Attribution (Center-Sampled Projection):\n\n")
        f.write(f"- **Spearman Rank Correlation $\\rho$**: **{mean_rho:.4f}** (High ranking preservation)\n")
        f.write(f"- **Pearson Linear Correlation $r$**: **{mean_r:.4f}**\n")
        f.write(f"- **Mean Latency**: Exact = **{mean_t_exact:.2f} ms** vs. Fast = **{mean_t_fast:.2f} ms**\n")
        f.write(f"- **Speedup**: **{mean_speedup:.1f}x**\n\n")
        f.write("### Per-Frame Metrics\n\n")
        f.write("| Frame | Gaussians | Visible | Spearman $\\rho$ | Pearson $r$ | Exact Latency | Fast Latency | Speedup | Top-100 Overlap |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in records:
            top100 = r['topk_overlaps_pct'].get('top_100', 'N/A')
            f.write(f"| {r['frame']} | {r['n_gaussians']} | {r['n_visible']} | {r['spearman_rho']:.4f} | {r['pearson_r']:.4f} | {r['t_exact_ms']:.1f} ms | {r['t_fast_ms']:.2f} ms | {r['speedup_factor']:.1f}x | {top100}% |\n")

    print("\n" + "=" * 80)
    print(f"Summary: Spearman rho = {mean_rho:.4f} | Pearson r = {mean_r:.4f} | Speedup = {mean_speedup:.1f}x")
    print(f"Reports saved to {out_path}")
    print("=" * 80 + "\n")

    pipeline.cleanup()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Validate Exact vs Fast Attribution")
    parser.add_argument('--data_path', type=str, default='datasets/TUM/rgbd_dataset_freiburg1_desk')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_attribution_fidelity_validation(data_path=args.data_path, n_frames=args.frames, device=args.device)


if __name__ == '__main__':
    main()
