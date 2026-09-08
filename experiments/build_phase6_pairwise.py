#!/usr/bin/env python3
"""Phase 6 — Stratified Pairwise Interaction Dataset Builder.

Generates a pairwise interaction dataset with intentional IoU stratification
to answer: "Does overlap IoU predict non-additivity strength?"

For each pair (i, j), measures:
    ΔQ({i}), ΔQ({j}), ΔQ({i,j}), I(i,j) = ΔQ({i,j}) - ΔQ(i) - ΔQ(j)

Pairs are stratified across overlap bins:
    Low:      IoU < 0.10      (≥100 pairs)
    Medium:   0.10 ≤ IoU < 0.30  (≥100 pairs)
    High:     0.30 ≤ IoU < 0.50  (≥50 pairs, intentionally searched)
    VeryHigh: IoU ≥ 0.50      (as many as available)

Usage:
    python experiments/build_phase6_pairwise.py
    python experiments/build_phase6_pairwise.py --tiny
    python experiments/build_phase6_pairwise.py --scene tum_fr2_xyz --frames 10,20,30

Output:
    results/phase6_context_utility/datasets/pairwise_interaction_seed_42.json
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.phase6_oracle import ConditionalOracleExperiment, ConditionalOracleConfig
from research.phase6_context import _get_pixel_mask, _get_knn_indices
from research.attribution import render_with_attribution, compute_gaussian_statistics
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_oracle_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# IoU Bins
# ─────────────────────────────────────────────────────────────────────────────

IOU_BINS = {
    "low":       (0.0, 0.10),
    "medium":    (0.10, 0.30),
    "high":      (0.30, 0.50),
    "very_high": (0.50, 1.01),
}

MIN_PAIRS_PER_BIN = {
    "low": 100,
    "medium": 100,
    "high": 50,
    "very_high": 20,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (reuse from build_phase6_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_sequence(data_path, camera, n_frames, H, W, device):
    """Load and resize TUM RGB-D sequence."""
    dataset = TUMDataset(data_path, max_frames=n_frames, camera=camera)
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
            'pose': item['pose'].to(device),
        })
    return frames, intrinsics


def build_pipeline(H, W, device):
    """Build OnlineReconstructionPipeline with standard config."""
    config = {
        'gaussian': {
            'sh_degree': 0, 'initial_opacity': 0.5,
            'max_gaussians': 30000, 'initial_scale': 0.02,
        },
        'rendering': {
            'tile_size': 16, 'image_width': W, 'image_height': H,
            'use_surface_aware_depth': True, 'attribution_top_k': 4,
        },
        'scheduler': {'gpu_budget_ms': 25.0, 'policy': 'budget_aware'},
        'densification': {
            'max_new_per_frame': 80, 'strategy': 'importance',
            'use_adaptive_thresholds': True,
        },
    }
    return OnlineReconstructionPipeline(config=config, device=device)


def hash_state(model):
    """Compute cryptographic hash of model state for integrity verification."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(param.detach().cpu().numpy().tobytes())
    for name, buf in sorted(model.named_buffers()):
        hasher.update(name.encode())
        hasher.update(buf.detach().cpu().numpy().tobytes())
    store = getattr(model, 'state_store', None)
    if store is not None:
        hasher.update(b'state_store_next_id')
        hasher.update(str(store._next_id).encode())
        hasher.update(store.persistent_ids.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified Pair Sampling
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_pairwise_iou(
    visible_indices: List[int],
    contrib_indices: torch.Tensor,
    contrib_weights: torch.Tensor,
    threshold: float = 0.01,
    max_pairs: int = 5000,
) -> List[Tuple[int, int, float]]:
    """Compute pixel IoU for candidate pairs, prioritizing diverse overlap bins.

    Uses a two-pass strategy:
    1. First pass: compute IoU for spatial neighbors (likely high overlap)
    2. Second pass: random pairs to fill remaining bins

    Returns:
        List of (idx_i, idx_j, iou) tuples.
    """
    # Precompute pixel masks for all visible Gaussians
    masks = {}
    for idx in visible_indices:
        masks[idx] = _get_pixel_mask(idx, contrib_indices, contrib_weights, threshold)

    pairs_with_iou = []
    seen_pairs = set()

    N = len(visible_indices)
    if N < 2:
        return pairs_with_iou

    # Pass 1: All pairs among first min(N, 80) visible Gaussians
    # (these tend to be spatially co-located and produce higher overlap)
    subset = visible_indices[:min(N, 80)]
    for a_pos in range(len(subset)):
        for b_pos in range(a_pos + 1, len(subset)):
            idx_i, idx_j = subset[a_pos], subset[b_pos]
            if (idx_i, idx_j) in seen_pairs:
                continue
            seen_pairs.add((idx_i, idx_j))

            mask_i = masks[idx_i]
            mask_j = masks[idx_j]
            intersection = (mask_i & mask_j).sum().item()
            union = (mask_i | mask_j).sum().item()
            iou = intersection / max(union, 1)
            pairs_with_iou.append((idx_i, idx_j, iou))

            if len(pairs_with_iou) >= max_pairs:
                return pairs_with_iou

    # Pass 2: Random pairs from full visible set to diversify
    rng = np.random.default_rng(42)
    n_extra = min(max_pairs - len(pairs_with_iou), N * 5)
    for _ in range(n_extra):
        a, b = rng.choice(N, size=2, replace=False)
        idx_i, idx_j = visible_indices[a], visible_indices[b]
        pair_key = (min(idx_i, idx_j), max(idx_i, idx_j))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        mask_i = masks[idx_i]
        mask_j = masks[idx_j]
        intersection = (mask_i & mask_j).sum().item()
        union = (mask_i | mask_j).sum().item()
        iou = intersection / max(union, 1)
        pairs_with_iou.append((idx_i, idx_j, iou))

    return pairs_with_iou


def stratified_sample_pairs(
    all_pairs: List[Tuple[int, int, float]],
    bins: Dict[str, Tuple[float, float]],
    min_per_bin: Dict[str, int],
    rng: np.random.Generator,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """Stratify pairs by IoU bin and sample target counts per bin."""
    binned = defaultdict(list)
    for idx_i, idx_j, iou in all_pairs:
        for bin_name, (lo, hi) in bins.items():
            if lo <= iou < hi:
                binned[bin_name].append((idx_i, idx_j, iou))
                break

    result = {}
    for bin_name, target in min_per_bin.items():
        pool = binned.get(bin_name, [])
        if len(pool) <= target:
            result[bin_name] = pool
        else:
            indices = rng.choice(len(pool), size=target, replace=False)
            result[bin_name] = [pool[i] for i in indices]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Stratified Pairwise Interaction Dataset")
    parser.add_argument("--tiny", action="store_true",
                        help="Tiny mode: 1 frame, 20 pairs for quick verification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz")
    parser.add_argument("--frames", type=str, default="10,20",
                        help="Comma-separated frame indices")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = args.seed
    scene_name = args.scene
    frames_to_sample = [int(f) for f in args.frames.split(",")]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    output_dir = args.output_dir or os.path.join(
        repo_root, "results", "phase6_context_utility", "datasets"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*80}")
    print(f"  PHASE 6 — STRATIFIED PAIRWISE INTERACTION DATASET")
    print(f"{'='*80}")
    print(f"  Scene: {scene_name}")
    print(f"  Frames: {frames_to_sample}")
    print(f"  Seed: {seed}")
    print(f"  Device: {device}")
    print(f"  IoU Bins: {IOU_BINS}")
    print()

    # ─── Load protocol and data ───
    protocol = load_protocol()
    oracle_cfg = get_oracle_config(protocol)
    H, W = get_resolution(scene_name, protocol)

    camera = "freiburg2" if "fr2" in scene_name else "freiburg1"
    ds_cfg = get_dataset_config(scene_name, protocol)
    data_path = ds_cfg["full_path"]
    max_frame = max(frames_to_sample) + 1

    print(f"[Data] Loading {scene_name} ({max_frame} frames, {W}x{H})...")
    frames, intrinsics = load_sequence(data_path, camera, max_frame, H, W, device)
    print(f"[Data] Loaded {len(frames)} frames.")

    # ─── Build pipeline ───
    torch.manual_seed(seed)
    np.random.seed(seed)
    pipeline = build_pipeline(H, W, device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])

    print(f"\n[Pipeline] Running {max_frame-1} frames to build model...")
    for t in range(1, max_frame):
        pipeline.process_frame(frames[t]['rgb'], frames[t]['depth'], frames[t]['pose'])
        if t % 5 == 0 or t == max_frame - 1:
            N = pipeline.gaussian_model.num_gaussians
            print(f"  Frame {t}: {N} Gaussians")

    # ─── Configure oracle ───
    cond_config = ConditionalOracleConfig(n_opt_steps=oracle_cfg['n_opt_steps'])
    cond_oracle = ConditionalOracleExperiment(
        pipeline=pipeline,
        config=cond_config,
        oracle_config={
            'n_opt_steps': oracle_cfg['n_opt_steps'],
            'w_rgb': oracle_cfg['w_rgb'],
            'w_depth': oracle_cfg['w_depth'],
            'min_influence_pixels': oracle_cfg['min_influence_pixels'],
            'group_size': 1,
            'seed': seed,
        },
    )

    rng = np.random.default_rng(seed)
    all_records: List[Dict[str, Any]] = []
    frame_summaries: List[Dict[str, Any]] = []

    for frame_idx in frames_to_sample:
        print(f"\n{'─'*60}")
        print(f"  FRAME {frame_idx}")
        print(f"{'─'*60}")

        if frame_idx >= len(frames):
            print(f"  [SKIP] Frame {frame_idx} beyond loaded frames ({len(frames)})")
            continue

        rgb = frames[frame_idx]['rgb']
        depth = frames[frame_idx]['depth']
        model = pipeline.gaussian_model
        N = model.num_gaussians

        hash_before = hash_state(model)
        print(f"  N_gaussians: {N}")
        print(f"  State hash: {hash_before[:16]}...")

        # Render attribution
        print(f"  Computing attribution...")
        attr_out = render_with_attribution(
            means3D=model.positions,
            cov3D=model.build_covariance(),
            colors=model.get_colors(),
            opacities=model.opacities.squeeze(-1),
            extrinsics=pipeline.current_pose,
            intrinsics=pipeline.intrinsics,
            image_width=W, image_height=H,
            tile_size=pipeline.config.get('rendering', {}).get('tile_size', 16),
            top_k=pipeline.config.get('rendering', {}).get('attribution_top_k', 4),
        )
        contrib_indices = attr_out['contrib_indices']
        contrib_weights = attr_out['contrib_weights']

        # Get visible indices
        valid_attr = contrib_weights > 0.01
        visible_set = contrib_indices[valid_attr].unique().tolist() if valid_attr.any() else []
        visible_indices = [int(i) for i in visible_set if 0 <= int(i) < N]
        print(f"  Visible Gaussians: {len(visible_indices)}")

        # Compute all pairwise IoUs
        print(f"  Computing pairwise IoU...")
        t_iou_start = time.perf_counter()
        all_pairs = compute_all_pairwise_iou(
            visible_indices, contrib_indices, contrib_weights,
            threshold=cond_config.contribution_threshold,
            max_pairs=3000 if not args.tiny else 100,
        )
        t_iou = time.perf_counter() - t_iou_start
        print(f"  Computed {len(all_pairs)} pair IoUs in {t_iou:.1f}s")

        # Stratify by IoU bin
        pair_targets = MIN_PAIRS_PER_BIN.copy()
        if args.tiny:
            pair_targets = {k: min(5, v) for k, v in pair_targets.items()}

        stratified = stratified_sample_pairs(all_pairs, IOU_BINS, pair_targets, rng)

        for bin_name, pairs in stratified.items():
            lo, hi = IOU_BINS[bin_name]
            print(f"  [{bin_name:>10s}] IoU [{lo:.2f}, {hi:.2f}): "
                  f"{len(pairs)} pairs (target: {pair_targets[bin_name]})")

        # Measure pairwise interactions
        print(f"\n  Measuring pairwise interactions...")
        t_measure_start = time.perf_counter()
        frame_records = []

        for bin_name, pairs in stratified.items():
            for pair_idx, (idx_i, idx_j, iou_precomputed) in enumerate(pairs):
                result = cond_oracle.measure_pairwise_interaction(
                    idx_i=idx_i, idx_j=idx_j,
                    rgb_gt=rgb, depth_gt=depth,
                    contrib_indices=contrib_indices,
                    contrib_weights=contrib_weights,
                )

                # Get per-Gaussian features for additional analysis
                stats_i = compute_gaussian_statistics(
                    rendered_color=attr_out['color'],
                    rendered_depth=attr_out['depth'],
                    gt_color=rgb, gt_depth=depth,
                    contrib_weights=contrib_weights,
                    contrib_indices=contrib_indices,
                    n_gaussians=N,
                )

                record = {
                    # Identification
                    "scene": scene_name,
                    "frame": frame_idx,
                    "seed": seed,
                    "iou_bin": bin_name,
                    "idx_i": int(idx_i),
                    "idx_j": int(idx_j),

                    # Quality measurements
                    "delta_q_i": result["delta_q_i"],
                    "delta_q_j": result["delta_q_j"],
                    "delta_q_ij": result["delta_q_ij"],
                    "interaction_residual": result["interaction_residual"],
                    "is_sub_additive": result["is_sub_additive"],
                    "is_super_additive": result["is_super_additive"],
                    "additivity_ratio": result["additivity_ratio"],

                    # Geometric features
                    "overlap_iou": result["overlap_iou"],
                    "iou_precomputed": iou_precomputed,
                    "distance_3d": result["distance_3d"],

                    # Per-Gaussian statistics
                    "influence_mass_i": float(stats_i['influence_mass'][idx_i]),
                    "influence_mass_j": float(stats_i['influence_mass'][idx_j]),
                    "rgb_error_i": float(stats_i['color_error'][idx_i]),
                    "rgb_error_j": float(stats_i['color_error'][idx_j]),
                    "depth_error_i": float(stats_i['depth_error'][idx_i]) if 'depth_error' in stats_i else 0.0,
                    "depth_error_j": float(stats_i['depth_error'][idx_j]) if 'depth_error' in stats_i else 0.0,
                    "pixel_count_i": int(stats_i['pixel_count'][idx_i]),
                    "pixel_count_j": int(stats_i['pixel_count'][idx_j]),

                    # Co-visibility features
                    "pixel_intersection": int(
                        (_get_pixel_mask(idx_i, contrib_indices, contrib_weights, 0.01)
                         & _get_pixel_mask(idx_j, contrib_indices, contrib_weights, 0.01)).sum().item()
                    ),
                    "pixel_union": int(
                        (_get_pixel_mask(idx_i, contrib_indices, contrib_weights, 0.01)
                         | _get_pixel_mask(idx_j, contrib_indices, contrib_weights, 0.01)).sum().item()
                    ),
                }

                frame_records.append(record)

                if (pair_idx + 1) % 20 == 0:
                    print(f"    {bin_name}: {pair_idx+1}/{len(pairs)} pairs measured")

        t_measure = time.perf_counter() - t_measure_start
        print(f"  Measured {len(frame_records)} interactions in {t_measure:.1f}s")

        # Verify state integrity
        hash_after = hash_state(model)
        state_ok = (hash_before == hash_after)
        print(f"  State integrity: {'✓ PASS' if state_ok else '✗ FAIL'}")

        # Frame summary
        bin_summary = {}
        for bin_name in IOU_BINS:
            bin_records = [r for r in frame_records if r["iou_bin"] == bin_name]
            if bin_records:
                interactions = [r["interaction_residual"] for r in bin_records]
                bin_summary[bin_name] = {
                    "n_pairs": len(bin_records),
                    "mean_interaction": float(np.mean(interactions)),
                    "std_interaction": float(np.std(interactions)),
                    "median_interaction": float(np.median(interactions)),
                    "sub_additive_fraction": float(np.mean([r["is_sub_additive"] for r in bin_records])),
                    "mean_iou": float(np.mean([r["overlap_iou"] for r in bin_records])),
                    "mean_additivity_ratio": float(np.mean([r["additivity_ratio"] for r in bin_records])),
                }
            else:
                bin_summary[bin_name] = {"n_pairs": 0}

        frame_summaries.append({
            "frame": frame_idx,
            "n_pairs_total": len(frame_records),
            "state_integrity": state_ok,
            "time_iou_s": t_iou,
            "time_measure_s": t_measure,
            "bins": bin_summary,
        })

        all_records.extend(frame_records)

    # ─── Save dataset ───
    output_path = os.path.join(output_dir, f"pairwise_interaction_seed_{seed}.json")
    print(f"\n[Save] Writing {len(all_records)} pair records to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_records, f, indent=2, default=str)

    # ─── Summary ───
    summary_path = os.path.join(output_dir, f"pairwise_summary_seed_{seed}.json")
    summary = {
        "total_pairs": len(all_records),
        "scene": scene_name,
        "frames": frames_to_sample,
        "seed": seed,
        "iou_bins": IOU_BINS,
        "frame_summaries": frame_summaries,
        "aggregate": {},
    }

    # Aggregate across frames
    for bin_name in IOU_BINS:
        bin_records = [r for r in all_records if r["iou_bin"] == bin_name]
        if bin_records:
            interactions = [r["interaction_residual"] for r in bin_records]
            summary["aggregate"][bin_name] = {
                "n_pairs": len(bin_records),
                "mean_interaction": float(np.mean(interactions)),
                "std_interaction": float(np.std(interactions)),
                "sub_additive_fraction": float(np.mean([r["is_sub_additive"] for r in bin_records])),
                "super_additive_fraction": float(np.mean([r["is_super_additive"] for r in bin_records])),
                "mean_iou": float(np.mean([r["overlap_iou"] for r in bin_records])),
                "mean_distance_3d": float(np.mean([r["distance_3d"] for r in bin_records])),
                "mean_additivity_ratio": float(np.mean([r["additivity_ratio"] for r in bin_records])),
                "correlation_iou_vs_interaction": float(
                    np.corrcoef(
                        [r["overlap_iou"] for r in bin_records],
                        [r["interaction_residual"] for r in bin_records]
                    )[0, 1]
                ) if len(bin_records) > 2 else 0.0,
            }
        else:
            summary["aggregate"][bin_name] = {"n_pairs": 0}

    # Overall correlation: IoU vs |I|
    if len(all_records) > 5:
        ious = [r["overlap_iou"] for r in all_records]
        interactions = [abs(r["interaction_residual"]) for r in all_records]
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(ious, interactions)
        summary["overall_iou_interaction_spearman_rho"] = float(rho)
        summary["overall_iou_interaction_spearman_p"] = float(p_val)
        summary["hypothesis_iou_drives_interaction"] = bool(rho > 0 and p_val < 0.05)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  PAIRWISE INTERACTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total pairs: {len(all_records)}")
    for bin_name, stats in summary.get("aggregate", {}).items():
        n = stats.get("n_pairs", 0)
        if n > 0:
            print(f"  [{bin_name:>10s}] N={n:4d}  "
                  f"I_mean={stats['mean_interaction']:.2e}  "
                  f"sub_add={stats['sub_additive_fraction']:.1%}  "
                  f"IoU_mean={stats['mean_iou']:.3f}")
        else:
            print(f"  [{bin_name:>10s}] N=   0  (no pairs found)")

    if "overall_iou_interaction_spearman_rho" in summary:
        rho = summary["overall_iou_interaction_spearman_rho"]
        p = summary["overall_iou_interaction_spearman_p"]
        confirmed = summary["hypothesis_iou_drives_interaction"]
        print(f"\n  Spearman(IoU, |I|): ρ={rho:.4f}, p={p:.4f}")
        print(f"  Hypothesis 'IoU drives interaction': {'✓ CONFIRMED' if confirmed else '✗ NOT CONFIRMED'}")

    print(f"\n  Saved: {output_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
