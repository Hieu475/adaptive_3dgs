#!/usr/bin/env python3
"""Phase 6 — Conditional Oracle Dataset Builder (Prototype).

Generates a conditional marginal utility dataset for training context-aware
utility models. For each candidate Gaussian, measures U*(i|S) across multiple
context sets S with varying types and sizes.

Usage:
    python experiments/build_phase6_dataset.py                   # Full prototype
    python experiments/build_phase6_dataset.py --tiny             # Tiny verification (2 frames, 10 candidates)
    python experiments/build_phase6_dataset.py --max-candidates 50 --frames 10,20,30

Output:
    results/phase6_context_utility/datasets/
    ├── conditional_oracle_seed_42.json
    ├── dataset_summary.json
    └── prototype_verification.json

Invariants:
    - Uses frozen Phase 4 canonical features (11-dim) as self features.
    - Phase 5 is NOT modified; this script only reads from it.
    - All measurements use snapshot/restore isolation (non-destructive).
    - Context types: empty (S=∅), spatial_knn, overlap_top, random.
    - Quality Q(·) uses global metrics (full-frame), same as Phase 3/4/5.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.tum_dataset import TUMDataset
from research.pipeline import OnlineReconstructionPipeline
from research.oracle_utility import OracleUtilityExperiment
from research.phase6_context import (
    build_full_context_batch,
    ContextConfig,
    PHASE6_FEATURE_NAMES,
    PHASE6_FEATURE_DIM,
)
from research.phase6_oracle import (
    ConditionalOracleExperiment,
    ConditionalOracleConfig,
)
from research.utility_features import extract_feature_vector, CANONICAL_FEATURE_NAMES
from research.protocol import (
    load_protocol,
    get_seeds,
    get_resolution,
    get_dataset_config,
    get_oracle_config,
)
from research.attribution import render_with_attribution, compute_gaussian_statistics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_state(model, optimizer=None):
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
            'pose': item['pose'].to(device)
        })

    return frames, intrinsics


def build_pipeline(H, W, device):
    """Build OnlineReconstructionPipeline with standard config."""
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
    return OnlineReconstructionPipeline(config=config, device=device)


def extract_all_features(pipeline, rgb, depth) -> np.ndarray:
    """Extract canonical 11-dim features for ALL Gaussians in the model.

    Uses the same attribution-based feature computation as Phase 3/4.

    Returns:
        (N, 11) numpy array of canonical features.
    """
    model = pipeline.gaussian_model
    N = model.num_gaussians

    if N == 0:
        return np.zeros((0, 11), dtype=np.float32)

    H, W = rgb.shape[:2]

    # Render with attribution to get per-Gaussian statistics
    attr_out = render_with_attribution(
        means3D=model.positions,
        cov3D=model.build_covariance(),
        colors=model.get_colors(),
        opacities=model.opacities.squeeze(-1),
        extrinsics=pipeline.current_pose,
        intrinsics=pipeline.intrinsics,
        image_width=W,
        image_height=H,
        tile_size=pipeline.config.get('rendering', {}).get('tile_size', 16),
        top_k=pipeline.config.get('rendering', {}).get('attribution_top_k', 4),
    )

    contrib_indices = attr_out['contrib_indices']
    contrib_weights = attr_out['contrib_weights']

    # Compute per-Gaussian error and attribution statistics
    stats = compute_gaussian_statistics(
        rendered_color=attr_out['color'],
        rendered_depth=attr_out['depth'],
        gt_color=rgb,
        gt_depth=depth,
        contrib_weights=contrib_weights,
        contrib_indices=contrib_indices,
        n_gaussians=N,
    )

    # Get state store data
    store = getattr(model, 'state_store', None)

    features = np.zeros((N, 11), dtype=np.float32)
    for i in range(N):
        features[i, 0] = float(stats['color_error'][i])     # rgb_error
        features[i, 1] = float(stats['depth_error'][i])      # depth_error
        # gradient_norm: approximate as influence * error
        features[i, 2] = float(
            (stats['color_error'][i] + stats['depth_error'][i]) * stats['influence_mass'][i]
        )
        features[i, 3] = float(stats['pixel_count'][i])      # visibility_count
        features[i, 4] = float(stats['influence_mass'][i])    # influence_mass

        if store is not None and i < len(store.position_drift):
            features[i, 5] = float(store.position_drift[i].item())
            features[i, 6] = float(store.residual_drift_ema[i].item())
            features[i, 7] = float(store.uncertainty[i].item())
            age_val = max(1, int(store.ages[i].item()))
            features[i, 9] = float(store.update_counts[i].item()) / age_val
            features[i, 10] = float(store.ages[i].item())

        features[i, 8] = float(stats['projected_area'][i])   # projected_area

    return features


def sample_candidates(pipeline, n_candidates, seed=42):
    """Sample candidate Gaussian indices using geometry-stratified sampling.

    Falls back to uniform random if not enough visible Gaussians.
    """
    model = pipeline.gaussian_model
    N = model.num_gaussians
    n_sample = min(n_candidates, N)

    if n_sample <= 0:
        return []

    rng = np.random.default_rng(seed)
    return rng.choice(N, size=n_sample, replace=False).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Conditional Oracle Dataset Builder")
    parser.add_argument("--tiny", action="store_true",
                        help="Tiny mode: 2 frames, 10 candidates for quick verification")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max-candidates", type=int, default=25,
                        help="Max candidates per frame (default: 25)")
    parser.add_argument("--frames", type=str, default="10,20",
                        help="Comma-separated frame indices to evaluate (default: 10,20)")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz",
                        help="Scene name (default: tum_fr2_xyz for prototype)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/phase6_context_utility/datasets)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=" * 80)
    print(f"  PHASE 6 — CONDITIONAL ORACLE DATASET BUILDER [Device: {device}]")
    print(f"=" * 80)

    # ─── Configuration ───
    if args.tiny:
        frames_to_sample = [10, 20]
        max_candidates = 10
        print(f"  MODE: TINY PROTOTYPE (2 frames, {max_candidates} candidates)")
    else:
        frames_to_sample = [int(f) for f in args.frames.split(",")]
        max_candidates = args.max_candidates
        print(f"  MODE: STANDARD ({len(frames_to_sample)} frames, {max_candidates} candidates/frame)")

    seed = args.seed
    scene_name = args.scene
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(repo_root, "results", "phase6_context_utility", "datasets")
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Seed: {seed}")
    print(f"  Scene: {scene_name}")
    print(f"  Frames: {frames_to_sample}")
    print(f"  Max candidates/frame: {max_candidates}")
    print(f"  Output: {output_dir}")
    print(f"  Feature dim: {PHASE6_FEATURE_DIM}")
    print()

    # ─── Load protocol and data ───
    protocol = load_protocol()
    oracle_cfg = get_oracle_config(protocol)
    H, W = get_resolution(scene_name, protocol)

    # Determine camera type
    if "fr1" in scene_name:
        camera = "freiburg1"
    elif "fr2" in scene_name:
        camera = "freiburg2"
    else:
        camera = "freiburg1"

    ds_cfg = get_dataset_config(scene_name, protocol)
    data_path = ds_cfg["full_path"]
    max_frame = max(frames_to_sample) + 1

    print(f"[Data] Loading {scene_name} ({max_frame} frames, {W}x{H})...")
    frames, intrinsics = load_sequence(data_path, camera, max_frame, H, W, device)
    print(f"[Data] Loaded {len(frames)} frames.")

    # ─── Build pipeline and warm up ───
    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = build_pipeline(H, W, device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])

    # Process frames sequentially to build up Gaussian model
    print(f"\n[Pipeline] Running {max_frame-1} frames to build model...")
    for t in range(1, max_frame):
        pipeline.process_frame(frames[t]['rgb'], frames[t]['depth'], frames[t]['pose'])
        if t % 5 == 0 or t == max_frame - 1:
            N = pipeline.gaussian_model.num_gaussians
            print(f"  Frame {t}: {N} Gaussians")

    # ─── Configure conditional oracle ───
    cond_config = ConditionalOracleConfig(
        n_opt_steps=oracle_cfg['n_opt_steps'],
        k_neighbors=8,
        epsilon=1e-6,
        contribution_threshold=0.01,
    )

    cond_oracle = ConditionalOracleExperiment(
        pipeline=pipeline,
        config=cond_config,
        oracle_config={
            'n_samples': max_candidates,
            'n_opt_steps': oracle_cfg['n_opt_steps'],
            'w_rgb': oracle_cfg['w_rgb'],
            'w_depth': oracle_cfg['w_depth'],
            'min_influence_pixels': oracle_cfg['min_influence_pixels'],
            'group_size': 1,
            'seed': seed,
        },
    )

    # ─── Generate dataset ───
    all_samples: List[Dict[str, Any]] = []
    frame_stats: List[Dict[str, Any]] = []

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

        # Verify state integrity before measurement
        hash_before = hash_state(model)
        print(f"  N_gaussians: {N}")
        print(f"  State hash (before): {hash_before[:16]}...")

        # Extract canonical features for ALL Gaussians
        print(f"  Extracting features for {N} Gaussians...")
        all_features = extract_all_features(pipeline, rgb, depth)
        print(f"  Feature shape: {all_features.shape}")

        # Get attribution for overlap computation
        print(f"  Computing attribution...")
        attr_out = render_with_attribution(
            means3D=model.positions,
            cov3D=model.build_covariance(),
            colors=model.get_colors(),
            opacities=model.opacities.squeeze(-1),
            extrinsics=pipeline.current_pose,
            intrinsics=pipeline.intrinsics,
            image_width=W,
            image_height=H,
            tile_size=pipeline.config.get('rendering', {}).get('tile_size', 16),
            top_k=pipeline.config.get('rendering', {}).get('attribution_top_k', 4),
        )
        contrib_indices = attr_out['contrib_indices']
        contrib_weights = attr_out['contrib_weights']

        # Sample candidates
        candidates = sample_candidates(pipeline, max_candidates, seed=seed + frame_idx)
        print(f"  Candidates: {len(candidates)}")

        # Determine split
        if scene_name == "tum_fr2_xyz":
            split = "cross_scene_test"
        elif frame_idx <= 40:
            split = "train"
        else:
            split = "validation"

        # Generate conditional measurements
        t_start = time.perf_counter()
        frame_samples = cond_oracle.generate_conditional_dataset(
            candidate_pool=candidates,
            rgb_gt=rgb,
            depth_gt=depth,
            contrib_indices=contrib_indices,
            contrib_weights=contrib_weights,
            all_features=all_features,
            scene_name=scene_name,
            frame_idx=frame_idx,
            split=split,
            seed=seed,
            max_candidates=max_candidates,
        )
        elapsed = time.perf_counter() - t_start

        # Verify state integrity after measurement
        hash_after = hash_state(model)
        state_ok = (hash_before == hash_after)
        print(f"  State hash (after):  {hash_after[:16]}...")
        print(f"  State integrity:     {'✓ PASS' if state_ok else '✗ FAIL'}")
        print(f"  Samples generated:   {len(frame_samples)}")
        print(f"  Time:                {elapsed:.1f}s ({elapsed/max(len(frame_samples),1):.2f}s/sample)")

        if not state_ok:
            print(f"  ⚠ WARNING: State corruption detected! Snapshot/restore failed.")

        # Collect statistics
        if frame_samples:
            dqs = [s["delta_q_conditional"] for s in frame_samples]
            utils = [s["utility_conditional"] for s in frame_samples]
            context_sizes = [s["context_size"] for s in frame_samples]

            frame_stats.append({
                "frame": frame_idx,
                "n_candidates": len(candidates),
                "n_samples": len(frame_samples),
                "n_gaussians": N,
                "state_integrity": state_ok,
                "time_s": elapsed,
                "delta_q_mean": float(np.mean(dqs)),
                "delta_q_std": float(np.std(dqs)),
                "delta_q_min": float(np.min(dqs)),
                "delta_q_max": float(np.max(dqs)),
                "utility_mean": float(np.mean(utils)),
                "utility_std": float(np.std(utils)),
                "positive_utility_frac": float(np.mean([u > 0 for u in utils])),
                "context_size_distribution": {
                    str(s): int(context_sizes.count(s))
                    for s in sorted(set(context_sizes))
                },
            })

        all_samples.extend(frame_samples)

    # ─── Save dataset ───
    dataset_path = os.path.join(output_dir, f"conditional_oracle_seed_{seed}.json")
    print(f"\n[Save] Writing {len(all_samples)} samples to {dataset_path}")
    with open(dataset_path, 'w') as f:
        json.dump(all_samples, f, indent=2, default=str)

    # ─── Verification checks ───
    print(f"\n{'='*60}")
    print(f"  PROTOTYPE VERIFICATION")
    print(f"{'='*60}")

    verification = {
        "total_samples": len(all_samples),
        "frames_evaluated": frames_to_sample,
        "scene": scene_name,
        "seed": seed,
        "feature_dim": PHASE6_FEATURE_DIM,
        "checks": {},
    }

    # Check 1: All samples have correct feature vector length
    all_vectors_correct = all(
        len(s["full_feature_vector"]) == PHASE6_FEATURE_DIM
        for s in all_samples
    )
    verification["checks"]["feature_vector_dim_32"] = all_vectors_correct
    print(f"  [{'✓' if all_vectors_correct else '✗'}] Feature vector dim = {PHASE6_FEATURE_DIM}")

    # Check 2: Empty context recovers marginal utility (dq_s = 0 when |S| = 0)
    empty_samples = [s for s in all_samples if s["context_size"] == 0]
    if empty_samples:
        all_dq_s_zero = all(abs(s["delta_q_s"]) < 1e-10 for s in empty_samples)
        verification["checks"]["empty_context_dq_s_zero"] = all_dq_s_zero
        print(f"  [{'✓' if all_dq_s_zero else '✗'}] S=∅ ⟹ ΔQ(S) = 0 ({len(empty_samples)} samples)")

        # delta_q_conditional should equal delta_q_si for empty context
        identity_ok = all(
            abs(s["delta_q_conditional"] - s["delta_q_si"]) < 1e-10
            for s in empty_samples
        )
        verification["checks"]["empty_context_identity"] = identity_ok
        print(f"  [{'✓' if identity_ok else '✗'}] S=∅ ⟹ ΔQ(i|∅) = ΔQ({{i}})")

    # Check 3: ΔQ identity: delta_q_conditional = delta_q_si - delta_q_s
    identity_errors = []
    for s in all_samples:
        expected = s["delta_q_si"] - s["delta_q_s"]
        actual = s["delta_q_conditional"]
        identity_errors.append(abs(expected - actual))

    max_identity_error = max(identity_errors) if identity_errors else 0.0
    identity_pass = max_identity_error < 1e-8
    verification["checks"]["delta_q_identity"] = identity_pass
    verification["checks"]["delta_q_identity_max_error"] = max_identity_error
    print(f"  [{'✓' if identity_pass else '✗'}] ΔQ(i|S) = ΔQ(S∪{{i}}) - ΔQ(S)  (max err: {max_identity_error:.2e})")

    # Check 4: Different context sizes exist
    context_sizes = set(s["context_size"] for s in all_samples)
    has_variety = len(context_sizes) >= 2
    verification["checks"]["context_size_variety"] = has_variety
    print(f"  [{'✓' if has_variety else '✗'}] Context size variety: {sorted(context_sizes)}")

    # Check 5: Multiple context types exist
    context_types = set(s["context_type"] for s in all_samples)
    has_type_variety = len(context_types) >= 2
    verification["checks"]["context_type_variety"] = has_type_variety
    print(f"  [{'✓' if has_type_variety else '✗'}] Context type variety: {sorted(context_types)}")

    # Check 6: No NaN in features
    nan_count = sum(
        1 for s in all_samples
        if any(np.isnan(v) for v in s["full_feature_vector"])
    )
    no_nans = (nan_count == 0)
    verification["checks"]["no_nan_features"] = no_nans
    print(f"  [{'✓' if no_nans else '✗'}] No NaN in features ({nan_count} NaN samples)")

    # Check 7: State integrity across all frames
    all_integrity = all(fs["state_integrity"] for fs in frame_stats)
    verification["checks"]["state_integrity"] = all_integrity
    print(f"  [{'✓' if all_integrity else '✗'}] State integrity (snapshot/restore)")

    # Overall verdict
    all_pass = all(
        v for k, v in verification["checks"].items()
        if isinstance(v, bool)
    )
    verification["overall_pass"] = all_pass
    print(f"\n  {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")

    # Save verification
    verification_path = os.path.join(output_dir, "prototype_verification.json")
    with open(verification_path, 'w') as f:
        json.dump(verification, f, indent=2)

    # Save summary
    summary = {
        "phase": "Phase 6 — Conditional Oracle Dataset",
        "scene": scene_name,
        "seed": seed,
        "total_samples": len(all_samples),
        "feature_dim": PHASE6_FEATURE_DIM,
        "feature_names": PHASE6_FEATURE_NAMES,
        "frame_stats": frame_stats,
        "verification": verification,
        "config": {
            "n_opt_steps": cond_config.n_opt_steps,
            "k_neighbors": cond_config.k_neighbors,
            "context_sizes": cond_config.context_sizes,
            "context_size_weights": cond_config.context_size_weights,
            "context_types": cond_config.context_types,
        },
    }
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Done] Saved:")
    print(f"  Dataset:      {dataset_path}")
    print(f"  Verification: {verification_path}")
    print(f"  Summary:      {summary_path}")


if __name__ == "__main__":
    main()
