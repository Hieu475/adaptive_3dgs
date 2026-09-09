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
from typing import Dict, List, Any, Optional, Tuple

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


def sample_candidates(pipeline, n_candidates, visible_indices=None, seed=42):
    """Sample candidate Gaussian indices from visible Gaussians if available."""
    model = pipeline.gaussian_model
    N = model.num_gaussians

    if visible_indices is not None and len(visible_indices) > 0:
        pool = [int(i) for i in visible_indices if 0 <= int(i) < N]
        if len(pool) == 0:
            pool = list(range(N))
    else:
        pool = list(range(N))

    n_sample = min(n_candidates, len(pool))
    if n_sample <= 0:
        return []

    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_sample, replace=False).tolist()
    return sorted([int(x) for x in chosen])


# ─────────────────────────────────────────────────────────────────────────────
# Scene Sample Generation & Dataset Verification
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples_for_scene(
    scene_name: str,
    frames_to_sample: List[int],
    max_candidates: int,
    seed: int,
    device: str,
    protocol: Dict[str, Any],
    context_specs: Optional[List[Tuple[str, int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate context-centric conditional oracle samples for a given scene and frame list."""
    oracle_cfg = get_oracle_config(protocol)
    H, W = get_resolution(scene_name, protocol)

    camera = "freiburg1" if "fr1" in scene_name else ("freiburg2" if "fr2" in scene_name else "freiburg1")
    ds_cfg = get_dataset_config(scene_name, protocol)
    data_path = ds_cfg["full_path"]
    max_frame = max(frames_to_sample) + 1

    print(f"\n[Data] Loading {scene_name} ({max_frame} frames, {W}x{H})...")
    frames, intrinsics = load_sequence(data_path, camera, max_frame, H, W, device)
    print(f"[Data] Loaded {len(frames)} frames.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    pipeline = build_pipeline(H, W, device)
    pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0]['pose'])

    print(f"[Pipeline] Running {max_frame-1} frames to build model...")
    for t in range(1, max_frame):
        pipeline.process_frame(frames[t]['rgb'], frames[t]['depth'], frames[t]['pose'])
        if t % 10 == 0 or t == max_frame - 1:
            N = pipeline.gaussian_model.num_gaussians
            print(f"  Frame {t}: {N} Gaussians")

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

    if context_specs is None:
        context_specs = [
            ("empty", 0),
            ("spatial_knn", 1),
            ("spatial_knn", 4),
            ("random", 8),
        ]

    scene_samples: List[Dict[str, Any]] = []
    frame_stats: List[Dict[str, Any]] = []

    for frame_idx in frames_to_sample:
        print(f"\n{'─'*60}")
        print(f"  SCENE: {scene_name} | FRAME {frame_idx}")
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
        print(f"  State hash (before): {hash_before[:16]}...")

        print(f"  Extracting features for {N} Gaussians...")
        all_features = extract_all_features(pipeline, rgb, depth)

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

        valid_attr = contrib_weights > 0.01
        visible_indices = contrib_indices[valid_attr].unique().tolist() if valid_attr.any() else None

        candidates = sample_candidates(pipeline, max_candidates, visible_indices=visible_indices, seed=seed + frame_idx)
        print(f"  Candidates: {len(candidates)} (sampled from {len(visible_indices) if visible_indices else N} visible)")

        if scene_name == "tum_fr2_xyz":
            split = "cross_scene_test"
        elif frame_idx <= 40:
            split = "train"
        else:
            split = "validation"

        t_start = time.perf_counter()
        frame_samples = cond_oracle.generate_context_centric_dataset(
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
            visible_indices=visible_indices,
            context_specs=context_specs,
        )
        elapsed = time.perf_counter() - t_start

        hash_after = hash_state(model)
        state_ok = (hash_before == hash_after)
        print(f"  State hash (after):  {hash_after[:16]}...")
        print(f"  State integrity:     {'✓ PASS' if state_ok else '✗ FAIL'}")
        print(f"  Samples generated:   {len(frame_samples)}")
        print(f"  Time:                {elapsed:.1f}s ({elapsed/max(len(frame_samples),1):.3f}s/sample)")

        if not state_ok:
            print(f"  ⚠ WARNING: State corruption detected! Snapshot/restore failed.")

        if frame_samples:
            dqs = [s["delta_q_conditional"] for s in frame_samples]
            utils = [s["utility_conditional"] for s in frame_samples]
            context_sizes = [s["context_size"] for s in frame_samples]

            frame_stats.append({
                "scene": scene_name,
                "frame": frame_idx,
                "split": split,
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
                "delta_q_single_mean": float(np.mean([s.get("delta_q_single", 0) for s in frame_samples])),
                "delta_q_single_std": float(np.std([s.get("delta_q_single", 0) for s in frame_samples])),
                "utility_single_mean": float(np.mean([s.get("utility_single", 0) for s in frame_samples])),
                "n_context_types": len(set(s["context_type"] for s in frame_samples)),
                "context_types_seen": sorted(list(set(s["context_type"] for s in frame_samples))),
                "context_size_distribution": {
                    str(s): int(context_sizes.count(s))
                    for s in sorted(set(context_sizes))
                },
            })

        scene_samples.extend(frame_samples)

    return scene_samples, frame_stats


def verify_dataset(
    all_samples: List[Dict[str, Any]],
    frame_stats: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    """Verify core Phase 6 mathematical invariants and 100% candidate coverage."""
    verification = {
        "total_samples": len(all_samples),
        "seed": seed,
        "feature_dim": PHASE6_FEATURE_DIM,
        "has_single_measurements": all("delta_q_single" in s for s in all_samples),
        "has_utility_single": all("utility_single" in s for s in all_samples),
        "has_candidate_pool_ids": all("candidate_pool_ids" in s for s in all_samples),
        "context_types_used": sorted(list(set(s["context_type"] for s in all_samples))),
        "context_sizes_used": sorted(list(set(s["context_size"] for s in all_samples))),
        "samples_per_context_type": {
            ct: sum(1 for s in all_samples if s["context_type"] == ct)
            for ct in sorted(set(s["context_type"] for s in all_samples))
        },
        "checks": {},
    }

    # Check 1: Feature vector length 32
    all_vectors_correct = all(
        len(s["full_feature_vector"]) == PHASE6_FEATURE_DIM
        for s in all_samples
    )
    verification["checks"]["feature_vector_dim_32"] = all_vectors_correct
    print(f"  [{'✓' if all_vectors_correct else '✗'}] Feature vector dim = {PHASE6_FEATURE_DIM}")

    # Check 2: Empty context recovers marginal utility
    empty_samples = [s for s in all_samples if s["context_size"] == 0]
    if empty_samples:
        all_dq_s_zero = all(abs(s["delta_q_s"]) < 1e-10 for s in empty_samples)
        verification["checks"]["empty_context_dq_s_zero"] = all_dq_s_zero
        print(f"  [{'✓' if all_dq_s_zero else '✗'}] S=∅ ⟹ ΔQ(S) = 0 ({len(empty_samples)} samples)")

        identity_ok = all(
            abs(s["delta_q_conditional"] - s["delta_q_si"]) < 1e-10
            for s in empty_samples
        )
        verification["checks"]["empty_context_identity"] = identity_ok
        print(f"  [{'✓' if identity_ok else '✗'}] S=∅ ⟹ ΔQ(i|∅) = ΔQ({{i}})")

    # Check 3: ΔQ identity: delta_q_conditional = delta_q_si - delta_q_s
    identity_errors = [
        abs((s["delta_q_si"] - s["delta_q_s"]) - s["delta_q_conditional"])
        for s in all_samples
    ]
    max_err = max(identity_errors) if identity_errors else 0.0
    id_pass = max_err < 1e-8
    verification["checks"]["delta_q_identity"] = id_pass
    verification["checks"]["delta_q_identity_max_error"] = max_err
    print(f"  [{'✓' if id_pass else '✗'}] ΔQ(i|S) = ΔQ(S∪{{i}}) - ΔQ(S) (max err: {max_err:.2e})")

    # Check 4: No NaN in features
    nan_count = sum(1 for s in all_samples if any(np.isnan(v) for v in s["full_feature_vector"]))
    verification["checks"]["no_nan_features"] = (nan_count == 0)
    print(f"  [{'✓' if nan_count == 0 else '✗'}] No NaN in features ({nan_count} NaN samples)")

    # Check 5: State integrity
    all_integrity = all(fs["state_integrity"] for fs in frame_stats)
    verification["checks"]["state_integrity"] = all_integrity
    print(f"  [{'✓' if all_integrity else '✗'}] State integrity (snapshot/restore)")

    # Check 6: 100% Candidate Pool Coverage across all exact non-empty groups
    exact_groups: Dict[Tuple[str, int, Tuple[int, ...]], List[Dict[str, Any]]] = {}
    for s in all_samples:
        if s.get("context_size", 0) > 0:
            k = (str(s["scene"]), int(s["frame"]), tuple(sorted(int(x) for x in s.get("context_ids", []))))
            exact_groups.setdefault(k, []).append(s)

    full_cov_groups = 0
    missing_count = 0
    dup_count = 0
    for k, s_list in exact_groups.items():
        pool_ids = set(s_list[0].get("candidate_pool_ids", []))
        measured_ids = [int(s["candidate_id"]) for s in s_list]
        unique_measured = set(measured_ids)
        if len(pool_ids) > 0:
            missing = len(pool_ids - unique_measured)
            dups = len(measured_ids) - len(unique_measured)
            missing_count += missing
            dup_count += dups
            if missing == 0 and dups == 0 and len(unique_measured) == len(pool_ids):
                full_cov_groups += 1

    cov_rate = full_cov_groups / max(len(exact_groups), 1)
    is_100pct_coverage = (full_cov_groups == len(exact_groups) and len(exact_groups) > 0)
    verification["checks"]["exact_context_full_candidate_coverage"] = is_100pct_coverage
    verification["checks"]["exact_context_groups_count"] = len(exact_groups)
    verification["checks"]["full_coverage_groups_count"] = full_cov_groups
    verification["checks"]["total_missing_candidates"] = missing_count
    verification["checks"]["total_duplicate_candidates"] = dup_count
    print(f"  [{'✓' if is_100pct_coverage else '✗'}] Exact Groups Coverage: {full_cov_groups}/{len(exact_groups)} ({cov_rate:.1%}), missing={missing_count}, dups={dup_count}")

    # Check 7: Candidate pool consistency within exact groups
    pool_consistent = all(
        tuple(s_list[0].get("candidate_pool_ids", [])) == tuple(s.get("candidate_pool_ids", []))
        for k, s_list in exact_groups.items() for s in s_list
    )
    verification["checks"]["candidate_pool_consistent_within_group"] = pool_consistent
    print(f"  [{'✓' if pool_consistent else '✗'}] Candidate pool consistent across all records in same group")

    # Check 8: Context S_t strictly disjoint from candidate pool P_t (S_t ∩ P_t = ∅)
    disjoint_ok = all(
        len(set(s.get("context_ids", [])) & set(s.get("candidate_pool_ids", []))) == 0
        for s in all_samples
    )
    verification["checks"]["context_disjoint_from_pool"] = disjoint_ok
    print(f"  [{'✓' if disjoint_ok else '✗'}] S_t ∩ P_t = ∅ across all {len(all_samples)} samples")

    # Check 9: Cost validity audit
    invalid_cost = sum(1 for s in all_samples if not s.get("delta_t_valid", True))
    neg_dt = sum(1 for s in all_samples if s.get("delta_t_conditional_ms", 1.0) < 0)
    zero_dt = sum(1 for s in all_samples if s.get("delta_t_conditional_ms", 1.0) == 0)
    verification["cost_diagnostics"] = {
        "invalid_cost_count": invalid_cost,
        "negative_delta_t_count": neg_dt,
        "zero_delta_t_count": zero_dt,
        "valid_cost_fraction": float(1.0 - invalid_cost / max(len(all_samples), 1)),
    }
    verification["checks"]["no_unhandled_invalid_cost"] = all(
        "effective_delta_t_ms" in s and s["effective_delta_t_ms"] > 0 for s in all_samples
    )
    print(f"  [{'✓' if verification['checks']['no_unhandled_invalid_cost'] else '✗'}] Cost validity: {invalid_cost}/{len(all_samples)} invalid raw ΔT (neg={neg_dt}, zero={zero_dt}); all handled via effective ΔT > 0")

    all_pass = all(v for k, v in verification["checks"].items() if isinstance(v, bool))
    verification["overall_pass"] = all_pass
    print(f"\n  {'✓ ALL CHECKS PASSED' if all_pass else '✗ SOME CHECKS FAILED'}")
    return verification


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Conditional Oracle Dataset Builder")
    parser.add_argument("--tiny", action="store_true",
                        help="Tiny mode: 2 frames, 10 candidates for quick verification")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed generation (e.g. 42,43,44,45,46)")
    parser.add_argument("--max-candidates", type=int, default=20,
                        help="Max candidates per frame (default: 20)")
    parser.add_argument("--frames", type=str, default="10,20,30",
                        help="Comma-separated frame indices to evaluate (default: 10,20,30)")
    parser.add_argument("--scene", type=str, default="tum_fr2_xyz",
                        help="Scene name (default: tum_fr2_xyz)")
    parser.add_argument("--protocol-splits", action="store_true",
                        help="Generate ALL protocol splits: tum_fr1_desk (train 0-40, val 41-60) "
                             "AND tum_fr2_xyz (test). Overrides --scene and --frames.")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Append samples to existing dataset file if it exists, merging without overwriting.")
    parser.add_argument("--output-filename", type=str, default=None,
                        help="Custom filename for the output json dataset.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/phase6_context_utility/datasets)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=" * 80)
    print(f"  PHASE 6 — CONTEXT-CENTRIC CONDITIONAL ORACLE DATASET BUILDER [Device: {device}]")
    print(f"=" * 80)

    protocol = load_protocol()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(repo_root, "results", "phase6_context_utility", "datasets")
    os.makedirs(output_dir, exist_ok=True)

    context_specs = [
        ("empty", 0),
        ("spatial_knn", 1),
        ("spatial_knn", 4),
        ("random", 8),
    ]

    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"  GENERATING DATASET FOR SEED {seed}")
        print(f"{'#'*80}")

        all_samples: List[Dict[str, Any]] = []
        all_stats: List[Dict[str, Any]] = []

        if args.protocol_splits:
            print("  MODE: PROTOCOL SPLITS (Train: tum_fr1_desk frames 10,20,30,40 | Val: frames 45,55 | Test: tum_fr2_xyz frames 10,20,30)")
            fr1_frames = [10, 20, 30, 40, 45, 55]
            fr2_frames = [10, 20, 30]

            s_fr1, stats_fr1 = generate_samples_for_scene(
                scene_name="tum_fr1_desk",
                frames_to_sample=fr1_frames,
                max_candidates=args.max_candidates,
                seed=seed,
                device=device,
                protocol=protocol,
                context_specs=context_specs,
            )
            all_samples.extend(s_fr1)
            all_stats.extend(stats_fr1)

            s_fr2, stats_fr2 = generate_samples_for_scene(
                scene_name="tum_fr2_xyz",
                frames_to_sample=fr2_frames,
                max_candidates=args.max_candidates,
                seed=seed,
                device=device,
                protocol=protocol,
                context_specs=context_specs,
            )
            all_samples.extend(s_fr2)
            all_stats.extend(stats_fr2)

        elif args.tiny:
            print("  MODE: TINY PROTOTYPE (tum_fr2_xyz frames 10, 20, 10 candidates)")
            s_tiny, stats_tiny = generate_samples_for_scene(
                scene_name="tum_fr2_xyz",
                frames_to_sample=[10, 20],
                max_candidates=10,
                seed=seed,
                device=device,
                protocol=protocol,
                context_specs=context_specs,
            )
            all_samples.extend(s_tiny)
            all_stats.extend(stats_tiny)

        else:
            frames_to_sample = [int(f) for f in args.frames.split(",")]
            print(f"  MODE: CUSTOM ({args.scene} frames {frames_to_sample}, {args.max_candidates} candidates)")
            s_custom, stats_custom = generate_samples_for_scene(
                scene_name=args.scene,
                frames_to_sample=frames_to_sample,
                max_candidates=args.max_candidates,
                seed=seed,
                device=device,
                protocol=protocol,
                context_specs=context_specs,
            )
            all_samples.extend(s_custom)
            all_stats.extend(stats_custom)

        # Save dataset
        fname = args.output_filename or f"conditional_oracle_seed_{seed}.json"
        dataset_path = os.path.join(output_dir, fname)

        if args.append and os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                existing_samples = json.load(f)
            print(f"\n[Append] Found {len(existing_samples)} existing samples in {dataset_path}")
            combined_samples = existing_samples + all_samples
            all_samples = combined_samples

        print(f"\n[Save] Writing {len(all_samples)} samples to {dataset_path}")
        with open(dataset_path, 'w') as f:
            json.dump(all_samples, f, indent=2, default=str)

        # Verification
        print(f"\n{'='*60}")
        print(f"  VERIFICATION (Seed {seed})")
        print(f"{'='*60}")
        verification = verify_dataset(all_samples, all_stats, seed)

        verification_path = os.path.join(output_dir, f"prototype_verification_seed_{seed}.json" if len(seeds) > 1 else "prototype_verification.json")
        with open(verification_path, 'w') as f:
            json.dump(verification, f, indent=2)

        summary = {
            "phase": "Phase 6 — Context-Centric Conditional Oracle Dataset",
            "seed": seed,
            "total_samples": len(all_samples),
            "feature_dim": PHASE6_FEATURE_DIM,
            "feature_names": PHASE6_FEATURE_NAMES,
            "frame_stats": all_stats,
            "verification": verification,
            "context_specs": context_specs,
        }
        summary_path = os.path.join(output_dir, f"dataset_summary_seed_{seed}.json" if len(seeds) > 1 else "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\n[Done] Successfully built dataset for seed(s): {seeds}")


if __name__ == "__main__":
    main()

