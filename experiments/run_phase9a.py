#!/usr/bin/env python3
"""Phase 9A: Scale-Invariant Feature Representation Experiment Runner.

Evaluates whether scale-invariant geometric representation (A1) and
optimization-relative representation (A2) reduce generalization degradation
from tum_fr1_desk to unseen tum_fr2_xyz compared to raw baseline (A0).

Variants:
    A0: raw canonical features (11 dims) + train-only normalizer
    A1: geometry_relative (scale-relative depth_error, position_drift, projected_area)
    A2: geometry_optimization_relative (A1 + relative gradient, influence, uncertainty, residual)

Execution Flow:
    1. For each representation variant (A0, A1, A2):
       a. For each seed (42, 43, 44, 45, 46):
          i. Prepare train/val splits with variant transform + train-only normalizer.
          ii. Train TwoHeadMLP(in_features=11, hidden_dim=64) for 200 epochs.
          iii. Obtain evaluation candidates for in-domain (fr1_desk) and zero-shot (fr2_xyz).
          iv. Transform evaluation candidate features with variant transform + normalizer.
          v. Predict utility and evaluate RQ1 (prediction) & RQ2 (selection at equal budgets).
    2. Save per-seed JSONs, prediction_metrics.csv, and stage results.
"""
import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase9_protocol import (
    SEEDS, TRAIN_SCENE, TEST_SCENE, VAL_SCENE,
    BUDGETS, TOP_K_FRACTIONS, CANONICAL_FEATURE_SCHEMA,
    FEATURE_VARIANTS, VARIANT_ALIASES, BASELINES,
    MODEL_IN_FEATURES, MODEL_HIDDEN_DIM, MODEL_EPS_COST,
    TRAIN_EPOCHS, TRAIN_LR, LOSS_LAMBDA_RANK, LOSS_LAMBDA_Q, LOSS_LAMBDA_T,
    GATE_CRITERIA, OUTPUT_DIR, OUTPUT_FILES, SEED_RESULT_PATTERN,
    get_repo_root, get_output_dir, validate_protocol_integrity,
    to_dict as protocol_to_dict,
)
from research.phase9_features import (
    transform_features_array,
    transform_grouped_features,
)
from research.protocol import (
    load_protocol, get_resolution, get_dataset_config,
)
from research.utility_dataset import (
    load_canonical_oracle_dataset,
    UtilityDataset,
    FeatureNormalizer,
)
from research.utility_models import TwoHeadMLP
from research.utility_losses import LossConfig
from research.utility_training import UtilityModelTrainer, TrainingConfig
from research.utility_metrics import (
    evaluate_rq1_prediction,
    evaluate_rq2_selection,
    safe_spearmanr,
)
from datasets.tum_dataset import TUMDataset
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
from research.pipeline import OnlineReconstructionPipeline


# ═══════════════════════════════════════════════════════════════════════
# Pipeline & Candidate Cache Helpers
# ═══════════════════════════════════════════════════════════════════════

def get_cache_dir() -> Path:
    cache_path = get_output_dir() / "eval_candidates_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_tum_frames(
    data_path: str,
    camera: str,
    n_frames: int,
    start_frame: int = 0,
    H: int = 240,
    W: int = 320,
    device: str = 'cuda',
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    """Load and resize TUM-RGBD frames with proper intrinsics scaling."""
    dataset = TUMDataset(data_path, max_frames=start_frame + n_frames + 5, camera=camera)
    frames = []
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H

    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0.0, dataset.cx * scale_x],
        [0.0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    for i in range(start_frame, min(start_frame + n_frames, len(dataset))):
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
            'frame_id': i,
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device),
        })

    return frames, intrinsics


def build_pipeline_and_collect_oracle(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    seed: int,
    device: str = 'cuda',
    scene_name: str = 'tum_fr2_xyz',
) -> List[Dict[str, Any]]:
    """Build pipeline and collect oracle utility candidates."""
    pipeline_cfg = {
        "gaussian": {
            "sh_degree": 0,
            "initial_opacity": 0.5,
            "max_gaussians": 30000,
            "initial_scale": 0.02,
        },
        "rendering": {
            "tile_size": 16,
            "image_width": int(intrinsics[0, 2].item() * 2),
            "image_height": int(intrinsics[1, 2].item() * 2),
            "use_surface_aware_depth": True,
            "attribution_top_k": 4,
        },
        "scheduler": {
            "gpu_budget_ms": 25.0,
            "policy": "budget_aware",
        },
        "densification": {
            "max_new_per_frame": 80,
            "strategy": "importance",
            "use_adaptive_thresholds": True,
        },
    }

    pipeline = OnlineReconstructionPipeline(config=pipeline_cfg, device=device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )

    for t in range(1, len(frames) - 1):
        pipeline.process_frame(
            rgb=frames[t]['rgb'],
            depth=frames[t]['depth'],
            gt_pose=frames[t]['pose'],
        )

    eval_frame = frames[-1]
    oracle = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=40,
        n_opt_steps=5,
        w_rgb=0.70,
        w_depth=0.30,
        seed=seed,
        min_influence_pixels=25,
    )

    results = oracle.run_oracle_experiment(
        rgb=eval_frame['rgb'],
        depth=eval_frame['depth'],
        population_type=SamplingPopulation.GEOMETRY_STRATIFIED,
        scene_name=scene_name,
        frame_idx=eval_frame.get('frame_id', len(frames) - 1),
        split='cross_scene_test',
        seed=seed,
    )

    visible = [r for r in results if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]
    return visible


def get_cached_candidates(
    domain: str,
    seed: int,
    device: str = 'cuda',
) -> List[Dict[str, Any]]:
    """Retrieve evaluation candidates from cache or collect live if missing."""
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{domain}_seed_{seed}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    protocol = load_protocol()
    if domain == "in_domain":
        cfg = get_dataset_config('tum_fr1_desk', protocol)
        H, W = get_resolution('tum_fr1_desk', protocol)
        frames, intrinsics = load_tum_frames(
            data_path=cfg['full_path'],
            camera=cfg.get('camera', 'freiburg1'),
            n_frames=60,
            start_frame=0,
            H=H, W=W,
            device=device,
        )
        val_frames = [f for f in frames if f['frame_id'] >= 41]
        candidates = build_pipeline_and_collect_oracle(
            frames=val_frames,
            intrinsics=intrinsics,
            seed=seed,
            device=device,
            scene_name='tum_fr1_desk',
        )
    else:
        cfg = get_dataset_config('tum_fr2_xyz', protocol)
        H, W = get_resolution('tum_fr2_xyz', protocol)
        frames, intrinsics = load_tum_frames(
            data_path=cfg['full_path'],
            camera=cfg.get('camera', 'freiburg2'),
            n_frames=20,
            start_frame=0,
            H=H, W=W,
            device=device,
        )
        candidates = build_pipeline_and_collect_oracle(
            frames=frames,
            intrinsics=intrinsics,
            seed=seed,
            device=device,
            scene_name='tum_fr2_xyz',
        )

    # Save to cache
    # Candidate dicts may contain torch tensors or non-serializable items
    serializable = []
    for c in candidates:
        s_cand = {
            'gaussian_id': int(c.get('gaussian_id', 0)),
            'scene': str(c.get('scene', '')),
            'frame': int(c.get('frame', 0)),
            'features': {k: float(v) for k, v in c.get('features', {}).items()},
            'oracle_utility_joint': float(c.get('oracle_utility_joint', 0.0)),
            'delta_quality_local': float(c.get('delta_quality_local', 0.0)),
            'measured_trial_cost_ms': float(c.get('measured_trial_cost_ms', 1.0)),
            'predicted_utility': float(c.get('predicted_utility', 0.0)),
            'visible': bool(c.get('visible', True)),
            'n_influence_pixels': int(c.get('n_influence_pixels', 0)),
        }
        serializable.append(s_cand)

    with open(cache_file, "w") as f:
        json.dump(serializable, f, indent=2)

    return serializable


def extract_features_from_candidates(
    candidates: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw [N, 11] feature matrix, oracle U*, delta_q, costs."""
    X, oracle_u, delta_q, costs = [], [], [], []
    for cand in candidates:
        f = cand.get('features', {})
        vec = [float(f.get(name, 0.0)) for name in CANONICAL_FEATURE_SCHEMA]
        X.append(vec)
        oracle_u.append(float(cand.get('oracle_utility_joint', 0.0)))
        delta_q.append(float(cand.get('delta_quality_local', 0.0)))
        costs.append(float(cand.get('measured_trial_cost_ms', 1.0)))
    return (
        np.array(X, dtype=np.float32),
        np.array(oracle_u, dtype=np.float32),
        np.array(delta_q, dtype=np.float32),
        np.array(costs, dtype=np.float32),
    )


def compute_baseline_scores(
    X: np.ndarray,
    candidates: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, np.ndarray]:
    """Compute baseline scoring functions."""
    n = len(X)
    rng = np.random.default_rng(seed)
    random_scores = rng.random(n).astype(np.float32)
    error_scores = X[:, 0] + X[:, 1]  # rgb_error + depth_error
    heuristic_scores = np.array([
        float(c.get('predicted_utility', 0.0)) for c in candidates
    ], dtype=np.float32)
    if np.std(heuristic_scores) < 1e-7:
        heuristic_scores = error_scores.copy()
    return {
        'random': random_scores,
        'error_only': error_scores,
        'heuristic': heuristic_scores,
    }


# ═══════════════════════════════════════════════════════════════════════
# Model Training for a Variant and Seed
# ═══════════════════════════════════════════════════════════════════════

def train_phase9_model(
    variant: str,
    seed: int,
    device: str = 'cuda',
) -> Tuple[TwoHeadMLP, FeatureNormalizer]:
    """Train TwoHeadMLP on train split with specified representation variant."""
    variant_norm = VARIANT_ALIASES.get(variant, variant)
    raw_dataset = load_canonical_oracle_dataset()

    train_raw = raw_dataset.get_split("train")
    val_raw = raw_dataset.get_split("validation")

    train_keys = [(m.scene, m.frame) for m in train_raw.metadata]
    val_keys = [(m.scene, m.frame) for m in val_raw.metadata]

    # 1. Transform features with variant
    Z_train = transform_grouped_features(train_raw.X_np, train_keys, variant=variant_norm)
    Z_val = transform_grouped_features(val_raw.X_np, val_keys, variant=variant_norm)

    # 2. Fit normalizer strictly on train split
    normalizer = FeatureNormalizer()
    normalizer.fit(Z_train)

    Z_train_norm = normalizer.transform(Z_train)
    Z_val_norm = normalizer.transform(Z_val)

    train_ds = UtilityDataset(
        X=Z_train_norm,
        delta_q=train_raw.delta_q_tensor,
        delta_t=train_raw.delta_t_tensor,
        utility=train_raw.utility_tensor,
        metadata=train_raw.metadata,
        feature_names=CANONICAL_FEATURE_SCHEMA,
        normalizer=normalizer,
    )
    val_ds = UtilityDataset(
        X=Z_val_norm,
        delta_q=val_raw.delta_q_tensor,
        delta_t=val_raw.delta_t_tensor,
        utility=val_raw.utility_tensor,
        metadata=val_raw.metadata,
        feature_names=CANONICAL_FEATURE_SCHEMA,
        normalizer=normalizer,
    )

    # 3. Train TwoHeadMLP with identical protocol to Phase 4
    loss_cfg = LossConfig(
        lambda_rank=LOSS_LAMBDA_RANK,
        lambda_q=LOSS_LAMBDA_Q,
        lambda_t=LOSS_LAMBDA_T,
    )
    training_cfg = TrainingConfig(
        epochs=TRAIN_EPOCHS,
        learning_rate=TRAIN_LR,
        loss_config=loss_cfg,
        device=device,
    )
    trainer = UtilityModelTrainer(config=training_cfg)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TwoHeadMLP(
        in_features=MODEL_IN_FEATURES,
        hidden_dim=MODEL_HIDDEN_DIM,
        eps_cost=MODEL_EPS_COST,
    ).to(device)

    trainer.train_two_head_model(model, train_ds, val_ds=val_ds, seed=seed)
    model.eval()
    model.to(device)

    # 4. Save checkpoint and normalizer
    out_dir = get_output_dir()
    ckpt_dir = out_dir / "checkpoints"
    norm_dir = out_dir / "normalizers"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{variant_norm}_seed_{seed}.pt"
    norm_path = norm_dir / f"{variant_norm}_seed_{seed}.json"

    normalizer.save_json(str(norm_path))
    trainer.save_checkpoint(
        model,
        str(ckpt_path),
        feature_names=CANONICAL_FEATURE_SCHEMA,
        metadata={
            "phase": "9A",
            "variant": variant_norm,
            "seed": seed,
            "model_type": "TwoHeadMLP",
        },
    )

    return model, normalizer


# ═══════════════════════════════════════════════════════════════════════
# Full Evaluation Engine
# ═══════════════════════════════════════════════════════════════════════

def run_phase9a(
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Run full Phase 9A experiment across variants and seeds."""
    if seeds is None:
        seeds = list(SEEDS)
    if variants is None:
        variants = list(FEATURE_VARIANTS)

    validate_protocol_integrity()
    out_dir = get_output_dir()

    print("=" * 80)
    print("PHASE 9A: SCALE-INVARIANT FEATURE REPRESENTATIONS")
    print(f"  Seeds:    {seeds}")
    print(f"  Variants: {variants}")
    print(f"  Device:   {device}")
    print("=" * 80)

    # Pre-cache or load evaluation candidates for in-domain and zero-shot
    print("\n>> Preparing evaluation candidate pools...")
    domain_candidates: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for seed in seeds:
        for dom in ["in_domain", "zero_shot"]:
            print(f"   [Seed {seed}] Loading/caching {dom} candidates...")
            cands = get_cached_candidates(dom, seed=seed, device=device)
            domain_candidates[(dom, seed)] = cands
            print(f"   [Seed {seed}] {dom}: {len(cands)} candidates ready.")

    prediction_rows = []
    all_results: Dict[str, Any] = {
        "phase": "Phase 9A: Scale-Invariant Features",
        "generated_at": datetime.datetime.now().isoformat(),
        "seeds": seeds,
        "variants": variants,
        "device": device,
        "per_variant": {},
    }

    for variant in variants:
        variant_norm = VARIANT_ALIASES.get(variant, variant)
        print(f"\n{'#' * 80}")
        print(f"   VARIANT: {variant_norm.upper()}")
        print(f"{'#' * 80}")

        variant_seed_results = {}

        for seed in seeds:
            print(f"\n--- Training & Evaluating: {variant_norm} | Seed {seed} ---")
            t0 = time.time()
            model, normalizer = train_phase9_model(variant_norm, seed=seed, device=device)
            t_train = time.time() - t0
            print(f"    Trained {variant_norm} seed {seed} in {t_train:.2f}s")

            seed_result = {"seed": seed, "variant": variant_norm}

            for dom, dom_label in [("in_domain", "fr1_desk_val"), ("zero_shot", "fr2_xyz")]:
                cands = domain_candidates[(dom, seed)]
                X, oracle_u, delta_q, costs = extract_features_from_candidates(cands)
                baselines = compute_baseline_scores(X, cands, seed)

                # Transform raw candidates with variant representation (grouped by scene/frame)
                cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
                Z = transform_grouped_features(X, cand_keys, variant=variant_norm)
                # Standardize using train normalizer
                Z_norm = normalizer.transform(Z)

                # Model prediction
                with torch.no_grad():
                    Z_tensor = torch.from_numpy(Z_norm).float().to(device)
                    _, _, pred_u_t = model(Z_tensor)
                    learned_u = pred_u_t.cpu().numpy()

                dom_metrics = {}
                methods_to_eval = {
                    "learned": learned_u,
                    "error_only": baselines["error_only"],
                    "heuristic": baselines["heuristic"],
                    "random": baselines["random"],
                    "oracle": oracle_u,
                }

                for m_name, scores in methods_to_eval.items():
                    rq1 = evaluate_rq1_prediction(pred_u=scores, oracle_u=oracle_u)
                    rq2 = evaluate_rq2_selection(
                        pred_u=scores, oracle_u=oracle_u,
                        delta_q=delta_q, costs=costs,
                    )
                    combined = {**rq1, **rq2}
                    dom_metrics[m_name] = combined

                    prediction_rows.append({
                        "seed": seed,
                        "variant": variant_norm,
                        "domain": dom_label,
                        "method": m_name,
                        "n_candidates": len(cands),
                        "spearman_rho": combined["spearman_rho"],
                        "spearman_pval": combined["spearman_pval"],
                        "pearson_r": combined["pearson_r"],
                        "ndcg_10pct": combined.get("ndcg_10pct", float("nan")),
                        "ndcg_20pct": combined.get("ndcg_20pct", float("nan")),
                        "ose_20pct": combined.get("ose_20pct", float("nan")),
                        "regret_20pct": combined.get("regret_20pct", float("nan")),
                    })

                seed_result[dom] = dom_metrics
                m = dom_metrics["learned"]
                print(f"    [{dom_label}] Learned: rho={m['spearman_rho']:+.4f} | NDCG@20={m.get('ndcg_20pct', 0):.4f} | OSE@20={m.get('ose_20pct', 0):.3f}")

            variant_seed_results[str(seed)] = seed_result

        all_results["per_variant"][variant_norm] = variant_seed_results

    # Save per-seed combined result JSONs
    for seed in seeds:
        seed_data = {"seed": seed}
        for variant_norm in all_results["per_variant"]:
            seed_data[variant_norm] = all_results["per_variant"][variant_norm][str(seed)]
        seed_path = out_dir / SEED_RESULT_PATTERN.format(seed=seed)
        with open(seed_path, "w") as f:
            json.dump(seed_data, f, indent=2, default=str)

    # Save prediction_metrics.csv
    pred_csv_path = out_dir / OUTPUT_FILES["prediction_metrics"]
    import pandas as pd
    df_pred = pd.DataFrame(prediction_rows)
    df_pred.to_csv(pred_csv_path, index=False)
    print(f"\nSaved prediction metrics to {pred_csv_path}")

    # Save raw stage results
    raw_results_path = out_dir / "stage_results.json"
    with open(raw_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved raw stage results to {raw_results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 9A Scale-Invariant Evaluation Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 1-seed smoke test (seed 42, A0 & A1)")
    parser.add_argument("--variants", nargs="+", default=None, help="Variants to run (e.g. A0 A1 A2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        print(">>> RUNNING 1-SEED SMOKE TEST (Step 5) <<<")
        seeds = [42]
        variants = ["raw", "geometry_relative"]
    else:
        seeds = args.seeds or list(SEEDS)
        variants = args.variants or list(FEATURE_VARIANTS)

    run_phase9a(seeds=seeds, variants=variants, device=device)


if __name__ == "__main__":
    main()
