#!/usr/bin/env python3
"""Phase 9B: Robust Normalization Experiment Runner.

Evaluates whether robust static normalization (B1: median/MAD) or online
test-time adaptive normalization (B2: EMA) recovers the in-domain utility quality
lost under scale-invariant geometric representation (A1) while preserving
cross-scene zero-shot transfer robustness on tum_fr2_xyz.

Variants:
    B0: "A1_standard"        (A1 + train mean/std z-score baseline)
    B1: "A1_robust_static"   (A1 + train median/MAD outlier-resistant normalizer)
    B2: "A1_online_adaptive" (A1 + online test-time EMA covariate adaptation)

Controlled Invariants:
    - Feature representation phi_A1 strictly frozen (geometry_relative).
    - Architecture: TwoHeadMLP(11 -> 64).
    - Loss: TwoHeadUtilityLoss(lambda_rank=1.0, lambda_q=0.25, lambda_t=0.125).
    - Schedule: Adam(lr=0.005, epochs=200).
    - Re-uses cached candidate sets from Phase 9A.
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

from research.phase9b_protocol import (
    SEEDS,
    TRAIN_SCENE,
    VAL_SCENE,
    TEST_SCENE,
    BUDGETS,
    TOP_K_FRACTIONS,
    NORMALIZATION_VARIANTS,
    VARIANT_ALIASES_9B,
    BASE_REPRESENTATION,
    B2_EMA_BETA,
    EPS,
    MODEL_IN_FEATURES,
    MODEL_HIDDEN_DIM,
    MODEL_EPS_COST,
    TRAIN_EPOCHS,
    TRAIN_LR,
    LOSS_LAMBDA_RANK,
    LOSS_LAMBDA_Q,
    LOSS_LAMBDA_T,
    OUTPUT_DIR_9B,
    OUTPUT_FILES_9B,
    SEED_RESULT_PATTERN_9B,
    get_repo_root,
    get_output_dir_9b,
    validate_protocol_integrity_9b,
    to_dict as protocol_to_dict,
)
from research.phase9_protocol import CANONICAL_FEATURE_SCHEMA
from research.phase9_features import transform_grouped_features
from research.phase9b_normalization import (
    StandardNormalizer,
    RobustMADNormalizer,
    OnlineEMANormalizer,
    create_normalizer,
)
from research.utility_dataset import (
    load_canonical_oracle_dataset,
    UtilityDataset,
)
from research.utility_models import TwoHeadMLP
from research.utility_losses import LossConfig
from research.utility_training import UtilityModelTrainer, TrainingConfig
from research.utility_metrics import (
    evaluate_rq1_prediction,
    evaluate_rq2_selection,
    safe_spearmanr,
)


def get_candidates_cache_dir() -> Path:
    """Return path to Phase 9A candidate cache directory."""
    repo = get_repo_root()
    p = repo / "results" / "phase9a_scale_invariant" / "eval_candidates_cache"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate cache directory {p}. Run Phase 9A first.")
    return p


def load_cached_candidates(domain: str, seed: int) -> List[Dict[str, Any]]:
    """Load cached candidate dictionaries from Phase 9A cache."""
    cache_dir = get_candidates_cache_dir()
    cache_file = cache_dir / f"{domain}_seed_{seed}.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached candidate file {cache_file}")
    with open(cache_file) as f:
        return json.load(f)


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


def train_phase9b_model(
    variant: str,
    seed: int,
    device: str = 'cuda',
) -> Tuple[TwoHeadMLP, Any]:
    """Train TwoHeadMLP on train split with specified Phase 9B normalizer."""
    variant_norm = VARIANT_ALIASES_9B.get(variant, variant)
    raw_dataset = load_canonical_oracle_dataset()

    train_raw = raw_dataset.get_split("train")
    val_raw = raw_dataset.get_split("validation")

    train_keys = [(m.scene, m.frame) for m in train_raw.metadata]
    val_keys = [(m.scene, m.frame) for m in val_raw.metadata]

    # 1. Fixed representation: A1 geometry_relative
    Z_train = transform_grouped_features(train_raw.X_np, train_keys, variant=BASE_REPRESENTATION)
    Z_val = transform_grouped_features(val_raw.X_np, val_keys, variant=BASE_REPRESENTATION)

    # 2. Fit normalizer strictly on train split
    normalizer = create_normalizer(variant_norm, eps=EPS, beta=B2_EMA_BETA)
    normalizer.fit(Z_train)

    if variant_norm == "A1_online_adaptive":
        # At training time, B2 normalizer uses train reference statistics mu_0, sigma_0
        Z_train_norm = normalizer.transform_static(Z_train)
        Z_val_norm = normalizer.transform_static(Z_val)
    else:
        Z_train_norm = normalizer.transform(Z_train)
        Z_val_norm = normalizer.transform(Z_val)

    train_ds = UtilityDataset(
        X=Z_train_norm,
        delta_q=train_raw.delta_q_tensor,
        delta_t=train_raw.delta_t_tensor,
        utility=train_raw.utility_tensor,
        metadata=train_raw.metadata,
        feature_names=CANONICAL_FEATURE_SCHEMA,
    )
    val_ds = UtilityDataset(
        X=Z_val_norm,
        delta_q=val_raw.delta_q_tensor,
        delta_t=val_raw.delta_t_tensor,
        utility=val_raw.utility_tensor,
        metadata=val_raw.metadata,
        feature_names=CANONICAL_FEATURE_SCHEMA,
    )

    # 3. Train TwoHeadMLP with identical protocol
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

    # Deterministic seeding before model construction
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
    out_dir = get_output_dir_9b()
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
            "phase": "9B",
            "variant": variant_norm,
            "seed": seed,
            "model_type": "TwoHeadMLP",
            "base_representation": BASE_REPRESENTATION,
        },
    )

    return model, normalizer


def evaluate_temporal_stability(
    seed: int,
    trained_models: Optional[Dict[str, Any]] = None,
    device: str = 'cuda',
) -> List[Dict[str, Any]]:
    """Run sequential frame normalization on evaluation scenes to track stability and latency.

    Measures:
        1. D_t^{norm} = ||mu_t - mu_{t-1}||_2
        2. D_t^{sigma} = ||sigma_t - sigma_{t-1}||_2
        3. Overlap@20(t, t-1) ranking stability
        4. T_{norm} / N_{candidate} (microseconds per candidate)
    """
    raw_dataset = load_canonical_oracle_dataset()
    train_raw = raw_dataset.get_split("train")
    train_keys = [(m.scene, m.frame) for m in train_raw.metadata]
    Z_train = transform_grouped_features(train_raw.X_np, train_keys, variant=BASE_REPRESENTATION)

    val_raw = raw_dataset.get_split("validation")
    val_keys = [(m.scene, m.frame) for m in val_raw.metadata]
    Z_val = transform_grouped_features(val_raw.X_np, val_keys, variant=BASE_REPRESENTATION)

    test_raw = raw_dataset.get_split("test")
    test_keys = [(m.scene, m.frame) for m in test_raw.metadata]
    Z_test = transform_grouped_features(test_raw.X_np, test_keys, variant=BASE_REPRESENTATION)

    eval_splits = [
        ("tum_fr1_desk_val", val_raw, Z_val),
        ("tum_fr2_xyz", test_raw, Z_test),
    ]

    stability_rows = []
    for var in NORMALIZATION_VARIANTS:
        model = trained_models.get(var) if trained_models else None

        for dom_label, split_raw, Z_split in eval_splits:
            normalizer = create_normalizer(var, eps=EPS, beta=B2_EMA_BETA)
            normalizer.fit(Z_train)
            if hasattr(normalizer, "reset"):
                normalizer.reset()

            unique_frames = sorted(list(set(m.frame for m in split_raw.metadata)))
            prev_top_k = None

            for step, fr in enumerate(unique_frames):
                indices = [i for i, m in enumerate(split_raw.metadata) if m.frame == fr]
                if not indices:
                    continue
                X_frame = Z_split[indices]

                t0 = time.perf_counter()
                if var == "A1_online_adaptive":
                    Z_norm, step_m = normalizer.update_and_transform_frame(X_frame, frame_id=fr)
                    d_norm = step_m["d_norm"]
                    mu_shift = step_m["mu_l2_shift"]
                    sigma_shift = step_m["sigma_l2_shift"]
                    mean_mu = step_m["mean_mu"]
                    mean_sigma = step_m["mean_sigma"]
                else:
                    Z_norm = normalizer.transform(X_frame)
                    d_norm = 0.0
                    mu_shift = 0.0
                    sigma_shift = 0.0
                    mean_mu = float(np.mean(getattr(normalizer, "mean", getattr(normalizer, "median", np.zeros(11)))))
                    mean_sigma = float(np.mean(getattr(normalizer, "std", getattr(normalizer, "scale", np.ones(11)))))

                lat_ms = (time.perf_counter() - t0) * 1000.0
                t_norm_per_cand_us = (lat_ms / max(len(indices), 1)) * 1000.0

                overlap_20 = 1.0
                if model is not None:
                    with torch.no_grad():
                        Z_t = torch.from_numpy(Z_norm).float().to(device)
                        _, _, pred_u_t = model(Z_t)
                        pred_u = pred_u_t.cpu().numpy()
                    k = max(1, int(0.20 * len(pred_u)))
                    top_k_indices = set(np.argsort(-pred_u)[:k])
                    if prev_top_k is not None and len(top_k_indices) > 0 and len(prev_top_k) > 0:
                        overlap_20 = len(top_k_indices & prev_top_k) / float(len(top_k_indices))
                    else:
                        overlap_20 = 1.0
                    prev_top_k = top_k_indices

                stability_rows.append({
                    "seed": seed,
                    "variant": var,
                    "domain": dom_label,
                    "step": step,
                    "frame": fr,
                    "n_samples": len(indices),
                    "latency_ms": lat_ms,
                    "t_norm_per_candidate_us": t_norm_per_cand_us,
                    "d_norm": d_norm,
                    "mu_l2_shift": mu_shift,
                    "sigma_l2_shift": sigma_shift,
                    "overlap_20": overlap_20,
                    "mean_mu": mean_mu,
                    "mean_sigma": mean_sigma,
                })

    return stability_rows


def run_phase9b(
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Run full Phase 9B experiment across variants and seeds."""
    if seeds is None:
        seeds = list(SEEDS)
    if variants is None:
        variants = list(NORMALIZATION_VARIANTS)

    validate_protocol_integrity_9b()
    out_dir = get_output_dir_9b()

    print("=" * 80)
    print("PHASE 9B: ROBUST NORMALIZATION")
    print(f"  Seeds:               {seeds}")
    print(f"  Variants:            {variants}")
    print(f"  Base Representation: {BASE_REPRESENTATION} (A1 fixed)")
    print(f"  Device:              {device}")
    print("=" * 80)

    # Pre-load evaluation candidate pools from Phase 9A cache
    print("\n>> Loading evaluation candidate pools from Phase 9A cache...")
    domain_candidates: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for seed in seeds:
        for dom in ["in_domain", "zero_shot"]:
            cands = load_cached_candidates(dom, seed=seed)
            domain_candidates[(dom, seed)] = cands
            print(f"   [Seed {seed}] {dom}: {len(cands)} candidates ready.")

    prediction_rows = []
    all_stability_rows = []
    runtime_rows = []
    trained_models: Dict[Tuple[str, int], Any] = {}

    all_results: Dict[str, Any] = {
        "phase": "Phase 9B: Robust Normalization",
        "generated_at": datetime.datetime.now().isoformat(),
        "base_representation": BASE_REPRESENTATION,
        "seeds": seeds,
        "variants": variants,
        "device": device,
        "per_variant": {},
    }

    for variant in variants:
        variant_norm = VARIANT_ALIASES_9B.get(variant, variant)
        print(f"\n{'#' * 80}")
        print(f"   VARIANT: {variant_norm.upper()}")
        print(f"{'#' * 80}")

        variant_seed_results = {}

        for seed in seeds:
            print(f"\n--- Training & Evaluating: {variant_norm} | Seed {seed} ---")
            t0 = time.time()
            model, normalizer = train_phase9b_model(variant_norm, seed=seed, device=device)
            trained_models[(variant_norm, seed)] = model
            t_train = time.time() - t0
            print(f"    Trained {variant_norm} seed {seed} in {t_train:.2f}s")

            seed_result = {"seed": seed, "variant": variant_norm}

            for dom, dom_label in [("in_domain", "fr1_desk_val"), ("zero_shot", "fr2_xyz")]:
                cands = domain_candidates[(dom, seed)]
                X, oracle_u, delta_q, costs = extract_features_from_candidates(cands)
                baselines = compute_baseline_scores(X, cands, seed)

                # 1. Transform raw features with fixed A1 representation
                cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
                Z = transform_grouped_features(X, cand_keys, variant=BASE_REPRESENTATION)

                # 2. Normalize features with variant normalizer
                t_norm_start = time.perf_counter()
                if variant_norm == "A1_online_adaptive":
                    # Unsupervised test-time online adaptation frame-by-frame
                    normalizer.reset()
                    unique_frames = sorted(list(set(c.get('frame', 0) for c in cands)))
                    Z_norm = np.zeros_like(Z)
                    total_d_norm = 0.0
                    total_mu_shift = 0.0
                    for fr in unique_frames:
                        idx = [i for i, c in enumerate(cands) if c.get('frame', 0) == fr]
                        fr_norm, step_m = normalizer.update_and_transform_frame(Z[idx], frame_id=fr)
                        Z_norm[idx] = fr_norm
                        total_d_norm += step_m["d_norm"]
                        total_mu_shift += step_m["mu_l2_shift"]
                    d_norm = total_d_norm / max(len(unique_frames), 1)
                    mu_shift = total_mu_shift / max(len(unique_frames), 1)
                else:
                    Z_norm = normalizer.transform(Z)
                    d_norm = 0.0
                    mu_shift = 0.0
                t_norm_ms = (time.perf_counter() - t_norm_start) * 1000.0

                # 3. Model inference
                t_infer_start = time.perf_counter()
                with torch.no_grad():
                    Z_tensor = torch.from_numpy(Z_norm).float().to(device)
                    _, _, pred_u_t = model(Z_tensor)
                    learned_u = pred_u_t.cpu().numpy()
                t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

                runtime_rows.append({
                    "seed": seed,
                    "variant": variant_norm,
                    "domain": dom_label,
                    "n_candidates": len(cands),
                    "t_norm_ms": t_norm_ms,
                    "t_norm_per_candidate_us": (t_norm_ms / max(len(cands), 1)) * 1000.0,
                    "t_infer_ms": t_infer_ms,
                    "t_infer_per_candidate_us": (t_infer_ms / max(len(cands), 1)) * 1000.0,
                    "d_norm": d_norm,
                    "mu_shift": mu_shift,
                })

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

    # Evaluate temporal stability across sequence of frames
    print("\n>> Evaluating temporal stability and adaptation dynamics...")
    for seed in seeds:
        seed_models = {var: trained_models.get((var, seed)) for var in variants}
        stab_rows = evaluate_temporal_stability(seed=seed, trained_models=seed_models, device=device)
        all_stability_rows.extend(stab_rows)

    # Save per-seed combined result JSONs
    for seed in seeds:
        seed_data = {"seed": seed}
        for variant_norm in all_results["per_variant"]:
            seed_data[variant_norm] = all_results["per_variant"][variant_norm][str(seed)]
        seed_path = out_dir / SEED_RESULT_PATTERN_9B.format(seed=seed)
        with open(seed_path, "w") as f:
            json.dump(seed_data, f, indent=2, default=str)

    # Save prediction_metrics.csv
    import pandas as pd
    pred_csv_path = out_dir / OUTPUT_FILES_9B["prediction_metrics"]
    df_pred = pd.DataFrame(prediction_rows)
    df_pred.to_csv(pred_csv_path, index=False)
    print(f"\nSaved prediction metrics to {pred_csv_path}")

    # Save normalization_stability.csv
    stab_csv_path = out_dir / OUTPUT_FILES_9B["normalization_stability"]
    df_stab = pd.DataFrame(all_stability_rows)
    df_stab.to_csv(stab_csv_path, index=False)
    print(f"Saved normalization stability to {stab_csv_path}")

    # Save runtime_overhead.csv
    run_csv_path = out_dir / OUTPUT_FILES_9B["runtime_overhead"]
    df_run = pd.DataFrame(runtime_rows)
    df_run.to_csv(run_csv_path, index=False)
    print(f"Saved runtime overhead to {run_csv_path}")

    # Save raw stage results
    raw_results_path = out_dir / OUTPUT_FILES_9B["stage_results"]
    with open(raw_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved raw stage results to {raw_results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 9B Robust Normalization Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 1-seed smoke test (seed 42, B0, B1, B2)")
    parser.add_argument("--variants", nargs="+", default=None, help="Variants to run (e.g. A1_standard A1_robust_static A1_online_adaptive)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        print(">>> RUNNING 1-SEED SMOKE TEST (Phase 9B) <<<")
        seeds = [42]
        variants = list(NORMALIZATION_VARIANTS)
    else:
        seeds = args.seeds or list(SEEDS)
        variants = args.variants or list(NORMALIZATION_VARIANTS)

    run_phase9b(seeds=seeds, variants=variants, device=device)


if __name__ == "__main__":
    main()
