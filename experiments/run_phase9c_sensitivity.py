#!/usr/bin/env python3
"""Phase 9C: B2 Robustness and Adaptation Necessity Experiment Runner.

Executes four rigorous experimental pillars:
    1. 9C-1: Sensitivity analysis across EMA timescales beta in {0.80, 0.90, 0.95}.
    2. 9C-2: Adaptation ablation: B0 vs B2 vs B2-static (equivalence check & active adaptation).
    3. 9C-3: Temporal dynamics: frame-by-frame drift, scale drift, selection overlap, and rho_t.
    4. 9C-4: Controlled perturbation robustness under scale shifts (a in {0.8, 1.0, 1.2})
             and offset shifts (b in {-0.2, 0.0, 0.2}).

Protocol Invariants:
    - Representation: Fixed A1 scale-invariant geometry.
    - Architecture: Frozen TwoHeadMLP(11 -> 64) loaded from Phase 9B checkpoints.
    - Seeds: n=5 seeds [42, 43, 44, 45, 46].
    - Budgets: [0.10, 0.20, 0.40, 0.60, 0.80].
    - Zero oracle leakage, zero future frame access.
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
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase9_protocol import CANONICAL_FEATURE_SCHEMA
from research.phase9_features import transform_grouped_features
from research.phase9b_protocol import (
    BASE_REPRESENTATION,
    EPS,
    BUDGETS,
    SEEDS,
)
from research.phase9b_normalization import (
    StandardNormalizer,
    OnlineEMANormalizer,
)
from research.phase9c_protocol import (
    BETA_VALUES,
    ABLATION_VARIANTS,
    ABLATION_VARIANT_NAMES,
    PERTURBATION_SCALES,
    PERTURBATION_OFFSETS,
    OUTPUT_FILES_9C,
    get_repo_root,
    get_output_dir_9c,
    validate_protocol_integrity_9c,
    to_dict as protocol_to_dict,
)
from research.phase9c_analysis import (
    B2StaticNormalizer,
    apply_feature_perturbation,
    compute_per_frame_spearman,
    verify_b0_b2_static_equivalence,
)
from research.utility_dataset import load_canonical_oracle_dataset
from research.utility_models import TwoHeadMLP
from research.utility_metrics import (
    evaluate_rq1_prediction,
    evaluate_rq2_selection,
)


def get_candidates_cache_dir() -> Path:
    """Return path to Phase 9A candidate cache directory."""
    repo = get_repo_root()
    p = repo / "results" / "phase9a_scale_invariant" / "eval_candidates_cache"
    if not p.exists():
        raise FileNotFoundError(f"Missing candidate cache directory {p}")
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


def load_frozen_phase9b_checkpoint(
    seed: int,
    device: str = 'cpu',
) -> TwoHeadMLP:
    """Load frozen Phase 9B model checkpoint for a given seed."""
    repo = get_repo_root()
    ckpt_path = (
        repo
        / "results"
        / "phase9b_robust_normalization"
        / "checkpoints"
        / f"A1_online_adaptive_seed_{seed}.pt"
    )
    if not ckpt_path.exists():
        # Fallback to standard checkpoint if online adaptive pt missing
        ckpt_path = (
            repo
            / "results"
            / "phase9b_robust_normalization"
            / "checkpoints"
            / f"A1_standard_seed_{seed}.pt"
        )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing frozen Phase 9B checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = TwoHeadMLP(in_features=11, hidden_dim=64, eps_cost=1e-4)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    return model


def get_train_features() -> np.ndarray:
    """Load train split raw features transformed with fixed A1 representation."""
    raw_dataset = load_canonical_oracle_dataset()
    train_raw = raw_dataset.get_split("train")
    train_keys = [(m.scene, m.frame) for m in train_raw.metadata]
    return transform_grouped_features(train_raw.X_np, train_keys, variant=BASE_REPRESENTATION)


def run_phase9c(
    seeds: Optional[List[int]] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Execute complete Phase 9C experimental suite."""
    if seeds is None:
        seeds = list(SEEDS)

    validate_protocol_integrity_9c()
    out_dir = get_output_dir_9c()

    print("=" * 80)
    print("PHASE 9C: B2 ROBUSTNESS & ADAPTATION NECESSITY")
    print(f"  Seeds:               {seeds}")
    print(f"  Beta Grid:           {BETA_VALUES}")
    print(f"  Ablation Variants:   {ABLATION_VARIANTS}")
    print(f"  Scale Shifts:        {PERTURBATION_SCALES}")
    print(f"  Offset Shifts:       {PERTURBATION_OFFSETS}")
    print(f"  Base Representation: {BASE_REPRESENTATION} (A1 fixed)")
    print(f"  Device:              {device}")
    print("=" * 80)

    # 1. Load train reference features and evaluation candidates
    print("\n>> Loading train reference features and evaluation candidate pools...")
    Z_train = get_train_features()
    print(f"   Train reference features: {Z_train.shape}")

    domain_candidates: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for seed in seeds:
        for dom in ["in_domain", "zero_shot"]:
            cands = load_cached_candidates(dom, seed=seed)
            domain_candidates[(dom, seed)] = cands
            print(f"   [Seed {seed}] {dom}: {len(cands)} candidates loaded.")

    # Storage containers
    sensitivity_rows = []
    ablation_rows = []
    temporal_rows = []
    perturbation_rows = []
    selection_rows = []
    runtime_rows = []

    stage_results: Dict[str, Any] = {
        "phase": "Phase 9C: B2 Robustness & Adaptation Necessity",
        "generated_at": datetime.datetime.now().isoformat(),
        "base_representation": BASE_REPRESENTATION,
        "seeds": seeds,
        "sensitivity_beta_values": BETA_VALUES,
        "ablation_variants": ABLATION_VARIANTS,
        "perturbation_scales": PERTURBATION_SCALES,
        "perturbation_offsets": PERTURBATION_OFFSETS,
        "device": device,
        "sensitivity": {},
        "ablation": {},
        "temporal": {},
        "perturbation": {},
    }

    # Pre-load frozen models for all seeds
    print("\n>> Loading frozen Phase-9B TwoHeadMLP models...")
    models = {seed: load_frozen_phase9b_checkpoint(seed, device=device) for seed in seeds}

    # ═══════════════════════════════════════════════════════════════════════
    # Part 9C-1: Sensitivity Analysis (beta in {0.80, 0.90, 0.95})
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 80}")
    print("   PART 9C-1: EMA SENSITIVITY ANALYSIS (beta in {0.80, 0.90, 0.95})")
    print(f"{'#' * 80}")

    for beta in BETA_VALUES:
        beta_key = f"beta_{beta:.2f}"
        print(f"\n--- Evaluating EMA Beta = {beta:.2f} ---")
        stage_results["sensitivity"][beta_key] = {}

        for seed in seeds:
            model = models[seed]
            seed_res = {}

            for dom, dom_label in [("in_domain", "tum_fr1_desk_val"), ("zero_shot", "tum_fr2_xyz")]:
                cands = domain_candidates[(dom, seed)]
                X, oracle_u, delta_q, costs = extract_features_from_candidates(cands)
                cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
                Z = transform_grouped_features(X, cand_keys, variant=BASE_REPRESENTATION)

                normalizer = OnlineEMANormalizer(beta=beta, eps=EPS).fit(Z_train)
                unique_frames = sorted(list(set(c.get('frame', 0) for c in cands)))

                learned_u = np.zeros(len(cands), dtype=np.float32)
                t_ema_ms = 0.0
                t_infer_ms = 0.0
                total_d_norm = 0.0
                total_mu_shift = 0.0
                total_sigma_shift = 0.0

                for fr in unique_frames:
                    idx = [i for i, c in enumerate(cands) if c.get('frame', 0) == fr]
                    t0 = time.perf_counter()
                    fr_norm, step_m = normalizer.update_and_transform_frame(Z[idx], frame_id=fr)
                    t_ema_ms += (time.perf_counter() - t0) * 1000.0
                    total_d_norm += step_m["d_norm"]
                    total_mu_shift += step_m["mu_l2_shift"]
                    total_sigma_shift += step_m["sigma_l2_shift"]

                    t1 = time.perf_counter()
                    with torch.no_grad():
                        fr_tensor = torch.from_numpy(fr_norm).float().to(device)
                        _, _, pred_u_fr = model(fr_tensor)
                        learned_u[idx] = pred_u_fr.cpu().numpy()
                    t_infer_ms += (time.perf_counter() - t1) * 1000.0

                t_sel_start = time.perf_counter()
                rq1 = evaluate_rq1_prediction(pred_u=learned_u, oracle_u=oracle_u)
                rq2 = evaluate_rq2_selection(
                    pred_u=learned_u, oracle_u=oracle_u,
                    delta_q=delta_q, costs=costs,
                )
                t_sel_ms = (time.perf_counter() - t_sel_start) * 1000.0

                combined = {**rq1, **rq2}
                seed_res[dom] = combined

                n_frames = max(len(unique_frames), 1)
                mean_d_norm = total_d_norm / n_frames
                mean_mu_shift = total_mu_shift / n_frames
                mean_sigma_shift = total_sigma_shift / n_frames

                sensitivity_rows.append({
                    "beta": beta,
                    "seed": seed,
                    "domain": dom_label,
                    "n_candidates": len(cands),
                    "spearman_rho": combined["spearman_rho"],
                    "spearman_pval": combined["spearman_pval"],
                    "pearson_r": combined["pearson_r"],
                    "ndcg_20pct": combined.get("ndcg_20pct", float("nan")),
                    "ose_20pct": combined.get("ose_20pct", float("nan")),
                    "regret_20pct": combined.get("regret_20pct", float("nan")),
                    "dq_10pct": combined.get("realized_delta_q_10pct", float("nan")),
                    "dq_20pct": combined.get("realized_delta_q_20pct", float("nan")),
                    "dq_40pct": combined.get("realized_delta_q_40pct", float("nan")),
                    "dq_60pct": combined.get("realized_delta_q_60pct", float("nan")),
                    "dq_80pct": combined.get("realized_delta_q_80pct", float("nan")),
                    "d_norm": mean_d_norm,
                    "mu_shift": mean_mu_shift,
                    "sigma_shift": mean_sigma_shift,
                    "t_ema_ms": t_ema_ms,
                    "t_infer_ms": t_infer_ms,
                    "t_selection_ms": t_sel_ms,
                    "t_total_ms": t_ema_ms + t_infer_ms + t_sel_ms,
                })

                # Selection rows for budget curves
                for b_frac in BUDGETS:
                    b_key = f"{int(b_frac * 100)}pct"
                    selection_rows.append({
                        "experiment": "sensitivity",
                        "variant": f"B2_beta_{beta:.2f}",
                        "beta": beta,
                        "seed": seed,
                        "domain": dom_label,
                        "budget_fraction": b_frac,
                        "realized_delta_q": combined.get(f"realized_delta_q_{b_key}", float("nan")),
                        "oracle_delta_q": combined.get(f"oracle_delta_q_{b_key}", float("nan")),
                        "ose": combined.get(f"ose_{b_key}", float("nan")),
                        "ndcg": combined.get(f"ndcg_{b_key}", float("nan")),
                    })

                print(f"   [{dom_label}] Seed {seed} | Beta={beta:.2f}: rho={combined['spearman_rho']:+.4f} | NDCG@20={combined['ndcg_20pct']:.4f} | OSE@20={combined['ose_20pct']:.4f} | D_t^norm={mean_d_norm:.4f}")

            stage_results["sensitivity"][beta_key][str(seed)] = seed_res

    # ═══════════════════════════════════════════════════════════════════════
    # Part 9C-2: Adaptation Ablation (B0 vs B2 vs B2-static)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 80}")
    print("   PART 9C-2: ADAPTATION ABLATION (B0 vs B2 vs B2-static)")
    print(f"{'#' * 80}")

    for var_code in ABLATION_VARIANTS:
        var_name = ABLATION_VARIANT_NAMES[var_code]
        print(f"\n--- Evaluating Ablation Variant: {var_code} ({var_name}) ---")
        stage_results["ablation"][var_code] = {}

        for seed in seeds:
            model = models[seed]
            seed_res = {}

            for dom, dom_label in [("in_domain", "tum_fr1_desk_val"), ("zero_shot", "tum_fr2_xyz")]:
                cands = domain_candidates[(dom, seed)]
                X, oracle_u, delta_q, costs = extract_features_from_candidates(cands)
                cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
                Z = transform_grouped_features(X, cand_keys, variant=BASE_REPRESENTATION)

                if var_code == "B0":
                    normalizer = StandardNormalizer(eps=EPS).fit(Z_train)
                elif var_code == "B2":
                    normalizer = OnlineEMANormalizer(beta=0.90, eps=EPS).fit(Z_train)
                elif var_code == "B2_static":
                    normalizer = B2StaticNormalizer(eps=EPS).fit(Z_train)
                else:
                    raise ValueError(f"Unknown ablation variant: {var_code}")

                unique_frames = sorted(list(set(c.get('frame', 0) for c in cands)))
                learned_u = np.zeros(len(cands), dtype=np.float32)
                t_norm_ms = 0.0
                t_infer_ms = 0.0

                for fr in unique_frames:
                    idx = [i for i, c in enumerate(cands) if c.get('frame', 0) == fr]
                    t0 = time.perf_counter()
                    if hasattr(normalizer, "update_and_transform_frame"):
                        fr_norm, _ = normalizer.update_and_transform_frame(Z[idx], frame_id=fr)
                    else:
                        fr_norm = normalizer.transform(Z[idx])
                    t_norm_ms += (time.perf_counter() - t0) * 1000.0

                    t1 = time.perf_counter()
                    with torch.no_grad():
                        fr_tensor = torch.from_numpy(fr_norm).float().to(device)
                        _, _, pred_u_fr = model(fr_tensor)
                        learned_u[idx] = pred_u_fr.cpu().numpy()
                    t_infer_ms += (time.perf_counter() - t1) * 1000.0

                rq1 = evaluate_rq1_prediction(pred_u=learned_u, oracle_u=oracle_u)
                rq2 = evaluate_rq2_selection(
                    pred_u=learned_u, oracle_u=oracle_u,
                    delta_q=delta_q, costs=costs,
                )
                combined = {**rq1, **rq2}
                seed_res[dom] = combined

                ablation_rows.append({
                    "variant_code": var_code,
                    "variant_name": var_name,
                    "seed": seed,
                    "domain": dom_label,
                    "n_candidates": len(cands),
                    "spearman_rho": combined["spearman_rho"],
                    "spearman_pval": combined["spearman_pval"],
                    "pearson_r": combined["pearson_r"],
                    "ndcg_20pct": combined.get("ndcg_20pct", float("nan")),
                    "ose_20pct": combined.get("ose_20pct", float("nan")),
                    "regret_20pct": combined.get("regret_20pct", float("nan")),
                    "dq_10pct": combined.get("realized_delta_q_10pct", float("nan")),
                    "dq_20pct": combined.get("realized_delta_q_20pct", float("nan")),
                    "dq_40pct": combined.get("realized_delta_q_40pct", float("nan")),
                    "dq_60pct": combined.get("realized_delta_q_60pct", float("nan")),
                    "dq_80pct": combined.get("realized_delta_q_80pct", float("nan")),
                    "t_norm_ms": t_norm_ms,
                    "t_infer_ms": t_infer_ms,
                })

                # Selection rows for ablation budget curves
                for b_frac in BUDGETS:
                    b_key = f"{int(b_frac * 100)}pct"
                    selection_rows.append({
                        "experiment": "ablation",
                        "variant": var_code,
                        "beta": 0.90 if var_code == "B2" else float("nan"),
                        "seed": seed,
                        "domain": dom_label,
                        "budget_fraction": b_frac,
                        "realized_delta_q": combined.get(f"realized_delta_q_{b_key}", float("nan")),
                        "oracle_delta_q": combined.get(f"oracle_delta_q_{b_key}", float("nan")),
                        "ose": combined.get(f"ose_{b_key}", float("nan")),
                        "ndcg": combined.get(f"ndcg_{b_key}", float("nan")),
                    })

                print(f"   [{dom_label}] Seed {seed} | {var_code}: rho={combined['spearman_rho']:+.4f} | NDCG@20={combined['ndcg_20pct']:.4f} | OSE@20={combined['ose_20pct']:.4f}")

            stage_results["ablation"][var_code][str(seed)] = seed_res

    # Equivalence verification check (Gate 9C-3)
    print("\n>> Verifying B0 vs B2-static Equivalence (Gate 9C-3)...")
    for seed in seeds:
        for dom in ["in_domain", "zero_shot"]:
            rho_b0 = stage_results["ablation"]["B0"][str(seed)][dom]["spearman_rho"]
            rho_b2s = stage_results["ablation"]["B2_static"][str(seed)][dom]["spearman_rho"]
            diff = abs(rho_b0 - rho_b2s)
            print(f"   [Seed {seed}] {dom}: B0 rho={rho_b0:+.5f}, B2_static rho={rho_b2s:+.5f}, |diff|={diff:.2e}")
            assert diff < 1e-5, f"Equivalence violation: B0 vs B2_static diff={diff}"

    # ═══════════════════════════════════════════════════════════════════════
    # Part 9C-3: Temporal Dynamics (Frame Stream & Adaptation Convergence)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 80}")
    print("   PART 9C-3: TEMPORAL ADAPTATION DYNAMICS ANALYSIS")
    print(f"{'#' * 80}")

    raw_dataset = load_canonical_oracle_dataset()
    val_raw = raw_dataset.get_split("validation")
    val_keys = [(m.scene, m.frame) for m in val_raw.metadata]
    Z_val = transform_grouped_features(val_raw.X_np, val_keys, variant=BASE_REPRESENTATION)

    test_raw = raw_dataset.get_split("test")
    test_keys = [(m.scene, m.frame) for m in test_raw.metadata]
    Z_test = transform_grouped_features(test_raw.X_np, test_keys, variant=BASE_REPRESENTATION)

    eval_sequences = [
        ("tum_fr1_desk_val", val_raw, Z_val),
        ("tum_fr2_xyz", test_raw, Z_test),
    ]

    # Evaluate temporal dynamics for B0, B2_static, and B2 across beta in {0.80, 0.90, 0.95}
    temp_configs = [
        ("B0", "A1_standard", 0.0),
        ("B2_static", "A1_b2_static_start", 0.0),
        ("B2_beta_0.80", "A1_online_adaptive", 0.80),
        ("B2_beta_0.90", "A1_online_adaptive", 0.90),
        ("B2_beta_0.95", "A1_online_adaptive", 0.95),
    ]

    for seed in seeds:
        model = models[seed]
        for dom_label, split_raw, Z_split in eval_sequences:
            unique_frames = sorted(list(set(m.frame for m in split_raw.metadata)))

            for cfg_label, base_type, beta_val in temp_configs:
                if base_type == "A1_standard":
                    norm = StandardNormalizer(eps=EPS).fit(Z_train)
                elif base_type == "A1_b2_static_start":
                    norm = B2StaticNormalizer(eps=EPS).fit(Z_train)
                else:
                    norm = OnlineEMANormalizer(beta=beta_val, eps=EPS).fit(Z_train)

                prev_top_k = None

                for step, fr in enumerate(unique_frames):
                    indices = [i for i, m in enumerate(split_raw.metadata) if m.frame == fr]
                    if not indices:
                        continue
                    X_frame = Z_split[indices]
                    oracle_u_frame = split_raw.utility_tensor[indices].numpy()

                    t0 = time.perf_counter()
                    if hasattr(norm, "update_and_transform_frame"):
                        Z_norm, step_m = norm.update_and_transform_frame(X_frame, frame_id=fr)
                        d_norm = step_m["d_norm"]
                        mu_shift = step_m["mu_l2_shift"]
                        sigma_shift = step_m["sigma_l2_shift"]
                        mean_mu = step_m["mean_mu"]
                        mean_sigma = step_m["mean_sigma"]
                    else:
                        Z_norm = norm.transform(X_frame)
                        d_norm = 0.0
                        mu_shift = 0.0
                        sigma_shift = 0.0
                        mean_mu = float(np.mean(norm.mean))
                        mean_sigma = float(np.mean(norm.std))
                    lat_ms = (time.perf_counter() - t0) * 1000.0

                    # Compute predictions
                    with torch.no_grad():
                        Z_t = torch.from_numpy(Z_norm).float().to(device)
                        _, _, pred_u_t = model(Z_t)
                        pred_u = pred_u_t.cpu().numpy()

                    # Compute per-frame rho_t
                    rho_t = compute_per_frame_spearman(pred_u, oracle_u_frame, list(range(len(indices))))

                    # Compute selection overlap with previous frame
                    k = max(1, int(0.20 * len(pred_u)))
                    top_k_indices = set(np.argsort(-pred_u)[:k])
                    if prev_top_k is not None and len(top_k_indices) > 0 and len(prev_top_k) > 0:
                        overlap_20 = len(top_k_indices & prev_top_k) / float(len(top_k_indices))
                    else:
                        overlap_20 = 1.0
                    prev_top_k = top_k_indices

                    temporal_rows.append({
                        "config": cfg_label,
                        "base_type": base_type,
                        "beta": beta_val,
                        "seed": seed,
                        "domain": dom_label,
                        "step": step,
                        "frame": fr,
                        "n_samples": len(indices),
                        "latency_ms": lat_ms,
                        "t_norm_per_cand_us": (lat_ms / max(len(indices), 1)) * 1000.0,
                        "d_norm": d_norm,
                        "mu_l2_shift": mu_shift,
                        "sigma_l2_shift": sigma_shift,
                        "mean_mu": mean_mu,
                        "mean_sigma": mean_sigma,
                        "overlap_20": overlap_20,
                        "per_frame_rho": rho_t,
                    })

    # Multi-step sequential adaptation stream to visualize convergence from early frames -> steady state
    print("\n>> Simulating multi-step sequential adaptation stream on zero-shot domain...")
    for seed in seeds:
        model = models[seed]
        cands = domain_candidates[("zero_shot", seed)]
        X, oracle_u, _, _ = extract_features_from_candidates(cands)
        cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
        Z = transform_grouped_features(X, cand_keys, variant=BASE_REPRESENTATION)

        # Partition into sub-batches to simulate 10 sequential arrival steps
        n_chunks = 10
        chunk_size = len(Z) // n_chunks
        chunks = [
            (i, Z[i * chunk_size : (i + 1) * chunk_size], oracle_u[i * chunk_size : (i + 1) * chunk_size])
            for i in range(n_chunks)
        ]

        for beta_val in [0.80, 0.90, 0.95]:
            norm = OnlineEMANormalizer(beta=beta_val, eps=EPS).fit(Z_train)
            prev_top = None

            for step, (chunk_idx, X_chk, o_chk) in enumerate(chunks):
                t0 = time.perf_counter()
                Z_norm, step_m = norm.update_and_transform_frame(X_chk, frame_id=step)
                lat_ms = (time.perf_counter() - t0) * 1000.0

                with torch.no_grad():
                    Z_t = torch.from_numpy(Z_norm).float().to(device)
                    _, _, pred_u_t = model(Z_t)
                    pred_u = pred_u_t.cpu().numpy()

                rho_t = compute_per_frame_spearman(pred_u, o_chk, list(range(len(X_chk))))
                k = max(1, int(0.20 * len(pred_u)))
                top_k = set(np.argsort(-pred_u)[:k])
                overlap = len(top_k & prev_top) / float(len(top_k)) if prev_top is not None else 1.0
                prev_top = top_k

                temporal_rows.append({
                    "config": f"B2_beta_{beta_val:.2f}",
                    "base_type": "sequential_stream",
                    "beta": beta_val,
                    "seed": seed,
                    "domain": "stream_tum_fr2_xyz",
                    "step": step,
                    "frame": step,
                    "n_samples": len(X_chk),
                    "latency_ms": lat_ms,
                    "t_norm_per_cand_us": (lat_ms / max(len(X_chk), 1)) * 1000.0,
                    "d_norm": step_m["d_norm"],
                    "mu_l2_shift": step_m["mu_l2_shift"],
                    "sigma_l2_shift": step_m["sigma_l2_shift"],
                    "mean_mu": step_m["mean_mu"],
                    "mean_sigma": step_m["mean_sigma"],
                    "overlap_20": overlap,
                    "per_frame_rho": rho_t,
                })

    # ═══════════════════════════════════════════════════════════════════════
    # Part 9C-4: Perturbation Robustness Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 80}")
    print("   PART 9C-4: CONTROLLED COVARIATE PERTURBATION ROBUSTNESS")
    print(f"{'#' * 80}")

    # Evaluate B0 vs B2 under scale shifts a in {0.8, 1.0, 1.2} and offsets b in {-0.2, 0.0, 0.2}
    perturbation_conditions = []
    # 1. Scale shifts (with zero offset)
    for a in PERTURBATION_SCALES:
        perturbation_conditions.append(("scale", a, 0.0, f"scale_{a:.1f}"))
    # 2. Offset shifts (with unit scale)
    for b in PERTURBATION_OFFSETS:
        if b != 0.0:  # offset 0.0 already covered by scale 1.0, offset 0.0
            perturbation_conditions.append(("offset", 1.0, b, f"offset_{b:+.1f}"))

    for p_type, scale_a, offset_b, cond_label in perturbation_conditions:
        print(f"\n--- Condition: {cond_label} (scale={scale_a:.2f}, offset={offset_b:+.2f}) ---")
        stage_results["perturbation"][cond_label] = {}

        for seed in seeds:
            model = models[seed]
            cands = domain_candidates[("zero_shot", seed)]
            X_clean, oracle_u, delta_q, costs = extract_features_from_candidates(cands)
            cand_keys = [(c.get('scene', ''), c.get('frame', 0)) for c in cands]
            Z_clean = transform_grouped_features(X_clean, cand_keys, variant=BASE_REPRESENTATION)

            # Apply perturbation
            Z_pert = apply_feature_perturbation(Z_clean, scale=scale_a, offset=offset_b)

            cond_seed_res = {}

            for var_code in ["B0", "B2"]:
                if var_code == "B0":
                    norm = StandardNormalizer(eps=EPS).fit(Z_train)
                    Z_norm = norm.transform(Z_pert)
                    with torch.no_grad():
                        Z_t = torch.from_numpy(Z_norm).float().to(device)
                        _, _, pred_u_t = model(Z_t)
                        pred_u = pred_u_t.cpu().numpy()
                else:
                    norm = OnlineEMANormalizer(beta=0.90, eps=EPS).fit(Z_train)
                    unique_frames = sorted(list(set(c.get('frame', 0) for c in cands)))
                    pred_u = np.zeros(len(cands), dtype=np.float32)
                    for fr in unique_frames:
                        idx = [i for i, c in enumerate(cands) if c.get('frame', 0) == fr]
                        fr_norm, _ = norm.update_and_transform_frame(Z_pert[idx], frame_id=fr)
                        with torch.no_grad():
                            fr_tensor = torch.from_numpy(fr_norm).float().to(device)
                            _, _, pred_u_fr = model(fr_tensor)
                            pred_u[idx] = pred_u_fr.cpu().numpy()

                rq1 = evaluate_rq1_prediction(pred_u=pred_u, oracle_u=oracle_u)
                rq2 = evaluate_rq2_selection(
                    pred_u=pred_u, oracle_u=oracle_u,
                    delta_q=delta_q, costs=costs,
                )
                combined = {**rq1, **rq2}
                cond_seed_res[var_code] = combined

                perturbation_rows.append({
                    "condition": cond_label,
                    "perturbation_type": p_type,
                    "scale": scale_a,
                    "offset": offset_b,
                    "variant": var_code,
                    "seed": seed,
                    "spearman_rho": combined["spearman_rho"],
                    "spearman_pval": combined["spearman_pval"],
                    "pearson_r": combined["pearson_r"],
                    "ndcg_20pct": combined.get("ndcg_20pct", float("nan")),
                    "ose_20pct": combined.get("ose_20pct", float("nan")),
                    "regret_20pct": combined.get("regret_20pct", float("nan")),
                    "dq_10pct": combined.get("realized_delta_q_10pct", float("nan")),
                    "dq_20pct": combined.get("realized_delta_q_20pct", float("nan")),
                    "dq_40pct": combined.get("realized_delta_q_40pct", float("nan")),
                    "dq_60pct": combined.get("realized_delta_q_60pct", float("nan")),
                    "dq_80pct": combined.get("realized_delta_q_80pct", float("nan")),
                })

                print(f"   [Seed {seed}] {var_code} | {cond_label}: rho={combined['spearman_rho']:+.4f} | NDCG@20={combined['ndcg_20pct']:.4f} | OSE@20={combined['ose_20pct']:.4f}")

            stage_results["perturbation"][cond_label][str(seed)] = cond_seed_res

    # ═══════════════════════════════════════════════════════════════════════
    # Export Results & Artifacts
    # ═══════════════════════════════════════════════════════════════════════
    print("\n>> Exporting Phase 9C CSV artifacts...")

    # 1. sensitivity_metrics.csv
    sens_path = out_dir / OUTPUT_FILES_9C["sensitivity_metrics"]
    pd.DataFrame(sensitivity_rows).to_csv(sens_path, index=False)
    print(f"   Saved sensitivity metrics: {sens_path}")

    # 2. ablation_metrics.csv
    ablation_path = out_dir / OUTPUT_FILES_9C["ablation_metrics"]
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    print(f"   Saved ablation metrics: {ablation_path}")

    # 3. temporal_dynamics.csv
    temp_path = out_dir / OUTPUT_FILES_9C["temporal_dynamics"]
    pd.DataFrame(temporal_rows).to_csv(temp_path, index=False)
    print(f"   Saved temporal dynamics: {temp_path}")

    # 4. perturbation_metrics.csv
    pert_path = out_dir / OUTPUT_FILES_9C["perturbation_metrics"]
    pd.DataFrame(perturbation_rows).to_csv(pert_path, index=False)
    print(f"   Saved perturbation metrics: {pert_path}")

    # 5. selection_metrics.csv
    sel_path = out_dir / OUTPUT_FILES_9C["selection_metrics"]
    pd.DataFrame(selection_rows).to_csv(sel_path, index=False)
    print(f"   Saved selection metrics: {sel_path}")

    # 6. runtime_metrics.csv
    run_cols = [
        "beta", "seed", "domain", "n_candidates",
        "t_ema_ms", "t_infer_ms", "t_selection_ms", "t_total_ms",
    ]
    df_run = pd.DataFrame(sensitivity_rows)[run_cols].copy()
    df_run["t_ema_per_cand_us"] = (df_run["t_ema_ms"] / df_run["n_candidates"]) * 1000.0
    df_run["t_infer_per_cand_us"] = (df_run["t_infer_ms"] / df_run["n_candidates"]) * 1000.0
    run_path = out_dir / OUTPUT_FILES_9C["runtime_metrics"]
    df_run.to_csv(run_path, index=False)
    print(f"   Saved runtime metrics: {run_path}")

    # 7. protocol.json
    proto_path = out_dir / OUTPUT_FILES_9C["protocol"]
    with open(proto_path, "w") as f:
        json.dump(protocol_to_dict(), f, indent=2)
    print(f"   Saved protocol: {proto_path}")

    # 8. stage_results.json
    raw_path = out_dir / OUTPUT_FILES_9C["stage_results"]
    with open(raw_path, "w") as f:
        json.dump(stage_results, f, indent=2, default=str)
    print(f"   Saved raw stage results: {raw_path}")

    print("\n=== Phase 9C Sensitivity & Robustness Run Completed Successfully ===")
    return stage_results


def main():
    parser = argparse.ArgumentParser(description="Phase 9C Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 1-seed smoke test (seed 42)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    seeds = [42] if args.smoke else (args.seeds or list(SEEDS))
    run_phase9c(seeds=seeds, device=args.device)


if __name__ == "__main__":
    main()
