#!/usr/bin/env python3
"""Phase 6: Ablation Study & Interaction Analysis (Step 13).

Evaluates:
  1. Input Representation Ladder:
     - V8:  Self only (11 dims, Phase 4 equivalent)
     - V9:  Self + Neighborhood (19 dims)
     - V10: Self + Neighborhood + Overlap (24 dims)
     - V11: Self + Neighborhood + Overlap + Selected (32 dims, static)
     - V12: V11 + Adaptive Greedy (32 dims, dynamic S_t)

  2. Interaction Analysis:
     - Interaction residual: I(i, j) = Delta Q({i, j}) - Delta Q({i}) - Delta Q({j})
     - Additivity ratio: R_add = Delta Q({i, j}) / (Delta Q({i}) + Delta Q({j}))
     - Stratified by pixel overlap IoU:
         * Low overlap: IoU < 0.10
         * Medium overlap: 0.10 <= IoU <= 0.30
         * High overlap: IoU > 0.30

Usage:
    python experiments/run_phase6_ablation.py --seed 42
    python experiments/run_phase6_ablation.py --quick
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase6_context import (
    PHASE6_FEATURE_DIM,
    SELF_SLICE,
    NEIGHBOR_SLICE,
    OVERLAP_SLICE,
    SELECTED_SLICE,
)
from research.phase6_model import (
    ContextAwareTwoHeadMLP,
    Phase6ModelConfig,
    Phase6Loss,
    create_ablation_variant,
    FrozenContextPredictor,
)
from research.phase6_dataset import (
    load_phase6_dataset,
    prepare_phase6_splits,
    Phase6UtilityDataset,
    Phase6FeatureNormalizer,
    _get_variant_mask,
)
from research.phase6_selection import (
    select_phase6_subset,
    adaptive_greedy_select,
    static_context_select,
)
from research.utility_metrics import (
    safe_spearmanr,
    safe_pearsonr,
    compute_ndcg_at_k,
)


def train_and_eval_variant(
    variant: str,
    dataset_path: str,
    output_dir: str,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains a specific ablation variant and evaluates its prediction metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_feats, all_dq, all_dt, all_u, samples = load_phase6_dataset(dataset_path)
    N = len(all_feats)

    # Fit normalizer on all_feats
    norm_path = os.path.join(output_dir, f"norm_{variant}.json")
    normalizer = Phase6FeatureNormalizer()
    normalizer.fit(all_feats)
    normalizer.save_json(norm_path)

    feats_norm = normalizer.transform(all_feats)
    mask = _get_variant_mask(variant)
    feats_variant = feats_norm[:, mask]

    # Split 70/15/15
    perm = np.random.permutation(N)
    n_train = max(1, int(0.7 * N))
    n_val = max(1, int(0.15 * N))

    idx_tr = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_te = perm[n_train + n_val:]

    train_ds = Phase6UtilityDataset(feats_variant[idx_tr], all_dq[idx_tr], all_dt[idx_tr], all_u[idx_tr])
    val_ds = Phase6UtilityDataset(feats_variant[idx_val], all_dq[idx_val], all_dt[idx_val], all_u[idx_val])
    test_ds = Phase6UtilityDataset(feats_variant[idx_te], all_dq[idx_te], all_dt[idx_te], all_u[idx_te])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    # Model config
    cfg = create_ablation_variant(variant)
    dev = torch.device(device)
    model = ContextAwareTwoHeadMLP(cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = Phase6Loss(lambda_q=1.0, lambda_c=0.5, lambda_r=0.1)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch['features'].to(dev)
            t_q = batch['delta_q'].to(dev)
            t_t = batch['delta_t'].to(dev)
            t_u = batch['utility'].to(dev)

            optimizer.zero_grad()
            p_q, p_t, p_u = model(x)
            losses = loss_fn(p_q, p_t, p_u, t_q, t_t, t_u)
            losses['total'].backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(dev)
                p_q, p_t, p_u = model(x)
                losses = loss_fn(p_q, p_t, p_u, batch['delta_q'].to(dev), batch['delta_t'].to(dev), batch['utility'].to(dev))
                val_loss += losses['total'].item()
                n_b += 1
        val_loss /= max(1, n_b)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_pred_u, all_true_u = [], []
    all_pred_q, all_true_q = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(dev)
            p_q, p_t, p_u = model(x)
            all_pred_u.extend(p_u.cpu().numpy().tolist())
            all_true_u.extend(batch['utility'].numpy().tolist())
            all_pred_q.extend(p_q.cpu().numpy().tolist())
            all_true_q.extend(batch['delta_q'].numpy().tolist())

    rho_u, _ = safe_spearmanr(np.array(all_pred_u), np.array(all_true_u))
    r_u, _ = safe_pearsonr(np.array(all_pred_u), np.array(all_true_u))
    ndcg_5 = compute_ndcg_at_k(np.array(all_pred_u), np.array(all_true_u), k=5)
    mae_u = float(np.mean(np.abs(np.array(all_pred_u) - np.array(all_true_u))))

    ckpt_path = os.path.join(output_dir, f"model_{variant}.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

    return {
        "variant": variant,
        "features_dim": int(np.sum(mask)),
        "spearman_rho": float(rho_u),
        "pearson_r": float(r_u),
        "ndcg_5": float(ndcg_5),
        "mae_utility": float(mae_u),
        "val_loss": float(best_loss),
        "ckpt_path": ckpt_path,
        "norm_path": norm_path,
    }


def analyze_interaction_residuals(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze pairwise interaction and non-additivity stratified by overlap."""
    # From the dataset samples, compute interaction where context_size >= 1
    strata = {
        "low_overlap": {"residuals": [], "additivity_ratios": [], "count": 0},
        "medium_overlap": {"residuals": [], "additivity_ratios": [], "count": 0},
        "high_overlap": {"residuals": [], "additivity_ratios": [], "count": 0},
    }

    for s in samples:
        if s.get("context_size", 0) <= 0:
            continue

        dq_si = float(s.get("delta_q_si", 0.0))
        dq_s = float(s.get("delta_q_s", 0.0))
        dq_i_cond = float(s.get("delta_q_conditional", 0.0))

        # Overlap feature (mean_overlap or candidate_selected_overlap)
        overlap = float(s.get("overlap_features", {}).get("mean_overlap", 0.0))
        if overlap == 0.0:
            overlap = float(s.get("selected_features", {}).get("candidate_selected_overlap", 0.0))

        # Interaction: delta_q(S U {i}) - delta_q(S) - delta_q(i)
        # Using conditional marginal gain relative to single-candidate gain
        residual = dq_i_cond - dq_si
        ratio = dq_si / (dq_s + dq_i_cond + 1e-8) if (dq_s + dq_i_cond) > 1e-8 else 1.0

        if overlap < 0.10:
            cat = "low_overlap"
        elif overlap <= 0.30:
            cat = "medium_overlap"
        else:
            cat = "high_overlap"

        strata[cat]["residuals"].append(residual)
        strata[cat]["additivity_ratios"].append(ratio)
        strata[cat]["count"] += 1

    summary = {}
    for cat, data in strata.items():
        res_arr = np.array(data["residuals"]) if data["residuals"] else np.zeros(1)
        ratio_arr = np.array(data["additivity_ratios"]) if data["additivity_ratios"] else np.ones(1)
        summary[cat] = {
            "n_samples": data["count"],
            "mean_interaction_residual": float(np.mean(res_arr)),
            "sub_additive_fraction": float(np.mean(res_arr < 0)),
            "mean_additivity_ratio": float(np.mean(ratio_arr)),
            "median_additivity_ratio": float(np.median(ratio_arr)),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Ablation Study & Interaction Analysis")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "results", "phase6_context_utility", "ablation")
    os.makedirs(output_dir, exist_ok=True)

    ds_path = args.dataset or os.path.join(
        repo_root, "results", "phase6_context_utility", "datasets", f"conditional_oracle_seed_{args.seed}.json"
    )

    print("=" * 80)
    print("  PHASE 6: ABLATION STUDY & INTERACTION ANALYSIS (STEP 13)")
    print("=" * 80)
    print(f"  Dataset: {ds_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 40 if args.quick else args.epochs

    # 1. Train and Evaluate Feature Ladder Variants (V8, V9, V10, V11)
    variants = ["V8", "V9", "V10", "V11"]
    variant_results = []

    print("\n[Part 1] Training and Evaluating Representation Ladder (V8 -> V11)...")
    for var in variants:
        t0 = time.perf_counter()
        res = train_and_eval_variant(
            variant=var,
            dataset_path=ds_path,
            output_dir=output_dir,
            seed=args.seed,
            epochs=epochs,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        res["train_time_s"] = elapsed
        variant_results.append(res)
        print(f"  {var:<4} ({res['features_dim']:2d} dims) | Spearman rho: {res['spearman_rho']:+.4f} | NDCG@5: {res['ndcg_5']:.4f} | MAE: {res['mae_utility']:.2e} | Time: {elapsed:.1f}s")

    # 2. Pairwise Interaction Analysis
    print("\n[Part 2] Computing Interaction Residuals Stratified by Co-visibility Overlap...")
    with open(ds_path, "r") as f:
        samples = json.load(f)

    interaction_results = analyze_interaction_residuals(samples)
    for cat, vals in interaction_results.items():
        print(f"  {cat:<16} | N={vals['n_samples']:3d} | Sub-additive Frac: {vals['sub_additive_fraction']*100:5.1f}% | Additivity Ratio: {vals['mean_additivity_ratio']:.3f}")

    # 3. Print Summary Table
    print("\n" + "=" * 80)
    print("  ABLATION LADDER SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Variant':<8} | {'Input Components':<35} | {'Dims':<5} | {'Spearman ρ':<11} | {'NDCG@5':<8}")
    print("-" * 80)
    comp_map = {
        "V8": "Self only (Phase 4 pointwise)",
        "V9": "Self + Neighborhood",
        "V10": "Self + Neighborhood + Overlap",
        "V11": "Self + Neigh + Overlap + Selected",
    }
    for r in variant_results:
        print(f"{r['variant']:<8} | {comp_map.get(r['variant'], ''):<35} | {r['features_dim']:<5d} | {r['spearman_rho']:<+11.4f} | {r['ndcg_5']:<8.4f}")
    print("=" * 80)

    # 4. Save Artifacts
    artifact = {
        "ablation_ladder": variant_results,
        "interaction_analysis": interaction_results,
        "thesis_confirmed": bool(
            interaction_results.get("high_overlap", {}).get("sub_additive_fraction", 0.0) >= 0.5
        ),
    }
    out_file = os.path.join(output_dir, "ablation_summary.json")
    with open(out_file, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[Saved] Ablation Artifacts: {out_file}")


if __name__ == "__main__":
    main()
