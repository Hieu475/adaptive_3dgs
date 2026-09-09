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
    ResidualContextModel,
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
    GroupedBatchSampler,
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


def _get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def train_and_eval_variant(
    variant: str,
    dataset_path: str,
    output_dir: str,
    architecture: str = "residual",
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains a specific ablation variant and evaluates its prediction metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # C4 FIX: Use prepare_phase6_splits which fits normalizer on TRAIN ONLY
    mask = _get_variant_mask(variant)
    norm_path = os.path.join(output_dir, f"norm_{variant}.json")

    train_ds, val_ds, test_ds, normalizer = prepare_phase6_splits(
        dataset_paths=[dataset_path],
        normalizer_save_path=norm_path,
        variant=variant,
    )

    if len(train_ds) == 0:
        # Fall back to scene-based split if all samples are from same scene
        all_feats, all_dq, all_dt, all_u, samples = load_phase6_dataset(dataset_path)
        N = len(all_feats)

        # Fit normalizer on first 70% (deterministic, not random)
        normalizer = Phase6FeatureNormalizer()
        n_train = max(1, int(0.7 * N))
        normalizer.fit(all_feats[:n_train])  # FIT ON TRAIN PORTION ONLY
        normalizer.save_json(norm_path)

        feats_norm = normalizer.transform(all_feats)
        mask = _get_variant_mask(variant)
        feats_variant = feats_norm[:, mask]

        # Pre-pass empty utils
        empty_utils = {}
        for s in samples:
            if s.get("context_size", 0) == 0 or s.get("context_type") == "empty":
                k = (str(s["scene"]), int(s["frame"]), int(s["candidate_id"]))
                empty_utils[k] = float(s["utility_conditional"])

        target_r_list, empty_list, sample_gids = [], [], []
        group_key_to_id = {}
        for s in samples:
            k = (str(s["scene"]), int(s["frame"]), int(s["candidate_id"]))
            u = float(s["utility_conditional"])
            u_empty = empty_utils.get(k, u)
            target_r_list.append(u - u_empty)
            empty_list.append(1.0 if (s.get("context_size", 0) == 0 or s.get("context_type") == "empty") else 0.0)

            selected_ids = s.get("context_ids", []) or []
            context_ident = tuple(sorted([int(x) for x in selected_ids]))
            g_key = (str(s["scene"]), int(s["frame"]), context_ident)
            gid = group_key_to_id.setdefault(g_key, len(group_key_to_id))
            sample_gids.append(gid)

        target_r_arr = np.array(target_r_list, dtype=np.float32)
        empty_arr = np.array(empty_list, dtype=np.float32)
        sample_gids_arr = np.array(sample_gids, dtype=np.int64)

        n_val = max(1, int(0.15 * N))
        train_ds = Phase6UtilityDataset(
            feats_variant[:n_train], all_dq[:n_train], all_dt[:n_train], all_u[:n_train],
            target_r=target_r_arr[:n_train], is_empty=empty_arr[:n_train], group_ids=sample_gids_arr[:n_train],
        )
        val_ds = Phase6UtilityDataset(
            feats_variant[n_train:n_train + n_val],
            all_dq[n_train:n_train + n_val],
            all_dt[n_train:n_train + n_val],
            all_u[n_train:n_train + n_val],
            target_r=target_r_arr[n_train:n_train + n_val],
            is_empty=empty_arr[n_train:n_train + n_val],
            group_ids=sample_gids_arr[n_train:n_train + n_val],
        )
        test_ds = Phase6UtilityDataset(
            feats_variant[n_train + n_val:],
            all_dq[n_train + n_val:],
            all_dt[n_train + n_val:],
            all_u[n_train + n_val:],
            target_r=target_r_arr[n_train + n_val:],
            is_empty=empty_arr[n_train + n_val:],
            group_ids=sample_gids_arr[n_train + n_val:],
        )
        print(f"  [C4 FIX] Normalizer fit on train only (first {n_train}/{N} samples)")

    train_sampler = GroupedBatchSampler(train_ds.group_ids, max_batch_size=16, shuffle=True, seed=seed)
    val_sampler = GroupedBatchSampler(val_ds.group_ids, max_batch_size=16, shuffle=False)
    test_sampler = GroupedBatchSampler(test_ds.group_ids, max_batch_size=16, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_sampler=test_sampler)

    # Model config
    cfg = create_ablation_variant(variant)
    dev = torch.device(device)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p4_ckpt = None
    if architecture == "residual":
        p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", f"two_head_mlp_seed_{seed}.pt")
        if not os.path.exists(p4_ckpt):
            p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", "two_head_mlp_seed_42.pt")
        if not os.path.exists(p4_ckpt):
            raise FileNotFoundError(f"P0.2 REQUIREMENT: Pretrained Phase 4 checkpoint required at {p4_ckpt}")

        from research.utility_models import TwoHeadMLP
        p4_net = TwoHeadMLP(in_features=11)
        ckpt_data = torch.load(p4_ckpt, map_location="cpu", weights_only=False)
        p4_net.load_state_dict(ckpt_data.get("model_state", ckpt_data))
        p4_net.eval()
        for p in p4_net.parameters():
            p.requires_grad = False
        print(f"  [Residual] Loaded and strictly froze pre-trained Phase 4 weights from {p4_ckpt}")

        model = ResidualContextModel(cfg, p4_model=p4_net).to(dev)
        trainable_params = model.context_parameters()
        assert all(not p.requires_grad for p in model.p4_model.parameters()), "P4 parameters must not require grad!"
        p4_before = {k: v.clone().cpu() for k, v in model.p4_model.state_dict().items()}
    else:
        model = ContextAwareTwoHeadMLP(cfg).to(dev)
        trainable_params = list(model.parameters())

    train_lr = lr if architecture != "residual" else (lr if lr != 1e-3 else 2e-4)
    optimizer = torch.optim.Adam(trainable_params, lr=train_lr, weight_decay=1e-5)

    if architecture == "residual":
        loss_fn = Phase6Loss(
            lambda_q=0.0,
            lambda_c=0.0,
            lambda_r=1.0,
            lambda_list=0.5,
            lambda_res=1.0,
            lambda_zero=0.5,
            scale_u=1.0,
            require_group_ids=True,
        )
    else:
        loss_fn = Phase6Loss(
            lambda_q=1.0,
            lambda_c=0.5,
            lambda_r=0.1,
            lambda_list=0.5,
            lambda_res=0.0,
            lambda_zero=0.0,
            require_group_ids=True,
        )

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        if hasattr(model, 'p4_model'):
            model.p4_model.eval()

        for batch in train_loader:
            x = batch['features'].to(dev)
            t_q = batch['delta_q'].to(dev)
            t_t = batch['delta_t'].to(dev)
            t_u = batch['utility'].to(dev)
            t_r = batch.get('target_r')
            if t_r is not None:
                t_r = t_r.to(dev)
            is_empty = batch.get('is_empty')
            if is_empty is not None:
                is_empty = is_empty.to(dev)
            group_ids = batch.get('group_id')
            if group_ids is not None:
                group_ids = group_ids.to(dev)

            optimizer.zero_grad()
            if hasattr(model, 'context_fusion'):
                p_q, p_t, p_u, p_r = model(x, return_residual=True)
            else:
                p_q, p_t, p_u = model(x)
                p_r = None

            losses = loss_fn(
                pred_q=p_q,
                pred_t=p_t,
                pred_u=p_u,
                target_q=t_q,
                target_t=t_t,
                target_u=t_u,
                group_ids=group_ids,
                pred_r=p_r,
                target_r=t_r,
                is_empty=is_empty,
            )
            losses['total'].backward()
            clip_params = model.context_parameters() if hasattr(model, 'context_parameters') else model.parameters()
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_b = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(dev)
                t_q = batch['delta_q'].to(dev)
                t_t = batch['delta_t'].to(dev)
                t_u = batch['utility'].to(dev)
                t_r = batch.get('target_r')
                if t_r is not None:
                    t_r = t_r.to(dev)
                is_empty = batch.get('is_empty')
                if is_empty is not None:
                    is_empty = is_empty.to(dev)
                group_ids = batch.get('group_id')
                if group_ids is not None:
                    group_ids = group_ids.to(dev)

                if hasattr(model, 'context_fusion'):
                    p_q, p_t, p_u, p_r = model(x, return_residual=True)
                else:
                    p_q, p_t, p_u = model(x)
                    p_r = None

                losses = loss_fn(
                    pred_q=p_q,
                    pred_t=p_t,
                    pred_u=p_u,
                    target_q=t_q,
                    target_t=t_t,
                    target_u=t_u,
                    group_ids=group_ids,
                    pred_r=p_r,
                    target_r=t_r,
                    is_empty=is_empty,
                )
                val_loss += losses['total'].item()
                n_b += 1
        val_loss /= max(1, n_b)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Verify P4 backbone invariance (P0.2)
    if architecture == "residual":
        for k, v in model.p4_model.state_dict().items():
            assert torch.equal(v.cpu(), p4_before[k]), f"P4 backbone weights changed for {k}!"
        print(f"  [Verification] P4 backbone weights remained strictly invariant for {variant}.")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_pred_u, all_true_u = [], []
    all_pred_q, all_true_q = [], []
    all_test_gids = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(dev)
            p_q, p_t, p_u = model(x)
            all_pred_u.extend(p_u.cpu().numpy().tolist())
            all_true_u.extend(batch['utility'].numpy().tolist())
            all_pred_q.extend(p_q.cpu().numpy().tolist())
            all_true_q.extend(batch['delta_q'].numpy().tolist())
            all_test_gids.extend(batch['group_id'].cpu().numpy().tolist())

    arr_pred_u = np.array(all_pred_u)
    arr_true_u = np.array(all_true_u)
    arr_test_gids = np.array(all_test_gids)

    rho_u, _ = safe_spearmanr(arr_pred_u, arr_true_u)
    r_u, _ = safe_pearsonr(arr_pred_u, arr_true_u)
    mae_u = float(np.mean(np.abs(arr_pred_u - arr_true_u)))

    # Compute Group-wise NDCG@5
    ndcg_list = []
    for gid in np.unique(arr_test_gids):
        m = (arr_test_gids == gid)
        if np.sum(m) >= 2:
            ndcg_list.append(compute_ndcg_at_k(arr_pred_u[m], arr_true_u[m], k=5))
    macro_ndcg_5 = float(np.mean(ndcg_list)) if ndcg_list else float(compute_ndcg_at_k(arr_pred_u, arr_true_u, k=5))

    ckpt_path = os.path.join(output_dir, f"model_{variant}.pt")
    torch.save({
        "schema_version": "phase6-v2",
        "architecture": "residual_context" if architecture == "residual" else "direct_context",
        "variant": variant,
        "seed": seed,
        "protocol_version": "v1",
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "normalizer_path": norm_path,
        "p4_checkpoint": p4_ckpt if architecture == "residual" else None,
        "p4_frozen": True if architecture == "residual" else False,
        "git_commit": _get_git_commit(),
        "test_metrics": {
            "spearman_rho": float(rho_u),
            "pearson_r": float(r_u),
            "ndcg_5": float(macro_ndcg_5),
            "mae_utility": float(mae_u),
        },
    }, ckpt_path)

    return {
        "variant": variant,
        "features_dim": int(np.sum(mask)),
        "spearman_rho": float(rho_u),
        "pearson_r": float(r_u),
        "ndcg_5": float(macro_ndcg_5),
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


def run_shuffle_test(
    model: torch.nn.Module,
    test_ds: Phase6UtilityDataset,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates the contribution of each context group by shuffling features across test samples.

    Section XXVIII Protocol:
      - Keep s_i (0:11) fixed, but shuffle each context group across candidates.
      - If delta_rho > 0: context group makes a positive contribution.
      - If delta_rho <= 0: context group is not helping or is hurting prediction.
    """
    dev = torch.device(device)
    model.eval()

    X_test = test_ds.features.clone()
    u_test = test_ds.utility.cpu().numpy()
    N = len(X_test)
    if N < 5:
        return {}

    with torch.no_grad():
        _, _, base_pred_u = model(X_test.to(dev))
        base_rho, _ = safe_spearmanr(base_pred_u.cpu().numpy(), u_test)

    shuffle_results = {"baseline_rho": float(base_rho), "groups": {}}
    rng = np.random.default_rng(42)

    groups = {
        "neighbor": slice(11, 19),
        "overlap": slice(19, 24),
        "selected": slice(24, 32),
    }

    for g_name, g_slice in groups.items():
        if X_test.shape[1] >= g_slice.stop:
            X_shuffled = X_test.clone()
            perm = rng.permutation(N)
            X_shuffled[:, g_slice] = X_test[perm, g_slice]

            with torch.no_grad():
                _, _, shuf_pred_u = model(X_shuffled.to(dev))
                shuf_rho, _ = safe_spearmanr(shuf_pred_u.cpu().numpy(), u_test)

            delta_rho = float(base_rho - shuf_rho)
            shuffle_results["groups"][g_name] = {
                "shuffled_rho": float(shuf_rho),
                "delta_rho": delta_rho,
                "actively_used": bool(abs(delta_rho) > 0.005),
                "positive_contribution": bool(delta_rho > 0.0),
            }

    return shuffle_results


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Ablation Study & Interaction Analysis")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true", default=False)
    parser.add_argument("--architecture", type=str, default="residual", choices=["direct", "residual"],
                        help="Model architecture: direct (ContextAwareTwoHeadMLP) or residual (ResidualContextModel)")
    parser.add_argument("--combinatorial", action="store_true", default=False,
                        help="Run full 8-variant 2-way combinatorial ablation")
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
    print(f"  Dataset:      {ds_path}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Mode:         {'Combinatorial (8 variants)' if args.combinatorial else 'Ladder (V8-V11)'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 40 if args.quick else args.epochs

    if args.combinatorial:
        variants = [
            "self_only", "self_neighbor", "self_overlap", "self_selected",
            "self_neighbor_overlap", "self_neighbor_selected", "self_overlap_selected", "all_features"
        ]
    else:
        variants = ["V8", "V9", "V10", "V11"]

    variant_results = []

    print(f"\n[Part 1] Training and Evaluating Representation ({len(variants)} variants)...")
    for var in variants:
        t0 = time.perf_counter()
        res = train_and_eval_variant(
            variant=var,
            dataset_path=ds_path,
            output_dir=output_dir,
            architecture=args.architecture,
            seed=args.seed,
            epochs=epochs,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        res["train_time_s"] = elapsed
        variant_results.append(res)
        print(f"  {var:<24} ({res['features_dim']:2d} dims) | Spearman rho: {res['spearman_rho']:+.4f} | NDCG@5: {res['ndcg_5']:.4f} | MAE: {res['mae_utility']:.2e} | Time: {elapsed:.1f}s")

    # 2. Pairwise Interaction Analysis
    print("\n[Part 2] Computing Interaction Residuals Stratified by Co-visibility Overlap...")
    with open(ds_path, "r") as f:
        samples = json.load(f)

    interaction_results = analyze_interaction_residuals(samples)
    for cat, vals in interaction_results.items():
        print(f"  {cat:<16} | N={vals['n_samples']:3d} | Sub-additive Frac: {vals['sub_additive_fraction']*100:5.1f}% | Additivity Ratio: {vals['mean_additivity_ratio']:.3f}")

    # 3. Shuffle Sensitivity Test (Section XXVIII)
    full_var = "all_features" if args.combinatorial else "V11"
    best_model_path = os.path.join(output_dir, f"model_{full_var}.pt")
    shuffle_results = {}
    if os.path.exists(best_model_path):
        cfg = create_ablation_variant(full_var)
        if args.architecture == "residual":
            p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", f"two_head_mlp_seed_{args.seed}.pt")
            if not os.path.exists(p4_ckpt):
                p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", "two_head_mlp_seed_42.pt")
            from research.utility_models import TwoHeadMLP
            p4_net = TwoHeadMLP(in_features=11)
            ckpt_data = torch.load(p4_ckpt, map_location="cpu", weights_only=False)
            p4_net.load_state_dict(ckpt_data.get("model_state", ckpt_data))
            p4_net.eval()
            for p in p4_net.parameters():
                p.requires_grad = False
            full_model = ResidualContextModel(cfg, p4_model=p4_net).to(device)
        else:
            full_model = ContextAwareTwoHeadMLP(cfg).to(device)
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state", ckpt)
        full_model.load_state_dict(state_dict)
        _, _, test_ds, _ = prepare_phase6_splits(dataset_paths=[ds_path], variant=full_var)
        if len(test_ds) > 0:
            shuffle_results = run_shuffle_test(full_model, test_ds, device=device)
            print("\n[Part 3] Shuffle Sensitivity Audit (Section XXVIII)...")
            print(f"  Baseline Test rho: {shuffle_results['baseline_rho']:+.4f}")
            for g_name, g_info in shuffle_results.get("groups", {}).items():
                print(f"  Shuffled {g_name:<8s} | rho={g_info['shuffled_rho']:+.4f} | delta_rho={g_info['delta_rho']:+.4f} | active={g_info['actively_used']}")

    # 4. Print Summary Table
    print("\n" + "=" * 80)
    print("  ABLATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Variant':<24} | {'Dims':<5} | {'Spearman ρ':<11} | {'NDCG@5':<8}")
    print("-" * 80)
    for r in variant_results:
        print(f"{r['variant']:<24} | {r['features_dim']:<5d} | {r['spearman_rho']:<+11.4f} | {r['ndcg_5']:<8.4f}")
    print("=" * 80)

    # 5. Save Artifacts
    artifact = {
        "architecture": args.architecture,
        "ablation_ladder": variant_results,
        "interaction_analysis": interaction_results,
        "shuffle_test": shuffle_results,
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
