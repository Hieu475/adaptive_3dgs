#!/usr/bin/env python3
"""Phase 6 — Context-Aware Utility Model Training.

Trains a ContextAwareTwoHeadMLP on the conditional oracle dataset.

Usage:
    python experiments/train_phase6_model.py                       # Default V11 (full context)
    python experiments/train_phase6_model.py --variant V9           # Ablation: self + neighbor only
    python experiments/train_phase6_model.py --seed 42 --epochs 200
    python experiments/train_phase6_model.py --dataset results/phase6_context_utility/datasets/conditional_oracle_seed_42.json

Output:
    results/phase6_context_utility/checkpoints/
    ├── context_mlp_{variant}_seed_{seed}.pt
    └── training_log_{variant}_seed_{seed}.json

Invariants:
    - Normalization fitted strictly on train split only.
    - Validation used for early stopping, NOT for training.
    - Model architecture matches Phase6ModelConfig.
    - Checkpoint contains model_state, config, normalizer, training metrics.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase6_model import (
    ContextAwareTwoHeadMLP,
    ResidualContextModel,
    Phase6ModelConfig,
    Phase6Loss,
    create_ablation_variant,
)
from research.phase6_dataset import (
    load_phase6_dataset,
    prepare_phase6_splits,
    Phase6FeatureNormalizer,
    Phase6UtilityDataset,
    PHASE6_FEATURE_DIM,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Phase6Loss,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch with group-aware listwise ranking and residual loss (P0).

    Returns:
        Dict with 'loss', 'loss_q', 'loss_t', 'loss_r', 'loss_list', 'loss_res', 'loss_zero' averages.
    """
    model.train()
    if hasattr(model, 'p4_model'):
        model.p4_model.eval()  # Strictly ensure frozen Phase 4 backbone remains in eval mode
    total_loss = 0.0
    total_lq = 0.0
    total_lt = 0.0
    total_lr = 0.0
    total_ll = 0.0
    total_lres = 0.0
    total_lzero = 0.0
    n_batches = 0

    is_residual = hasattr(model, 'context_fusion')

    for batch in loader:
        x = batch['features'].to(device)
        tgt_q = batch['delta_q'].to(device)
        tgt_t = batch['delta_t'].to(device)
        tgt_u = batch['utility'].to(device)
        tgt_r = batch.get('target_r')
        if tgt_r is not None:
            tgt_r = tgt_r.to(device)
        is_empty = batch.get('is_empty')
        if is_empty is not None:
            is_empty = is_empty.to(device)
        group_ids = batch.get('group_id')
        if group_ids is not None:
            group_ids = group_ids.to(device)

        optimizer.zero_grad()
        if is_residual:
            pred_q, pred_t, pred_u, pred_r = model(x, return_residual=True)
        else:
            pred_q, pred_t, pred_u = model(x)
            pred_r = None

        losses = loss_fn(
            pred_q=pred_q,
            pred_t=pred_t,
            pred_u=pred_u,
            target_q=tgt_q,
            target_t=tgt_t,
            target_u=tgt_u,
            group_ids=group_ids,
            pred_r=pred_r,
            target_r=tgt_r,
            is_empty=is_empty,
        )
        losses['total'].backward()
        clip_params = model.context_parameters() if hasattr(model, 'context_parameters') else model.parameters()
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_lq += losses['loss_q'].item()
        total_lt += losses['loss_t'].item()
        total_lr += losses['loss_r'].item()
        total_ll += losses['loss_list'].item()
        total_lres += losses['loss_res'].item()
        total_lzero += losses['loss_zero'].item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_q': total_lq / max(n_batches, 1),
        'loss_t': total_lt / max(n_batches, 1),
        'loss_r': total_lr / max(n_batches, 1),
        'loss_list': total_ll / max(n_batches, 1),
        'loss_res': total_lres / max(n_batches, 1),
        'loss_zero': total_lzero / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Phase6Loss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset with group-aware listwise ranking and residual loss.

    Returns:
        Dict with loss averages + correlation metrics.
    """
    from scipy.stats import spearmanr

    model.eval()
    total_loss = 0.0
    total_lq = 0.0
    total_lt = 0.0
    n_batches = 0

    all_pred_u = []
    all_tgt_u = []
    all_pred_q = []
    all_tgt_q = []

    is_residual = hasattr(model, 'context_fusion')

    for batch in loader:
        x = batch['features'].to(device)
        tgt_q = batch['delta_q'].to(device)
        tgt_t = batch['delta_t'].to(device)
        tgt_u = batch['utility'].to(device)
        tgt_r = batch.get('target_r')
        if tgt_r is not None:
            tgt_r = tgt_r.to(device)
        is_empty = batch.get('is_empty')
        if is_empty is not None:
            is_empty = is_empty.to(device)
        group_ids = batch.get('group_id')
        if group_ids is not None:
            group_ids = group_ids.to(device)

        if is_residual:
            pred_q, pred_t, pred_u, pred_r = model(x, return_residual=True)
        else:
            pred_q, pred_t, pred_u = model(x)
            pred_r = None

        losses = loss_fn(
            pred_q=pred_q,
            pred_t=pred_t,
            pred_u=pred_u,
            target_q=tgt_q,
            target_t=tgt_t,
            target_u=tgt_u,
            group_ids=group_ids,
            pred_r=pred_r,
            target_r=tgt_r,
            is_empty=is_empty,
        )
        total_loss += losses['total'].item()
        total_lq += losses['loss_q'].item()
        total_lt += losses['loss_t'].item()
        n_batches += 1

        all_pred_u.extend(pred_u.cpu().numpy().tolist())
        all_tgt_u.extend(tgt_u.cpu().numpy().tolist())
        all_pred_q.extend(pred_q.cpu().numpy().tolist())
        all_tgt_q.extend(tgt_q.cpu().numpy().tolist())

    # Correlation metrics
    rho_u, p_u = spearmanr(all_pred_u, all_tgt_u) if len(all_pred_u) > 2 else (0.0, 1.0)
    rho_q, p_q = spearmanr(all_pred_q, all_tgt_q) if len(all_pred_q) > 2 else (0.0, 1.0)

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_q': total_lq / max(n_batches, 1),
        'loss_t': total_lt / max(n_batches, 1),
        'spearman_utility': float(rho_u) if not np.isnan(rho_u) else 0.0,
        'spearman_quality': float(rho_q) if not np.isnan(rho_q) else 0.0,
        'p_value_utility': float(p_u) if not np.isnan(p_u) else 1.0,
        'n_samples': len(all_pred_u),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Context-Aware Utility Training")
    parser.add_argument("--dataset", type=str, nargs="+", default=None,
                        help="Path(s) to conditional oracle JSON files")
    parser.add_argument("--variant", type=str, default="V11",
                        choices=["V8", "V9", "V10", "V11"],
                        help="Ablation variant (default: V11 = full context)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--lambda-q", type=float, default=1.0)
    parser.add_argument("--lambda-c", type=float, default=0.5)
    parser.add_argument("--lambda-r", type=float, default=0.1)
    parser.add_argument("--lambda-list", type=float, default=0.5,
                        help="Listwise KL loss weight")
    parser.add_argument("--architecture", type=str, default="residual",
                        choices=["direct", "residual"],
                        help="Model architecture: direct (ContextAwareTwoHeadMLP) or residual (ResidualContextModel, recommended)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"{'='*70}")
    print(f"  PHASE 6 — CONTEXT-AWARE UTILITY MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Variant:   {args.variant}")
    print(f"  Seed:      {args.seed}")
    print(f"  Device:    {device}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  λ_Q:       {args.lambda_q}")
    print(f"  λ_C:       {args.lambda_c}")
    print(f"  λ_R:       {args.lambda_r}")
    print()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(repo_root, "results", "phase6_context_utility")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ─── Load dataset ───
    if args.dataset is None:
        dataset_dir = os.path.join(output_dir, "datasets")
        dataset_paths = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.startswith("conditional_oracle_seed_") and f.endswith(".json")
        ])
    else:
        dataset_paths = args.dataset

    if not dataset_paths:
        print("[ERROR] No dataset files found. Run build_phase6_dataset.py first.")
        return

    print(f"[Data] Loading {len(dataset_paths)} dataset file(s)...")
    for p in dataset_paths:
        print(f"  - {p}")

    normalizer_path = os.path.join(output_dir, f"normalization_{args.variant}.json")
    train_ds, val_ds, test_ds, normalizer = prepare_phase6_splits(
        dataset_paths=dataset_paths,
        normalizer_save_path=normalizer_path,
        variant=args.variant,
    )

    print(f"\n[Data] Split sizes:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")

    if len(train_ds) == 0:
        print("[ERROR] No training data found!")
        print("  This means the dataset only contains 'cross_scene_test' samples.")
        print("  You must generate proper train/val splits first:")
        print("    - Train: tum_fr1_desk frames 0-40")
        print("    - Val:   tum_fr1_desk frames 41-60")
        print("    - Test:  tum_fr2_xyz")
        print()
        print("  Run: python experiments/build_phase6_dataset.py --scene tum_fr1_desk")
        print("  Then: python experiments/build_phase6_dataset.py --scene tum_fr2_xyz")
        raise RuntimeError(
            "C4 FIX: Refusing to train with data leakage. "
            "No training data found — generate proper splits first. "
            "The old prototype fallback (fit normalizer on ALL data including test, "
            "then random 70/15/15 split) has been removed to prevent data leakage."
        )

    from research.phase6_dataset import GroupedBatchSampler
    train_sampler = GroupedBatchSampler(train_ds.group_ids, shuffle=True, seed=args.seed)
    val_sampler = GroupedBatchSampler(val_ds.group_ids, shuffle=False)
    test_sampler = GroupedBatchSampler(test_ds.group_ids, shuffle=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler)

    # ─── Build model ───
    config = create_ablation_variant(args.variant)
    input_dim = train_ds.features.shape[1]
    # Adjust config dims to match actual input
    # The variant mask may have reduced dimensions
    actual_dims = _count_variant_dims(args.variant)
    config = Phase6ModelConfig(
        self_dim=actual_dims['self'],
        neighbor_dim=actual_dims['neighbor'],
        overlap_dim=actual_dims['overlap'],
        selected_dim=actual_dims['selected'],
        use_neighbor=config.use_neighbor,
        use_overlap=config.use_overlap,
        use_selected=config.use_selected,
    )

    p4_ckpt = None
    if args.architecture == "residual":
        p4_ckpt = os.path.join(repo_root, "results", "learned_utility", "checkpoints", f"two_head_mlp_seed_{args.seed}.pt")
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

        model = ResidualContextModel(config, p4_model=p4_net).to(device)
        trainable_params = model.context_parameters()
        assert all(not p.requires_grad for p in model.p4_model.parameters()), "P4 backbone parameters must be strictly frozen!"
        p4_before = {k: v.clone().cpu() for k, v in model.p4_model.state_dict().items()}
    else:
        model = ContextAwareTwoHeadMLP(config).to(device)
        trainable_params = list(model.parameters())

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n[Model] Architecture: {args.architecture} ({config.variant_name})")
    print(f"  Input dim:          {input_dim}")
    print(f"  Total parameters:   {n_params:,}")
    print(f"  Trainable params:   {n_trainable:,} (P4 backbone strictly frozen)")

    # ─── Training ───
    if args.architecture == "residual":
        train_lr = args.lr if args.lr != 1e-3 else 2e-4
        loss_fn = Phase6Loss(
            lambda_q=0.0,
            lambda_c=0.0,
            lambda_r=args.lambda_r if args.lambda_r != 0.1 else 1.0,
            lambda_list=args.lambda_list,
            lambda_res=1.0,
            lambda_zero=0.5,
            scale_u=1.0,
        )
    else:
        train_lr = args.lr
        loss_fn = Phase6Loss(
            lambda_q=args.lambda_q,
            lambda_c=args.lambda_c,
            lambda_r=args.lambda_r,
            lambda_list=args.lambda_list,
            lambda_res=0.0,
            lambda_zero=0.0,
        )

    optimizer = torch.optim.Adam(trainable_params, lr=train_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    print(f"\n[Train] Starting training...")
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device) if len(val_ds) > 0 else train_metrics

        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_loss_q': train_metrics['loss_q'],
            'train_loss_t': train_metrics['loss_t'],
            'val_loss': val_metrics['loss'],
            'val_spearman_u': val_metrics.get('spearman_utility', 0.0),
            'val_spearman_q': val_metrics.get('spearman_quality', 0.0),
            'lr': current_lr,
        })

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1 or patience_counter == 0:
            print(
                f"  Epoch {epoch:4d} | "
                f"Train L={train_metrics['loss']:.6f} | "
                f"Val L={val_metrics['loss']:.6f} | "
                f"ρ_U={val_metrics.get('spearman_utility', 0.0):.3f} | "
                f"ρ_Q={val_metrics.get('spearman_quality', 0.0):.3f} | "
                f"LR={current_lr:.1e} | "
                f"{'★' if patience_counter == 0 else ''}"
            )

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    elapsed = time.perf_counter() - t_start
    print(f"\n[Train] Finished in {elapsed:.1f}s")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Verify P4 backbone invariance (P0.2)
    if args.architecture == "residual":
        for k, v in model.p4_model.state_dict().items():
            assert torch.equal(v.cpu(), p4_before[k]), f"P4 backbone weights changed for {k}!"
        print("  [Verification] P4 backbone weights remained strictly invariant throughout training.")

    # ─── Final evaluation ───
    print(f"\n[Eval] Final evaluation (best epoch {best_epoch})...")
    if len(test_ds) > 0:
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        print(f"  Test Loss:       {test_metrics['loss']:.6f}")
        print(f"  Test ρ(U):       {test_metrics['spearman_utility']:.4f}")
        print(f"  Test ρ(ΔQ):      {test_metrics['spearman_quality']:.4f}")
        print(f"  Test p-value(U): {test_metrics['p_value_utility']:.4e}")
    else:
        test_metrics = {}

    if len(val_ds) > 0:
        val_final = evaluate(model, val_loader, loss_fn, device)
        print(f"  Val ρ(U):        {val_final['spearman_utility']:.4f}")
    else:
        val_final = {}

    # ─── Save checkpoint ───
    config_dict = {
        'self_dim': config.self_dim,
        'neighbor_dim': config.neighbor_dim,
        'overlap_dim': config.overlap_dim,
        'selected_dim': config.selected_dim,
        'self_hidden': config.self_hidden,
        'neighbor_hidden': config.neighbor_hidden,
        'overlap_hidden': config.overlap_hidden,
        'selected_hidden': config.selected_hidden,
        'fusion_hidden': config.fusion_hidden,
        'head_hidden': config.head_hidden,
        'dropout': config.dropout,
        'eps_cost': config.eps_cost,
        'use_neighbor': config.use_neighbor,
        'use_overlap': config.use_overlap,
        'use_selected': config.use_selected,
    }

    ckpt_path = os.path.join(ckpt_dir, f"context_mlp_{args.variant}_seed_{args.seed}.pt")
    torch.save({
        'schema_version': 'phase6-v2',
        'architecture': 'residual_context' if args.architecture == 'residual' else 'direct_context',
        'variant': args.variant,
        'seed': args.seed,
        'protocol_version': 'v1',
        'model_state': model.state_dict(),
        'config': config_dict,
        'normalizer_path': f"normalization_{args.variant}.json",
        'p4_checkpoint': p4_ckpt if args.architecture == 'residual' else None,
        'p4_frozen': True if args.architecture == 'residual' else False,
        'git_commit': _get_git_commit(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'n_params': n_params,
        'test_metrics': test_metrics,
        'val_metrics': val_final,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'lambda_q': args.lambda_q,
            'lambda_c': args.lambda_c,
            'lambda_r': args.lambda_r,
            'patience': args.patience,
        },
        'metadata': {
            'schema_version': 'phase6-v2',
            'phase': 'phase6',
            'model_type': 'residual_context' if args.architecture == 'residual' else 'direct_context',
            'phase4_checkpoint': p4_ckpt if args.architecture == 'residual' else None,
            'phase4_frozen': True if args.architecture == 'residual' else False,
            'seed': args.seed,
            'variant': args.variant,
            'feature_variant': args.variant,
            'protocol': 'v1',
            'normalizer': f"normalization_{args.variant}.json",
            'git_commit': _get_git_commit(),
            'input_dim': input_dim,
            'dataset_paths': [str(p) for p in dataset_paths],
        },
    }, ckpt_path)
    print(f"\n[Save] Checkpoint: {ckpt_path}")

    # Save training log
    log_path = os.path.join(output_dir, f"training_log_{args.variant}_seed_{args.seed}.json")
    with open(log_path, 'w') as f:
        json.dump({
            'variant': args.variant,
            'seed': args.seed,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'n_params': n_params,
            'elapsed_s': elapsed,
            'history': history,
        }, f, indent=2)
    print(f"[Save] Training log: {log_path}")


def _get_git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _count_variant_dims(variant: str) -> Dict[str, int]:
    """Count actual feature dimensions per group for a variant."""
    dims = {'self': 11, 'neighbor': 0, 'overlap': 0, 'selected': 0}
    if variant in ('V9', 'V10', 'V11'):
        dims['neighbor'] = 8
    if variant in ('V10', 'V11'):
        dims['overlap'] = 5
    if variant == 'V11':
        dims['selected'] = 8
    return dims


if __name__ == "__main__":
    main()
