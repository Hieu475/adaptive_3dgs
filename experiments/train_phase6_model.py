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
    model: ContextAwareTwoHeadMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Phase6Loss,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict with 'loss', 'loss_q', 'loss_t', 'loss_r' averages.
    """
    model.train()
    total_loss = 0.0
    total_lq = 0.0
    total_lt = 0.0
    total_lr = 0.0
    n_batches = 0

    for batch in loader:
        x = batch['features'].to(device)
        tgt_q = batch['delta_q'].to(device)
        tgt_t = batch['delta_t'].to(device)
        tgt_u = batch['utility'].to(device)

        optimizer.zero_grad()
        pred_q, pred_t, pred_u = model(x)

        losses = loss_fn(pred_q, pred_t, pred_u, tgt_q, tgt_t, tgt_u)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_lq += losses['loss_q'].item()
        total_lt += losses['loss_t'].item()
        total_lr += losses['loss_r'].item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_q': total_lq / max(n_batches, 1),
        'loss_t': total_lt / max(n_batches, 1),
        'loss_r': total_lr / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: ContextAwareTwoHeadMLP,
    loader: DataLoader,
    loss_fn: Phase6Loss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

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

    for batch in loader:
        x = batch['features'].to(device)
        tgt_q = batch['delta_q'].to(device)
        tgt_t = batch['delta_t'].to(device)
        tgt_u = batch['utility'].to(device)

        pred_q, pred_t, pred_u = model(x)

        losses = loss_fn(pred_q, pred_t, pred_u, tgt_q, tgt_t, tgt_u)
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
        print("[WARN] No training data! Using all data for training (prototype mode).")
        # In prototype mode (cross_scene_test only), use all data for training
        all_feats, all_dq, all_dt, all_u, all_raw = load_phase6_dataset(dataset_paths[0])
        normalizer = Phase6FeatureNormalizer()
        normalizer.fit(all_feats)
        all_feats_norm = normalizer.transform(all_feats)
        normalizer.save_json(normalizer_path)

        from research.phase6_dataset import _get_variant_mask
        mask = _get_variant_mask(args.variant)
        all_feats_norm = all_feats_norm[:, mask]

        # 70/15/15 random split
        N = len(all_feats_norm)
        perm = np.random.permutation(N)
        n_train = max(1, int(0.7 * N))
        n_val = max(1, int(0.15 * N))

        train_ds = Phase6UtilityDataset(
            all_feats_norm[perm[:n_train]],
            all_dq[perm[:n_train]], all_dt[perm[:n_train]], all_u[perm[:n_train]],
        )
        val_ds = Phase6UtilityDataset(
            all_feats_norm[perm[n_train:n_train+n_val]],
            all_dq[perm[n_train:n_train+n_val]],
            all_dt[perm[n_train:n_train+n_val]],
            all_u[perm[n_train:n_train+n_val]],
        )
        test_ds = Phase6UtilityDataset(
            all_feats_norm[perm[n_train+n_val:]],
            all_dq[perm[n_train+n_val:]],
            all_dt[perm[n_train+n_val:]],
            all_u[perm[n_train+n_val:]],
        )
        print(f"  [Prototype split] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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

    model = ContextAwareTwoHeadMLP(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] {config.variant_name}")
    print(f"  Input dim:  {input_dim}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Config:     neighbor={config.use_neighbor}, overlap={config.use_overlap}, selected={config.use_selected}")

    # ─── Training ───
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    loss_fn = Phase6Loss(
        lambda_q=args.lambda_q,
        lambda_c=args.lambda_c,
        lambda_r=args.lambda_r,
    )

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    print(f"\n[Train] Starting training...")
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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
        'model_state': model.state_dict(),
        'config': config_dict,
        'seed': args.seed,
        'variant': args.variant,
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
            'phase': 'phase6',
            'variant': args.variant,
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
