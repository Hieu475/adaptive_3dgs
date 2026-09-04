#!/usr/bin/env python3
"""Phase 4: Train Learned Utility Models across Multiple Seeds.

Implements Step 8 of Phase 4:
  - Canonical dataset parsing via research.utility_dataset.
  - Normalization parameters fit strictly on train split only.
  - Trains TwoHeadMLP (B7), TwoHeadLinear (B6), and LinearUtilityModel (B5).
  - Multi-seed execution across protocol seeds [42, 43, 44, 45, 46].
  - Model selection tracked on validation split.
  - Serializes checkpoints to results/learned_utility/checkpoints/.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.protocol import (
    load_protocol,
    get_seeds,
    get_repo_root,
)
from research.utility_dataset import (
    load_canonical_oracle_dataset,
    prepare_normalized_splits,
)
from research.utility_losses import LossConfig
from research.utility_models import (
    TwoHeadMLP,
    TwoHeadLinear,
    LinearUtilityModel,
)
from research.utility_training import (
    UtilityModelTrainer,
    TrainingConfig,
)
from research.utility_metrics import safe_spearmanr


def main():
    parser = argparse.ArgumentParser(description="Train Phase 4 Learned Utility Estimator.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for MLP backbone.")
    parser.add_argument("--lambda-rank", type=float, default=1.0, help="Weight for pairwise ranking loss.")
    parser.add_argument("--lambda-q", type=float, default=0.25, help="Weight for quality head loss.")
    parser.add_argument("--lambda-t", type=float, default=0.125, help="Weight for cost head loss.")
    args = parser.parse_args()

    repo_root = get_repo_root()
    protocol = load_protocol()
    seeds = get_seeds(protocol)

    print("=" * 80)
    print("   PHASE 4: TRAIN LEARNED UTILITY ESTIMATOR ACROSS PROTOCOL SEEDS")
    print("=" * 80)
    print(f">> Seeds: {seeds}")
    print(f">> Epochs: {args.epochs} | LR: {args.lr} | Hidden: {args.hidden_dim}")
    print(f">> Loss Config: lambda_rank={args.lambda_rank}, lambda_q={args.lambda_q}, lambda_t={args.lambda_t}\n")

    # 1. Load canonical dataset and prepare train-normalized splits
    raw_dataset = load_canonical_oracle_dataset()
    manifest_path = os.path.join(repo_root, "results", "learned_utility", "dataset_manifest.json")
    raw_dataset.save_manifest(manifest_path)
    print(f">> Dataset manifest saved to: {manifest_path}")

    stats_file_legacy = os.path.join(repo_root, "results", "statistics", "normalization.json")
    stats_file_canonical = os.path.join(repo_root, "results", "learned_utility", "normalization.json")

    train_ds, val_ds, test_ds, normalizer = prepare_normalized_splits(
        dataset=raw_dataset,
        save_stats_path=stats_file_canonical,
    )
    normalizer.save_json(stats_file_legacy)

    print(f">> Dataset Splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f">> Features: {len(train_ds.feature_names)} canonical factors: {train_ds.feature_names}")
    print(f">> Normalization saved to: {stats_file_canonical}\n")

    loss_cfg = LossConfig(
        lambda_rank=args.lambda_rank,
        lambda_q=args.lambda_q,
        lambda_t=args.lambda_t,
    )
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_config=loss_cfg,
    )
    trainer = UtilityModelTrainer(config=config)

    ckpt_dir = os.path.join(repo_root, "results", "learned_utility", "checkpoints")
    models_dir = os.path.join(repo_root, "results", "learned_utility", "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    summary = {
        "seeds": seeds,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "feature_names": train_ds.feature_names,
        "loss_config": {
            "lambda_rank": args.lambda_rank,
            "lambda_q": args.lambda_q,
            "lambda_t": args.lambda_t,
        },
        "seed_results": {},
    }

    val_rhos_mlp = []
    val_rhos_linear = []
    val_rhos_direct = []

    for seed in seeds:
        print(f"--- Training on Seed {seed} ---")
        
        # B7: Two-Head MLP
        mlp_model = TwoHeadMLP(in_features=len(train_ds.feature_names), hidden_dim=args.hidden_dim)
        res_mlp = trainer.train_two_head_model(mlp_model, train_ds, val_ds=val_ds, seed=seed)
        
        with torch.no_grad():
            mlp_model.eval()
            _, _, p_val_mlp = mlp_model(val_ds.X)
            rho_mlp, _ = safe_spearmanr(p_val_mlp.cpu().numpy(), val_ds.utility_np)
        val_rhos_mlp.append(rho_mlp)

        mlp_ckpt_path = os.path.join(ckpt_dir, f"two_head_mlp_seed_{seed}.pt")
        model_canonical_path = os.path.join(models_dir, f"seed_{seed}.pt")
        trainer.save_checkpoint(
            mlp_model,
            mlp_ckpt_path,
            feature_names=train_ds.feature_names,
            metadata={"seed": seed, "val_spearman_rho": rho_mlp, "model_type": "TwoHeadMLP"},
        )
        trainer.save_checkpoint(
            mlp_model,
            model_canonical_path,
            feature_names=train_ds.feature_names,
            metadata={"seed": seed, "val_spearman_rho": rho_mlp, "model_type": "TwoHeadMLP"},
        )

        # B6: Two-Head Linear
        lin_model = TwoHeadLinear(in_features=len(train_ds.feature_names))
        res_lin = trainer.train_two_head_model(lin_model, train_ds, val_ds=val_ds, seed=seed)
        with torch.no_grad():
            lin_model.eval()
            _, _, p_val_lin = lin_model(val_ds.X)
            rho_lin, _ = safe_spearmanr(p_val_lin.cpu().numpy(), val_ds.utility_np)
        val_rhos_linear.append(rho_lin)

        lin_ckpt_path = os.path.join(ckpt_dir, f"two_head_linear_seed_{seed}.pt")
        trainer.save_checkpoint(
            lin_model,
            lin_ckpt_path,
            feature_names=train_ds.feature_names,
            metadata={"seed": seed, "val_spearman_rho": rho_lin, "model_type": "TwoHeadLinear"},
        )

        # B5: Direct Linear
        direct_model = LinearUtilityModel(in_features=len(train_ds.feature_names))
        res_dir = trainer.train_linear_utility_model(direct_model, train_ds, seed=seed)
        with torch.no_grad():
            direct_model.eval()
            p_val_dir = direct_model(val_ds.X)
            rho_dir, _ = safe_spearmanr(p_val_dir.cpu().numpy(), val_ds.utility_np)
        val_rhos_direct.append(rho_dir)

        direct_ckpt_path = os.path.join(ckpt_dir, f"linear_direct_seed_{seed}.pt")
        trainer.save_checkpoint(
            direct_model,
            direct_ckpt_path,
            feature_names=train_ds.feature_names,
            metadata={"seed": seed, "val_spearman_rho": rho_dir, "model_type": "LinearUtilityModel"},
        )

        print(f"   [Seed {seed}] Val rho: TwoHeadMLP = {rho_mlp:+.4f} | TwoHeadLinear = {rho_lin:+.4f} | DirectLinear = {rho_dir:+.4f}")

        summary["seed_results"][str(seed)] = {
            "two_head_mlp": {"val_rho": float(rho_mlp), "checkpoint": mlp_ckpt_path},
            "two_head_linear": {"val_rho": float(rho_lin), "checkpoint": lin_ckpt_path},
            "direct_linear": {"val_rho": float(rho_dir), "checkpoint": direct_ckpt_path},
        }

    summary["validation_aggregate"] = {
        "two_head_mlp_rho_mean": float(np.mean(val_rhos_mlp)),
        "two_head_mlp_rho_std": float(np.std(val_rhos_mlp)),
        "two_head_linear_rho_mean": float(np.mean(val_rhos_linear)),
        "two_head_linear_rho_std": float(np.std(val_rhos_linear)),
        "direct_linear_rho_mean": float(np.mean(val_rhos_direct)),
        "direct_linear_rho_std": float(np.std(val_rhos_direct)),
    }

    summary_file = os.path.join(repo_root, "results", "learned_utility", "training_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(">> Training Complete across 5 seeds:")
    print(f"   - TwoHeadMLP:     Val rho = {np.mean(val_rhos_mlp):+.4f} +/- {np.std(val_rhos_mlp):.4f}")
    print(f"   - TwoHeadLinear:  Val rho = {np.mean(val_rhos_linear):+.4f} +/- {np.std(val_rhos_linear):.4f}")
    print(f"   - DirectLinear:   Val rho = {np.mean(val_rhos_direct):+.4f} +/- {np.std(val_rhos_direct):.4f}")
    print(f">> Checkpoints saved to: {ckpt_dir}")
    print(f">> Training summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
