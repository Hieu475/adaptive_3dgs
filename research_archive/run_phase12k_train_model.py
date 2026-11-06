#!/usr/bin/env python3
"""
Phase 12-K: Train Interaction-Aware Conditional Utility Model.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, ndcg_score

# Add parent directory to sys.path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.interaction_model import (
    InteractionAwareModel,
    InteractionAwareLoss,
    InteractionModelTrainer,
    ConditionalUtilityDataset
)

# Optional imports if available, otherwise fallback
try:
    from research.phase12_protocol import PHASE12_OUTPUT_DIR, SEEDS
except ImportError:
    PHASE12_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase12_paper_evidence")
    SEEDS = [42]


def generate_synthetic_training_data(n_samples=1500):
    """Generate synthetic data matching Phase 12-J schema."""
    records = []
    for i in range(n_samples):
        s_i = np.random.randn(11).tolist()
        h_S = np.random.randn(51).tolist()
        noise_q = np.random.randn() * 0.05
        noise_c = np.random.randn() * 0.05
        
        # Ground truth utility depends on features
        true_q = 0.5 * s_i[0] + 0.3 * s_i[1] - 0.2 * h_S[0] + noise_q
        true_c = abs(1.0 + 0.1 * s_i[2] + noise_c)
        true_u = true_q / (true_c + 1e-6)
        
        # Context structure simulating 12-J
        records.append({
            's_i': s_i,
            'h_S': h_S,
            'delta_q': true_q,
            'delta_c_ms': true_c,
            'utility': true_u
        })
    return records


def evaluate_model(model, dataloader, device):
    """Evaluate model on the dataset and return metrics."""
    model.eval()
    
    all_pred_q = []
    all_pred_c = []
    all_pred_u = []
    
    all_true_q = []
    all_true_c = []
    all_true_u = []
    
    with torch.no_grad():
        for batch in dataloader:
            s_i = batch['s_i'].to(device)
            h_S = batch['h_S'].to(device)
            
            true_q = batch['delta_q'].cpu().numpy()
            true_c = batch['delta_c_ms'].cpu().numpy()
            true_u = batch['utility'].cpu().numpy()
            
            # Predict
            pred_q, pred_c, pred_u = model(s_i, h_S)
            
            all_pred_q.extend(pred_q.cpu().numpy())
            all_pred_c.extend(pred_c.cpu().numpy())
            all_pred_u.extend(pred_u.cpu().numpy())
            
            all_true_q.extend(true_q)
            all_true_c.extend(true_c)
            all_true_u.extend(true_u)
            
    all_pred_q = np.array(all_pred_q).flatten()
    all_pred_c = np.array(all_pred_c).flatten()
    all_pred_u = np.array(all_pred_u).flatten()
    
    all_true_q = np.array(all_true_q).flatten()
    all_true_c = np.array(all_true_c).flatten()
    all_true_u = np.array(all_true_u).flatten()
    
    # MSE
    mse_q = np.mean((all_pred_q - all_true_q)**2)
    mse_c = np.mean((all_pred_c - all_true_c)**2)
    
    # Spearman rho on utility
    rho, _ = spearmanr(all_pred_u, all_true_u)
    
    # AUROC for P(U > 0)
    true_positive_u = (all_true_u > 0).astype(int)
    # Check if there's both classes
    if len(np.unique(true_positive_u)) > 1:
        auroc = roc_auc_score(true_positive_u, all_pred_u)
    else:
        auroc = float('nan')
        
    # NDCG
    # Reshape for NDCG: assuming all instances as one single query for simplicity,
    # or pad/reshape if there were natural query groups. We'll treat the whole
    # eval set as one query for ranking evaluation.
    true_rel = all_true_u.reshape(1, -1)
    # Shift to non-negative for NDCG 
    min_rel = np.min(true_rel)
    if min_rel < 0:
        true_rel = true_rel - min_rel
        
    pred_rel = all_pred_u.reshape(1, -1)
    
    k10 = min(10, all_true_u.shape[0])
    k20 = min(20, all_true_u.shape[0])
    
    ndcg_10 = ndcg_score(true_rel, pred_rel, k=k10) if k10 > 1 else float('nan')
    ndcg_20 = ndcg_score(true_rel, pred_rel, k=k20) if k20 > 1 else float('nan')
    
    metrics = {
        'mse_delta_q': float(mse_q),
        'mse_delta_c': float(mse_c),
        'spearman_rho': float(rho),
        'auroc': float(auroc),
        'ndcg_10': float(ndcg_10),
        'ndcg_20': float(ndcg_20)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 12-K: Train Interaction-Aware Model")
    parser.add_argument("--data-dir", type=str, default="results/phase12_paper_evidence/conditional_dataset",
                        help="Directory containing the dataset from Phase 12-J")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="results/phase12_paper_evidence/conditional_model",
                        help="Output directory")
    parser.add_argument("--generate-synthetic", action="store_true", 
                        help="Generate synthetic data instead of loading")
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    records = []
    if args.generate_synthetic:
        print("Generating synthetic dataset...")
        records = generate_synthetic_training_data(n_samples=1500)
    else:
        dataset_path = os.path.join(args.data_dir, f"conditional_dataset_seed_{args.seed}.json")
        print(f"Loading dataset from {dataset_path}")
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset {dataset_path} not found.")
            sys.exit(1)
            
        with open(dataset_path, 'r') as f:
            records = json.load(f)
            
    # Train/Val split
    train_records, val_records = train_test_split(records, test_size=0.2, random_state=args.seed)
    print(f"Split data: {len(train_records)} train, {len(val_records)} val")
    
    # Create datasets
    train_dataset = ConditionalUtilityDataset(train_records)
    val_dataset = ConditionalUtilityDataset(val_records)
    
    # Initialize model
    s_i_dim = len(records[0]['s_i'])
    h_S_dim = len(records[0]['h_S'])
    
    model = InteractionAwareModel(local_dim=s_i_dim, context_dim=h_S_dim)
    
    trainer_config = {
        'device': str(device),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
    }
    
    trainer = InteractionModelTrainer(model, train_dataset, val_dataset, trainer_config)
    
    # Train
    print(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(n_epochs=args.epochs)
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate(trainer.val_loader)
    print("Evaluation metrics:", metrics)
    
    # Save outputs
    checkpoint_path = os.path.join(args.output_dir, f"interaction_model_seed_{args.seed}.pt")
    log_path = os.path.join(args.output_dir, f"training_log_seed_{args.seed}.json")
    report_path = os.path.join(args.output_dir, f"model_evaluation_seed_{args.seed}.json")
    
    torch.save(trainer.model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")
    
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # Convert numpy types for JSON serialization
    metrics_json = {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()}
    
    with open(report_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()
