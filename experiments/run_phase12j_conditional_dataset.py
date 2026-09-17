#!/usr/bin/env python3
"""
Experiment runner for Phase 12-J: Conditional Counterfactual Dataset generation.
Generates or summarizes dataset for conditional utility modeling.
"""
import os
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from research.phase12_protocol import PHASE12_OUTPUT_DIR, SEEDS, TRAIN_SCENE, ZERO_SHOT_SCENES
except ImportError:
    PHASE12_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase12_paper_evidence")
    SEEDS = [42, 43, 44, 45, 46]
    TRAIN_SCENE = "train_scene"
    ZERO_SHOT_SCENES = ["test_scene1", "test_scene2"]

try:
    from research.conditional_dataset import ConditionalDatasetGenerator, ConditionalDatasetConfig
except ImportError:
    ConditionalDatasetGenerator = None
    ConditionalDatasetConfig = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase12J-Runner")

DEFAULT_OUT_DIR = os.path.join(PHASE12_OUTPUT_DIR, "conditional_dataset")

def generate_synthetic_data(args) -> List[Dict[str, Any]]:
    """Generate realistic synthetic data for testing the dataset schema."""
    logger.info(f"Generating synthetic conditional dataset for {args.n_frames} frames, {args.n_candidates} cands")
    
    np.random.seed(args.seed)
    
    dataset = []
    
    for frame_idx in range(args.n_frames):
        # Generate some synthetic candidate features
        s_i_features = np.random.uniform(0.1, 5.0, size=(args.n_candidates, 11))
        
        # Determine inherent delta_q and cost based on features
        # e.g., feature 0 (rgb_error) and 1 (gradient_norm) predict delta_q
        delta_q_single = 0.5 * s_i_features[:, 0] + 0.3 * s_i_features[:, 1] + np.random.normal(0, 0.1, args.n_candidates)
        cost_single = np.random.uniform(0.5, 3.0, args.n_candidates)
        utility_single = delta_q_single / cost_single
        
        for cand_idx in range(args.n_candidates):
            # For each candidate, generate contexts of various sizes
            for context_size in [0, 1, 2, 4, 8]:
                if context_size == 0:
                    context_indices = []
                    context_type = "empty"
                    h_S = np.zeros(51).tolist()
                    iou = 0.0
                    depth_conflict = 0.0
                    attr_red = 0.0
                else:
                    context_indices = np.random.choice(
                        [i for i in range(args.n_candidates) if i != cand_idx], 
                        size=min(context_size, args.n_candidates - 1), 
                        replace=False
                    ).tolist()
                    context_type = "random"
                    h_S = np.random.uniform(0, 10, 51).tolist()
                    # Synthetic overlaps
                    iou = np.random.beta(2, 5) * (context_size / 8.0) 
                    depth_conflict = np.random.uniform(0, 0.5)
                    attr_red = iou * np.random.uniform(0.8, 1.2)
                
                # Model conditional effects: redundancy decreases delta_q
                redundancy_factor = max(0.0, 1.0 - iou * 1.5)
                delta_q_cond = delta_q_single[cand_idx] * redundancy_factor
                
                # Sign flips
                if iou > 0.3 and np.random.rand() < 0.15:
                    delta_q_cond = -0.1 * delta_q_single[cand_idx]  # destructive interference
                
                delta_c_cond = cost_single[cand_idx] * np.random.uniform(0.9, 1.1)
                utility_cond = delta_q_cond / delta_c_cond if delta_c_cond > 0 else 0
                
                record = {
                    'scene': args.scene,
                    'frame': frame_idx,
                    'split': args.split,
                    'seed': args.seed,
                    'candidate_idx': cand_idx,
                    'context_indices': context_indices,
                    'context_type': context_type,
                    'context_size': len(context_indices),
                    's_i': s_i_features[cand_idx].tolist(),
                    'h_S': h_S,
                    'delta_q': float(delta_q_cond),
                    'delta_c_ms': float(delta_c_cond),
                    'utility': float(utility_cond),
                    'delta_q_single': float(delta_q_single[cand_idx]),
                    'utility_single': float(utility_single[cand_idx]),
                    'interaction_descriptors': {
                        'mean_iou': float(iou),
                        'max_iou': float(iou * 1.2),
                        'mean_depth_conflict': float(depth_conflict),
                        'attribution_redundancy': float(attr_red),
                    },
                    'q_baseline_psnr': float(30.0 + np.random.uniform(-2, 2)),
                    'q_after_psnr': float(30.0 + delta_q_cond + np.random.uniform(-2, 2))
                }
                dataset.append(record)
                
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], out_dir: str, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = os.path.join(out_dir, f"conditional_dataset_seed_{seed}.json")
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved JSON to {json_path}")
    
    csv_path = os.path.join(out_dir, f"conditional_dataset_seed_{seed}.csv")
    if dataset:
        keys = list(dataset[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in dataset:
                row_copy = dict(row)
                row_copy['s_i'] = json.dumps(row_copy['s_i'])
                row_copy['h_S'] = json.dumps(row_copy['h_S'])
                row_copy['context_indices'] = json.dumps(row_copy['context_indices'])
                row_copy['interaction_descriptors'] = json.dumps(row_copy['interaction_descriptors'])
                writer.writerow(row_copy)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {
        "dataset_size": len(dataset),
        "seed": seed,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")

def summarize_dataset(data_dir: str):
    import glob
    json_files = glob.glob(os.path.join(data_dir, "conditional_dataset_seed_*.json"))
    if not json_files:
        logger.error(f"No dataset JSON files found in {data_dir}")
        return
        
    dataset = []
    for jf in json_files:
        with open(jf, 'r') as f:
            dataset.extend(json.load(f))
            
    if not dataset:
        logger.error("Dataset is empty")
        return
        
    logger.info(f"Loaded {len(dataset)} records from {data_dir}")
    
    # Compute stats
    scenes = set(r['scene'] for r in dataset)
    frames = set(f"{r['scene']}_{r['frame']}" for r in dataset)
    
    summary = {
        "total_records": len(dataset),
        "unique_scenes": list(scenes),
        "total_frames": len(frames),
        "context_size_stats": {},
    }
    
    logger.info("=== Dataset Summary ===")
    logger.info(f"Total records: {len(dataset)}")
    logger.info(f"Unique scenes: {len(scenes)} ({list(scenes)})")
    logger.info(f"Total frames: {len(frames)}")
    
    context_sizes = set(r['context_size'] for r in dataset)
    
    for c_size in sorted(list(context_sizes)):
        subset = [r for r in dataset if r['context_size'] == c_size]
        avg_dq = float(np.mean([r['delta_q'] for r in subset]))
        avg_dc = float(np.mean([r['delta_c_ms'] for r in subset]))
        avg_u = float(np.mean([r['utility'] for r in subset]))
        logger.info(f"Context size {c_size}: {len(subset)} records")
        logger.info(f"  Mean Delta Q: {avg_dq:.4f}, Mean Delta C: {avg_dc:.4f}, Mean Utility: {avg_u:.4f}")
        
        summary["context_size_stats"][c_size] = {
            "count": len(subset),
            "mean_delta_q": avg_dq,
            "mean_delta_c": avg_dc,
            "mean_utility": avg_u
        }
        
    # Sign flip rate
    flips = [1 for r in dataset if r['delta_q'] < 0 and r['delta_q_single'] > 0]
    flip_rate = float(len(flips) / len(dataset) * 100)
    logger.info(f"Sign flip rate: {flip_rate:.2f}%")
    summary["sign_flip_rate_pct"] = flip_rate
    
    # Simple correlation between utility and a feature
    utilities = [r['utility'] for r in dataset]
    if len(dataset) > 1:
        s_i_0 = [r['s_i'][0] for r in dataset]
        corr = float(np.corrcoef(utilities, s_i_0)[0, 1])
        logger.info(f"Correlation between s_i[0] and utility: {corr:.4f}")
        summary["utility_si0_correlation"] = corr
        
    summary_path = os.path.join(data_dir, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

def run_live(args):
    if ConditionalDatasetGenerator is None:
        logger.error("research.conditional_dataset not found. Run with --generate-synthetic")
        sys.exit(1)
        
    logger.info("Initializing Live Mode for Conditional Dataset Generation")
    
    # Minimal mock implementation for the pipeline if the real one isn't imported
    # Real implementations typically fetch from project registry or utils
    class MockPipeline:
        pass
        
    pipeline = MockPipeline()
    
    config = ConditionalDatasetConfig(
        max_context_size=8,
        context_types=["empty", "random", "knn", "cluster_center"]
    )
    generator = ConditionalDatasetGenerator(pipeline, config)
    
    full_dataset = []
    
    for frame_idx in range(args.n_frames):
        logger.info(f"Processing frame {frame_idx}")
        candidates = list(range(args.n_candidates))
        
        # This calls the real method on the generator if properly initialized
        # We wrap in try-except in case the mock pipeline causes issues
        try:
            frame_data = generator.generate_frame_dataset(candidates, frame_idx)
            full_dataset.extend(frame_data)
        except Exception as e:
            logger.error(f"Error calling generate_frame_dataset: {e}")
            logger.warning("Pipeline is likely a mock or not properly initialized.")
            break
            
    return full_dataset

def main():
    parser = argparse.ArgumentParser(description="Phase 12-J Conditional Dataset Runner")
    parser.add_argument("--scene", type=str, default=TRAIN_SCENE, help="Scene name")
    parser.add_argument("--seed", type=int, default=SEEDS[0], help="Random seed")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames")
    parser.add_argument("--n-candidates", type=int, default=20, help="Number of candidates per frame")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--summarize", action="store_true", help="Summarize dataset")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_OUT_DIR, help="Data directory for summarize")
    
    args = parser.parse_args()
    
    if args.summarize:
        summarize_dataset(args.data_dir)
        return
        
    if args.generate_synthetic:
        dataset = generate_synthetic_data(args)
    else:
        dataset = run_live(args)
        
    if dataset:
        save_dataset(dataset, args.output_dir, args.seed)
        
        # Also generate summary on fresh data
        summarize_dataset(args.output_dir)

if __name__ == "__main__":
    main()
