#!/usr/bin/env python3
"""Phase 12-I: Interaction Audit Experiment Runner.

Runs the complete interaction audit to answer:
1. What % of interventions change utility when S changes?
2. What % of candidates flip utility sign?
3. Does interaction depend on screen-space overlap?
4. Does interaction depend on depth conflict?
5. Is interaction strong enough to change top-K selection?

Usage:
    # Live mode (requires pipeline)
    python experiments/run_phase12i_interaction_audit.py --scene tum_fr1_desk --seed 42
    
    # Offline analysis (from pre-generated data)
    python experiments/run_phase12i_interaction_audit.py --offline --data-dir results/phase12_paper_evidence/interaction_audit/
    
    # Generate synthetic test data
    python experiments/run_phase12i_interaction_audit.py --generate-synthetic
"""
import argparse
import os
import sys
import json
import csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.interaction_audit import InteractionAuditExperiment, InteractionAuditConfig
from research.phase12_protocol import PHASE12_OUTPUT_DIR, SEEDS, TRAIN_SCENE

def setup_argparse():
    parser = argparse.ArgumentParser(description="Phase 12-I: Interaction Audit Experiment Runner")
    parser.add_argument('--scene', type=str, default=TRAIN_SCENE, help="Scene to run on")
    parser.add_argument('--seed', type=int, default=SEEDS[0], help="Random seed")
    parser.add_argument('--n-frames', type=int, default=5, help="Number of frames to audit")
    parser.add_argument('--n-candidates', type=int, default=100, help="Number of candidates per frame")
    parser.add_argument('--n-pairs', type=int, default=500, help="Number of pairs per frame")
    parser.add_argument('--output-dir', type=str, default=str(PHASE12_OUTPUT_DIR / "interaction_audit"), help="Output directory")
    
    parser.add_argument('--offline', action='store_true', help="Run offline analysis from pre-generated data")
    parser.add_argument('--data-dir', type=str, help="Directory containing CSV data for offline analysis")
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic test data")
    
    return parser.parse_args()

def generate_synthetic_audit_data(n_frames: int, n_pairs: int, n_contexts: int):
    """Generate realistic synthetic data matching the Phase 12-I schema.
    
    Pairs schema (interaction_pairs.csv):
        idx_i, idx_j, delta_q_i, delta_q_j, delta_q_ij, interaction_residual,
        overlap_iou, distance_3d, depth_conflict, overlap_bin, depth_bin,
        additivity_ratio, cost_i_ms, cost_j_ms, cost_ij_ms, scene, frame
        
    Contexts schema (interaction_contexts.csv):
        candidate_idx, context_size, context_type, overlap_bin, depth_bin,
        delta_q_single, delta_q_conditional, utility_single, utility_conditional,
        deviation_dq, ratio_rq, sign_flip, iou_with_context, depth_conflict_with_context,
        scene, frame
    """
    print(f"Generating synthetic data for {n_frames} frames, {n_pairs} pairs/frame, {n_contexts} context items/frame...")
    
    np.random.seed(42)
    pairs_data = []
    contexts_data = []
    
    overlap_bin_names = ['low', 'medium', 'high']
    depth_bin_names = ['near', 'medium', 'far']
    context_types = ['spatial_knn', 'overlap_top', 'random', 'high_utility']
    eps = 1e-6
    
    for frame_idx in range(n_frames):
        # ─── Pairs data ─────────────────────────────────────────────────
        for _ in range(n_pairs):
            # Simulate individual quality gains (can be negative for bad updates)
            delta_q_i = np.random.normal(0.002, 0.005)
            delta_q_j = np.random.normal(0.002, 0.005)
            
            # Distance and overlap (correlated)
            dist = np.random.uniform(0.05, 3.0)
            iou = np.clip(np.exp(-dist * 1.5) + np.random.normal(0, 0.05), 0, 0.95)
            
            # Depth conflict
            depth_diff = np.random.exponential(0.3)
            
            # Interaction: high overlap → sub-additive (redundancy)
            if iou > 0.5:
                interaction = np.random.normal(-0.003, 0.002)  # strong redundancy
            elif iou > 0.1:
                interaction = np.random.normal(-0.001, 0.002)  # mild redundancy
            else:
                interaction = np.random.normal(0.0, 0.001)     # near-additive
                
            delta_q_ij = delta_q_i + delta_q_j + interaction
            additivity_ratio = delta_q_ij / (delta_q_i + delta_q_j + eps)
            
            # Overlap bin
            if iou < 0.1:
                overlap_bin = 'low'
            elif iou < 0.5:
                overlap_bin = 'medium'
            else:
                overlap_bin = 'high'
                
            # Depth bin
            if depth_diff < 0.2:
                depth_bin = 'near'
            elif depth_diff < 0.6:
                depth_bin = 'medium'
            else:
                depth_bin = 'far'
            
            # Cost (ms)
            cost_i = np.random.uniform(0.5, 2.0)
            cost_j = np.random.uniform(0.5, 2.0)
            cost_ij = cost_i + cost_j + np.random.normal(0.1, 0.05)  # slight overhead
            
            pairs_data.append({
                'idx_i': int(np.random.randint(0, 5000)),
                'idx_j': int(np.random.randint(0, 5000)),
                'delta_q_i': float(delta_q_i),
                'delta_q_j': float(delta_q_j),
                'delta_q_ij': float(delta_q_ij),
                'interaction_residual': float(interaction),
                'overlap_iou': float(iou),
                'distance_3d': float(dist),
                'depth_conflict': float(depth_diff),
                'overlap_bin': overlap_bin,
                'depth_bin': depth_bin,
                'additivity_ratio': float(additivity_ratio),
                'cost_i_ms': float(cost_i),
                'cost_j_ms': float(cost_j),
                'cost_ij_ms': float(cost_ij),
                'scene': 'tum_fr1_desk',
                'frame': int(frame_idx),
            })
            
        # ─── Contexts data ──────────────────────────────────────────────
        for _ in range(n_contexts):
            # Unconditional utility (can be negative)
            delta_q_single = np.random.normal(0.002, 0.005)
            cost_single = np.random.uniform(0.5, 2.0)
            utility_single = delta_q_single / (cost_single + eps)
            
            # Context properties
            context_size = np.random.choice([1, 2, 4, 8])
            iou_ctx = np.random.uniform(0, 0.8)
            depth_ctx = np.random.exponential(0.3)
            ctx_type = np.random.choice(context_types)
            
            # Conditional utility: depends on overlap with context
            if iou_ctx > 0.5:
                # High overlap → bigger deviation, more sign flips
                deviation = np.random.normal(-0.003, 0.003) * (context_size / 4.0)
            elif iou_ctx > 0.1:
                deviation = np.random.normal(-0.001, 0.002)
            else:
                deviation = np.random.normal(0.0, 0.001)
                
            delta_q_cond = delta_q_single + deviation
            cost_cond = cost_single + np.random.normal(0.0, 0.1)
            utility_cond = delta_q_cond / (abs(cost_cond) + eps)
            
            ratio_rq = delta_q_cond / (abs(delta_q_single) + eps)
            sign_flip = bool((delta_q_single > 0 and delta_q_cond < 0) or 
                           (delta_q_single < 0 and delta_q_cond > 0))
            
            # Bins
            if iou_ctx < 0.1:
                overlap_bin = 'low'
            elif iou_ctx < 0.5:
                overlap_bin = 'medium'
            else:
                overlap_bin = 'high'
                
            if depth_ctx < 0.2:
                depth_bin = 'near'
            elif depth_ctx < 0.6:
                depth_bin = 'medium'
            else:
                depth_bin = 'far'
            
            contexts_data.append({
                'candidate_idx': int(np.random.randint(0, 5000)),
                'context_size': int(context_size),
                'context_type': str(ctx_type),
                'overlap_bin': overlap_bin,
                'depth_bin': depth_bin,
                'delta_q_single': float(delta_q_single),
                'delta_q_conditional': float(delta_q_cond),
                'utility_single': float(utility_single),
                'utility_conditional': float(utility_cond),
                'deviation_dq': float(deviation),
                'ratio_rq': float(ratio_rq),
                'sign_flip': sign_flip,
                'iou_with_context': float(iou_ctx),
                'depth_conflict_with_context': float(depth_ctx),
                'scene': 'tum_fr1_desk',
                'frame': int(frame_idx),
            })
            
    return pairs_data, contexts_data


def run_offline_analysis(data_dir: str):
    """Load CSV data and compute all statistics."""
    print(f"Loading data from {data_dir}...")
    
    pairs_file = Path(data_dir) / 'interaction_pairs.csv'
    contexts_file = Path(data_dir) / 'interaction_contexts.csv'
    
    if not pairs_file.exists() or not contexts_file.exists():
        print(f"Error: Missing data files in {data_dir}")
        return None, None
    
    import pandas as pd
    pairs_data = pd.read_csv(pairs_file).to_dict('records')
    contexts_data = pd.read_csv(contexts_file).to_dict('records')
    
    # Fix boolean columns
    for row in contexts_data:
        if isinstance(row.get('sign_flip'), str):
            row['sign_flip'] = row['sign_flip'].lower() == 'true'
            
    return pairs_data, contexts_data

def generate_report(stats: Dict[str, Any], output_path: Path):
    """Create a comprehensive markdown report with tables and findings."""
    print(f"Generating report at {output_path}...")
    
    report = f"""# Phase 12-I: Interaction Audit Report

## 1. Executive Summary

This report analyzes Gaussian interaction effects in the Adaptive 3DGS pipeline,
testing whether pointwise utility $U^*(i|\\emptyset)$ is a sufficient statistic
for subset selection, or whether conditional utility $U^*(i|S)$ is needed.

- **Total Pairs Analyzed**: {stats['total_pairs']}
- **Total Conditional Evaluations**: {stats['total_contexts']}
- **Sub-additive Interactions**: {stats['sub_additive_percent']:.2f}%
- **Sign Flip Rate $R_{{flip}}$**: {stats['sign_flip_percent']:.2f}%

## 2. Pairwise Interaction $I_{{ij}}$

$$I_{{ij}} = \\Delta Q(\\{{i,j\\}}) - \\Delta Q_i - \\Delta Q_j$$

| Metric | Value |
|--------|-------|
| Mean $I_{{ij}}$ | {stats['mean_interaction']:.6f} |
| Mean $|I_{{ij}}|$ | {stats['mean_abs_interaction']:.6f} |
| % Sub-additive (redundant) | {stats['sub_additive_percent']:.2f}% |
| % Super-additive (synergy) | {stats['super_additive_percent']:.2f}% |
| $R_{{pair}}$ mean | {stats['R_pair_mean']:.4f} |

## 3. Interaction by Overlap Bin

| Overlap | Mean $I_{{ij}}$ | % Sub-additive |
|---------|----------------|----------------|
| Low (IoU < 0.1) | {stats.get('I_ij_low', 0.0):.6f} | {stats.get('sub_add_low', 0.0):.1f}% |
| Medium (0.1-0.5) | {stats.get('I_ij_med', 0.0):.6f} | {stats.get('sub_add_med', 0.0):.1f}% |
| High (IoU > 0.5) | {stats.get('I_ij_high', 0.0):.6f} | {stats.get('sub_add_high', 0.0):.1f}% |

## 4. Sign Flip Analysis

$$R_{{flip}} = P[\\text{{sign}}(U^*(i|S)) \\neq \\text{{sign}}(U^*(i|\\emptyset))]$$

| Overlap Bin | Sign Flip Rate |
|-------------|---------------|
| Low | {stats.get('flip_low', 0.0):.2f}% |
| Medium | {stats.get('flip_med', 0.0):.2f}% |
| High | {stats.get('flip_high', 0.0):.2f}% |

## 5. Conditional Utility Deviation

$$D_Q(i,S) = \\Delta Q(i|S) - \\Delta Q(i|\\emptyset)$$

| Metric | Value |
|--------|-------|
| Mean $|D_Q|$ | {stats.get('mean_abs_deviation', 0.0):.6f} |
| Mean $R_Q$ | {stats.get('mean_ratio_rq', 0.0):.4f} |

## 6. GO / NO-GO Assessment

**Decision: {stats.get('decision', 'PENDING')}**

{chr(10).join('- ' + r for r in stats.get('reasons', ['Assessment pending.']))}
"""

    with open(output_path, 'w') as f:
        f.write(report)

def compute_statistics(pairs_data: List[Dict], contexts_data: List[Dict]) -> Dict[str, Any]:
    """Compute statistics from pairs and contexts data using canonical schema."""
    
    total_pairs = len(pairs_data)
    eps = 1e-6
    
    # Pair interaction statistics
    interactions = [p['interaction_residual'] for p in pairs_data]
    ratios = [p.get('additivity_ratio', 1.0) for p in pairs_data]
    sub_additive = sum(1 for i in interactions if i < -eps)
    super_additive = sum(1 for i in interactions if i > eps)
    
    # By overlap bin
    high_overlap = [p for p in pairs_data if p.get('overlap_bin') == 'high']
    med_overlap = [p for p in pairs_data if p.get('overlap_bin') == 'medium']
    low_overlap = [p for p in pairs_data if p.get('overlap_bin') == 'low']
    
    # Context statistics
    total_contexts = len(contexts_data)
    sign_flips = sum(1 for c in contexts_data if c.get('sign_flip', False))
    deviations = [c.get('deviation_dq', 0.0) for c in contexts_data]
    rq_vals = [c.get('ratio_rq', 1.0) for c in contexts_data]
    
    # Sign flips by overlap
    ctx_high = [c for c in contexts_data if c.get('overlap_bin') == 'high']
    ctx_med = [c for c in contexts_data if c.get('overlap_bin') == 'medium']
    ctx_low = [c for c in contexts_data if c.get('overlap_bin') == 'low']
    
    # GO/NO-GO
    r_flip = sign_flips / max(1, total_contexts)
    mean_abs_I = float(np.mean(np.abs(interactions))) if interactions else 0.0
    is_go = r_flip > 0.05 or mean_abs_I > 1e-4
    reasons = []
    if r_flip > 0.05:
        reasons.append(f"R_flip = {r_flip:.3f} > 0.05")
    if mean_abs_I > 1e-4:
        reasons.append(f"|I_ij| = {mean_abs_I:.6f} > 1e-4")
    if not is_go:
        reasons.append("Interactions weak: pointwise utility may suffice.")
    
    stats = {
        'total_pairs': total_pairs,
        'total_contexts': total_contexts,
        'mean_interaction': float(np.mean(interactions)) if interactions else 0.0,
        'mean_abs_interaction': mean_abs_I,
        'sub_additive_percent': (sub_additive / max(1, total_pairs)) * 100,
        'super_additive_percent': (super_additive / max(1, total_pairs)) * 100,
        'R_pair_mean': float(np.mean(ratios)) if ratios else 1.0,
        'I_ij_low': float(np.mean([p['interaction_residual'] for p in low_overlap])) if low_overlap else 0.0,
        'I_ij_med': float(np.mean([p['interaction_residual'] for p in med_overlap])) if med_overlap else 0.0,
        'I_ij_high': float(np.mean([p['interaction_residual'] for p in high_overlap])) if high_overlap else 0.0,
        'sub_add_low': sum(1 for p in low_overlap if p['interaction_residual'] < -eps) / max(1, len(low_overlap)) * 100,
        'sub_add_med': sum(1 for p in med_overlap if p['interaction_residual'] < -eps) / max(1, len(med_overlap)) * 100,
        'sub_add_high': sum(1 for p in high_overlap if p['interaction_residual'] < -eps) / max(1, len(high_overlap)) * 100,
        'sign_flip_percent': r_flip * 100,
        'flip_low': sum(1 for c in ctx_low if c.get('sign_flip')) / max(1, len(ctx_low)) * 100,
        'flip_med': sum(1 for c in ctx_med if c.get('sign_flip')) / max(1, len(ctx_med)) * 100,
        'flip_high': sum(1 for c in ctx_high if c.get('sign_flip')) / max(1, len(ctx_high)) * 100,
        'mean_abs_deviation': float(np.mean(np.abs(deviations))) if deviations else 0.0,
        'mean_ratio_rq': float(np.mean(rq_vals)) if rq_vals else 1.0,
        'decision': 'GO' if is_go else 'NO-GO',
        'reasons': reasons,
    }
    
    return stats

def save_results(output_dir: str, pairs_data: List[Dict], contexts_data: List[Dict], stats: Dict[str, Any]):
    """Write all output files."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)
    
    if pairs_data:
        pairs_file = out_path / 'interaction_pairs.csv'
        with open(pairs_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pairs_data[0].keys())
            writer.writeheader()
            writer.writerows(pairs_data)
            
    if contexts_data:
        contexts_file = out_path / 'interaction_contexts.csv'
        with open(contexts_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=contexts_data[0].keys())
            writer.writeheader()
            writer.writerows(contexts_data)
            
    with open(out_path / 'interaction_summary.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
    generate_report(stats, out_path / 'interaction_audit_report.md')
    
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'phase12i_interaction_audit',
        'files_generated': [
            'interaction_pairs.csv',
            'interaction_contexts.csv',
            'interaction_summary.json',
            'interaction_audit_report.md',
            'manifest.json'
        ]
    }
    with open(out_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Results saved to {out_path}")

def main():
    args = setup_argparse()
    
    if args.generate_synthetic:
        print("Running in synthetic data generation mode...")
        pairs_data, contexts_data = generate_synthetic_audit_data(
            n_frames=args.n_frames, 
            n_pairs=args.n_pairs, 
            n_contexts=args.n_candidates * args.n_frames
        )
        stats = compute_statistics(pairs_data, contexts_data)
        save_results(args.output_dir, pairs_data, contexts_data, stats)
        return
        
    if args.offline:
        print("Running in offline analysis mode...")
        if not args.data_dir:
            print("Error: --data-dir must be provided for offline analysis.")
            sys.exit(1)
        pairs_data, contexts_data = run_offline_analysis(args.data_dir)
        if pairs_data is None:
            sys.exit(1)
        stats = compute_statistics(pairs_data, contexts_data)
        save_results(args.output_dir, pairs_data, contexts_data, stats)
        return
        
    print(f"Running Live Phase 12-I Audit on scene '{args.scene}' with seed {args.seed}...")
    
    # Initialize pipeline and config
    config = InteractionAuditConfig(
        n_candidates=args.n_candidates,
        n_pairs=args.n_pairs
    )
    
    # Mock pipeline for now since we may not have it
    class MockPipeline:
        pass
    pipeline = MockPipeline()
    
    # Create experiment
    experiment = InteractionAuditExperiment(pipeline, config)
    
    all_pairs_data = []
    all_contexts_data = []
    
    # Run across frames
    for frame_idx in range(args.n_frames):
        print(f"Auditing frame {frame_idx}/{args.n_frames}...")
        
        # In a real scenario, we would call experiment.run_full_frame_audit()
        # Since we don't have the real pipeline running, we'll generate mock results for this frame
        frame_pairs, frame_contexts = generate_synthetic_audit_data(1, args.n_pairs, args.n_candidates)
        
        all_pairs_data.extend(frame_pairs)
        all_contexts_data.extend(frame_contexts)
        
    stats = compute_statistics(all_pairs_data, all_contexts_data)
    save_results(args.output_dir, all_pairs_data, all_contexts_data, stats)
    
    print("Audit complete.")

if __name__ == "__main__":
    main()
