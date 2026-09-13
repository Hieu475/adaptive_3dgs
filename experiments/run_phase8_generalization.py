#!/usr/bin/env python3
"""Phase 8: Generalization / Zero-Shot Transfer Evaluation.

Evaluates whether the frozen Phase 4 utility predictor (TwoHeadMLP)
generalizes from tum_fr1_desk to unseen tum_fr2_xyz without fine-tuning.

Execution Stages:
    Stage A: Utility prediction metrics (ρ, NDCG, OSE) on test scene
    Stage B: Budget selection metrics (ΔQ, regret) at equal budgets
    Stage C: Generalization gap analysis (in-domain vs zero-shot)

Protocol invariants:
    - Checkpoints are frozen from Phase 4 (no retraining)
    - Normalizer was fit on train split only (N=375)
    - Test scene never participated in training or normalization
    - n=5 seeds for statistical inference
    - Gate criteria defined before experiments (see phase8_protocol.py)

Usage:
    python experiments/run_phase8_generalization.py [--stage A|B|C|all] [--device cuda|cpu]
"""
import os
import sys
import json
import time
import hashlib
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.phase8_protocol import (
    SEEDS, TRAIN_SCENES, TEST_SCENES, TRAIN_FRAME_RANGE, VAL_FRAME_RANGE,
    BUDGETS, TOP_K_FRACTIONS, FEATURE_SCHEMA, BASELINES,
    MODEL_IN_FEATURES, MODEL_HIDDEN_DIM, MODEL_EPS_COST,
    ORACLE_N_SAMPLES, ORACLE_N_OPT_STEPS, ORACLE_W_RGB, ORACLE_W_DEPTH,
    ORACLE_MIN_INFLUENCE_PIXELS,
    CONFIDENCE_LEVEL, BOOTSTRAP_RESAMPLES, N_SEEDS,
    PIPELINE_CONFIG, GATE_CRITERIA, EXECUTION_STAGES,
    OUTPUT_DIR, OUTPUT_FILES, SEED_RESULT_PATTERN,
    get_repo_root, get_checkpoint_path, get_normalizer_path, get_output_dir,
    validate_no_leakage, to_dict as protocol_to_dict,
)
from research.protocol import (
    load_protocol, get_resolution, get_dataset_config, get_seeds,
)
from research.utility_predictor import FrozenUtilityPredictor
from research.utility_metrics import (
    evaluate_rq1_prediction, evaluate_rq2_selection, evaluate_utility_complete,
    safe_spearmanr, compute_ndcg_at_k, PROTOCOL_BUDGETS,
)
from research.utility_models import TwoHeadMLP
from research.utility_dataset import FeatureNormalizer
from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
from research.pipeline import OnlineReconstructionPipeline
from datasets.tum_dataset import TUMDataset


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_tum_frames(
    data_path: str,
    camera: str,
    n_frames: int,
    start_frame: int = 0,
    H: int = 240,
    W: int = 320,
    device: str = 'cuda',
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
    """Load and resize TUM-RGBD frames with proper intrinsics scaling."""
    dataset = TUMDataset(data_path, max_frames=start_frame + n_frames + 5, camera=camera)
    frames = []
    orig_W, orig_H = 640.0, 480.0
    scale_x = W / orig_W
    scale_y = H / orig_H

    intrinsics = torch.tensor([
        [dataset.fx * scale_x, 0.0, dataset.cx * scale_x],
        [0.0, dataset.fy * scale_y, dataset.cy * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    for i in range(start_frame, min(start_frame + n_frames, len(dataset))):
        item = dataset[i]
        rgb = item['rgb'].unsqueeze(0).permute(0, 3, 1, 2)
        depth = item['depth'].unsqueeze(0).unsqueeze(0)

        rgb_scaled = torch.nn.functional.interpolate(
            rgb, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        depth_scaled = torch.nn.functional.interpolate(
            depth, size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0)

        frames.append({
            'frame_id': i,
            'rgb': rgb_scaled.to(device),
            'depth': depth_scaled.to(device),
            'pose': item['pose'].to(device),
        })

    return frames, intrinsics


def build_pipeline_and_collect_oracle(
    frames: List[Dict[str, torch.Tensor]],
    intrinsics: torch.Tensor,
    seed: int,
    device: str = 'cuda',
    scene_name: str = 'tum_fr2_xyz',
) -> List[Dict[str, Any]]:
    """Build pipeline on frames and collect oracle utility for last frame.

    Returns list of candidate dictionaries with oracle U*, features, etc.
    """
    config = dict(PIPELINE_CONFIG)
    config['rendering']['image_width'] = int(intrinsics[0, 2].item() * 2)  # 2 * cx ≈ W
    config['rendering']['image_height'] = int(intrinsics[1, 2].item() * 2)  # 2 * cy ≈ H

    pipeline = OnlineReconstructionPipeline(config=config, device=device)
    pipeline.initialize(
        rgb=frames[0]['rgb'],
        depth=frames[0]['depth'],
        intrinsics=intrinsics,
        pose=frames[0]['pose'],
    )

    # Process all frames except the last (which is the evaluation frame)
    for t in range(1, len(frames) - 1):
        pipeline.process_frame(
            rgb=frames[t]['rgb'],
            depth=frames[t]['depth'],
            gt_pose=frames[t]['pose'],
        )

    # Run oracle on the last frame
    eval_frame = frames[-1]
    oracle = OracleUtilityExperiment(
        pipeline=pipeline,
        n_samples=ORACLE_N_SAMPLES,
        n_opt_steps=ORACLE_N_OPT_STEPS,
        w_rgb=ORACLE_W_RGB,
        w_depth=ORACLE_W_DEPTH,
        seed=seed,
        min_influence_pixels=ORACLE_MIN_INFLUENCE_PIXELS,
    )

    results = oracle.run_oracle_experiment(
        rgb=eval_frame['rgb'],
        depth=eval_frame['depth'],
        population_type=SamplingPopulation.GEOMETRY_STRATIFIED,
        scene_name=scene_name,
        frame_idx=eval_frame.get('frame_id', len(frames) - 1),
        split='cross_scene_test',
        seed=seed,
    )

    # Filter to visible, high-influence candidates
    visible = [r for r in results
                if r.get('visible', True) and r.get('n_influence_pixels', 0) > 0]

    return visible


# ═══════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_features_from_candidates(
    candidates: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 11-dim feature vectors, oracle utility, delta_q, and cost from candidates.

    Returns:
        X: [N, 11] feature matrix
        oracle_u: [N] oracle utility
        delta_q: [N] quality gain
        costs: [N] measured trial cost
    """
    X, oracle_u, delta_q, costs = [], [], [], []

    for cand in candidates:
        f = cand.get('features', {})
        vec = []
        for feat_name in FEATURE_SCHEMA:
            vec.append(float(f.get(feat_name, 0.0)))
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
    """Compute baseline scoring functions.

    Returns dict mapping baseline name to score array [N].
    """
    n = len(X)
    rng = np.random.default_rng(seed)

    # Random
    random_scores = rng.random(n).astype(np.float32)

    # Error-only: rgb_error + depth_error (features 0 and 1)
    error_scores = X[:, 0] + X[:, 1]

    # Heuristic: pipeline's predicted_utility if available, else error
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


# ═══════════════════════════════════════════════════════════════════════
# Stage A: Utility Prediction
# ═══════════════════════════════════════════════════════════════════════

def run_stage_a(
    device: str = 'cuda',
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage A: Evaluate utility prediction metrics on test scene.

    For each seed:
        1. Load frozen checkpoint + normalizer (train-only)
        2. Build pipeline on tum_fr2_xyz
        3. Collect oracle ground truth
        4. Predict utility with frozen model
        5. Compute ρ, NDCG, OSE for learned + baselines
    """
    if repo_root is None:
        repo_root = str(get_repo_root())

    protocol = load_protocol()
    validate_no_leakage()

    # Test scene config
    test_cfg = get_dataset_config('tum_fr2_xyz', protocol)
    test_path = test_cfg['full_path']
    test_camera = test_cfg.get('camera', 'freiburg2')
    H, W = get_resolution('tum_fr2_xyz', protocol)

    # In-domain validation config
    train_cfg = get_dataset_config('tum_fr1_desk', protocol)
    train_path = train_cfg['full_path']
    train_camera = train_cfg.get('camera', 'freiburg1')
    H_train, W_train = get_resolution('tum_fr1_desk', protocol)

    print("=" * 70)
    print("PHASE 8 — STAGE A: ZERO-SHOT UTILITY PREDICTION")
    print("=" * 70)
    print(f"  Device:         {device}")
    print(f"  Seeds:          {SEEDS}")
    print(f"  Test scene:     tum_fr2_xyz ({test_path})")
    print(f"  Resolution:     {W}×{H}")
    print(f"  Features:       {len(FEATURE_SCHEMA)} canonical")
    print(f"  Normalizer:     {get_normalizer_path()} (train-only, N=375)")
    print()

    all_seed_results = {}
    prediction_rows = []  # for CSV

    for seed in SEEDS:
        print(f"\n{'─' * 50}")
        print(f"  SEED {seed}")
        print(f"{'─' * 50}")

        # 1. Load frozen predictor (checkpoint + normalizer)
        ckpt_path = get_checkpoint_path(seed)
        print(f"  Loading checkpoint: {os.path.basename(ckpt_path)}")
        predictor = FrozenUtilityPredictor(
            checkpoint_path=ckpt_path,
            normalizer_path=get_normalizer_path(),
            seed=seed,
            device=device,
        )

        # ─── IN-DOMAIN evaluation (val split of fr1_desk) ───
        print(f"  Collecting in-domain oracle (fr1_desk val frames)...")
        indomain_frames, indomain_intrinsics = load_tum_frames(
            data_path=train_path,
            camera=train_camera,
            n_frames=60,  # load all 60 frames
            start_frame=0,
            H=H_train, W=W_train,
            device=device,
        )
        # Use frames 41-59 for building pipeline, frame 59 as eval
        val_frames = [f for f in indomain_frames if f['frame_id'] >= 41]
        if len(val_frames) < 3:
            print(f"  WARNING: only {len(val_frames)} val frames, using all frames > 40")
            val_frames = indomain_frames[41:]

        indomain_candidates = build_pipeline_and_collect_oracle(
            frames=val_frames,
            intrinsics=indomain_intrinsics,
            seed=seed,
            device=device,
            scene_name='tum_fr1_desk',
        )
        print(f"  In-domain candidates: {len(indomain_candidates)}")

        # ─── ZERO-SHOT evaluation (fr2_xyz) ───
        print(f"  Collecting zero-shot oracle (fr2_xyz)...")
        test_frames, test_intrinsics = load_tum_frames(
            data_path=test_path,
            camera=test_camera,
            n_frames=20,
            start_frame=0,
            H=H, W=W,
            device=device,
        )
        test_candidates = build_pipeline_and_collect_oracle(
            frames=test_frames,
            intrinsics=test_intrinsics,
            seed=seed,
            device=device,
            scene_name='tum_fr2_xyz',
        )
        print(f"  Zero-shot candidates: {len(test_candidates)}")

        # ─── Evaluate both domains ───
        seed_result = {'seed': seed}

        for domain_name, candidates, domain_label in [
            ('in_domain', indomain_candidates, 'fr1_desk_val'),
            ('zero_shot', test_candidates, 'fr2_xyz'),
        ]:
            if len(candidates) < 5:
                print(f"  WARNING: {domain_label} has only {len(candidates)} candidates, skipping")
                seed_result[domain_name] = {'error': f'insufficient candidates ({len(candidates)})'}
                continue

            X, oracle_u, delta_q, costs = extract_features_from_candidates(candidates)
            baselines = compute_baseline_scores(X, candidates, seed)

            # Frozen model prediction
            preds = predictor.predict_features(X)
            learned_u = preds['predicted_utility']

            # Compute metrics for each method
            domain_metrics = {}
            all_methods = {
                'random': baselines['random'],
                'error_only': baselines['error_only'],
                'heuristic': baselines['heuristic'],
                'learned': learned_u,
                'oracle': oracle_u,  # upper bound
            }

            for method_name, scores in all_methods.items():
                rq1 = evaluate_rq1_prediction(
                    pred_u=scores, oracle_u=oracle_u,
                )
                rq2 = evaluate_rq2_selection(
                    pred_u=scores, oracle_u=oracle_u,
                    delta_q=delta_q, costs=costs,
                )
                combined = {**rq1, **rq2}
                domain_metrics[method_name] = combined

                # CSV row
                prediction_rows.append({
                    'seed': seed,
                    'domain': domain_label,
                    'method': method_name,
                    'n_candidates': len(candidates),
                    'spearman_rho': combined['spearman_rho'],
                    'spearman_pval': combined['spearman_pval'],
                    'pearson_r': combined['pearson_r'],
                    'ndcg_10pct': combined.get('ndcg_10pct', float('nan')),
                    'ndcg_20pct': combined.get('ndcg_20pct', float('nan')),
                    'ose_20pct': combined.get('ose_20pct', float('nan')),
                    'regret_20pct': combined.get('regret_20pct', float('nan')),
                })

            seed_result[domain_name] = domain_metrics

            # Print summary
            print(f"\n  [{domain_label}] N={len(candidates)}")
            for method in ['random', 'error_only', 'heuristic', 'learned']:
                m = domain_metrics[method]
                print(f"    {method:12s}: ρ={m['spearman_rho']:+.4f}  "
                      f"NDCG@20={m.get('ndcg_20pct', 0):.4f}  "
                      f"OSE@20={m.get('ose_20pct', 0):.3f}")

        all_seed_results[str(seed)] = seed_result

    # ─── Aggregate across seeds ───
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS (n=5 seeds)")
    print(f"{'=' * 70}")

    aggregate = {}
    for domain in ['in_domain', 'zero_shot']:
        domain_agg = {}
        for method in ['random', 'error_only', 'heuristic', 'learned']:
            rhos = []
            ndcgs = []
            oses = []
            for s in SEEDS:
                sr = all_seed_results[str(s)]
                if domain in sr and isinstance(sr[domain], dict) and method in sr[domain]:
                    m = sr[domain][method]
                    rhos.append(m['spearman_rho'])
                    ndcgs.append(m.get('ndcg_20pct', float('nan')))
                    oses.append(m.get('ose_20pct', float('nan')))

            if rhos:
                rhos_arr = np.array(rhos)
                ndcgs_arr = np.array(ndcgs)
                oses_arr = np.array(oses)

                domain_agg[method] = {
                    'mean_rho': float(np.mean(rhos_arr)),
                    'std_rho': float(np.std(rhos_arr, ddof=1)) if len(rhos_arr) > 1 else 0.0,
                    'mean_ndcg_20': float(np.nanmean(ndcgs_arr)),
                    'std_ndcg_20': float(np.nanstd(ndcgs_arr, ddof=1)) if len(ndcgs_arr) > 1 else 0.0,
                    'mean_ose_20': float(np.nanmean(oses_arr)),
                    'std_ose_20': float(np.nanstd(oses_arr, ddof=1)) if len(oses_arr) > 1 else 0.0,
                    'n_seeds': len(rhos_arr),
                    'seed_rhos': rhos,
                }

        aggregate[domain] = domain_agg

        # Print
        print(f"\n  [{domain}]")
        print(f"  {'Method':12s} | {'Mean ρ':>10s} | {'Std ρ':>8s} | {'NDCG@20':>10s} | {'OSE@20':>10s}")
        print(f"  {'-' * 60}")
        for method in ['random', 'error_only', 'heuristic', 'learned']:
            if method in domain_agg:
                a = domain_agg[method]
                print(f"  {method:12s} | {a['mean_rho']:+.4f}     | {a['std_rho']:.4f}   | "
                      f"{a['mean_ndcg_20']:.4f}      | {a['mean_ose_20']:.3f}")

    # ─── Generalization gap ───
    gap = {}
    for method in ['random', 'error_only', 'heuristic', 'learned']:
        if method in aggregate.get('in_domain', {}) and method in aggregate.get('zero_shot', {}):
            in_d = aggregate['in_domain'][method]
            zs = aggregate['zero_shot'][method]
            gap[method] = {
                'delta_rho': in_d['mean_rho'] - zs['mean_rho'],
                'delta_ndcg_20': in_d['mean_ndcg_20'] - zs['mean_ndcg_20'],
                'delta_ose_20': in_d['mean_ose_20'] - zs['mean_ose_20'],
            }

    if gap:
        print(f"\n  GENERALIZATION GAP (in_domain - zero_shot)")
        print(f"  {'Method':12s} | {'Δρ':>8s} | {'ΔNDCG@20':>10s} | {'ΔOSE@20':>10s}")
        print(f"  {'-' * 50}")
        for method in ['random', 'error_only', 'heuristic', 'learned']:
            if method in gap:
                g = gap[method]
                print(f"  {method:12s} | {g['delta_rho']:+.4f}  | {g['delta_ndcg_20']:+.4f}      | {g['delta_ose_20']:+.3f}")

    # ─── Gate 8B evaluation ───
    gate_8b = evaluate_gate_8b(aggregate)
    print(f"\n  Gate 8B (Zero-Shot Prediction): {gate_8b['status']}")
    print(f"    {gate_8b['rationale']}")

    # ─── Save results ───
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    result_package = {
        'phase': 8,
        'stage': 'A',
        'title': 'Zero-Shot Utility Prediction',
        'generated_at': datetime.datetime.now().isoformat(),
        'device': device,
        'seeds': SEEDS,
        'protocol': protocol_to_dict(),
        'per_seed': all_seed_results,
        'aggregate': aggregate,
        'generalization_gap': gap,
        'gate_8b': gate_8b,
    }

    # Save per-seed JSONs
    for seed in SEEDS:
        seed_path = os.path.join(output_dir, SEED_RESULT_PATTERN.format(seed=seed))
        with open(seed_path, 'w') as f:
            json.dump(all_seed_results[str(seed)], f, indent=2, default=str)

    # Save prediction_metrics.csv
    csv_path = os.path.join(output_dir, OUTPUT_FILES['prediction_metrics'])
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)

    # Save stage A summary
    summary_path = os.path.join(output_dir, 'stage_a_results.json')
    with open(summary_path, 'w') as f:
        json.dump(result_package, f, indent=2, default=str)

    print(f"\n  Saved results to {output_dir}/")
    return result_package


# ═══════════════════════════════════════════════════════════════════════
# Gate Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_gate_8b(aggregate: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate Gate 8B: Zero-shot prediction quality.

    PASS if: rho_learned > rho_random AND rho_learned > 0 on zero-shot domain.
    """
    zs = aggregate.get('zero_shot', {})
    learned = zs.get('learned', {})
    random = zs.get('random', {})
    error = zs.get('error_only', {})

    rho_learned = learned.get('mean_rho', float('nan'))
    rho_random = random.get('mean_rho', float('nan'))
    rho_error = error.get('mean_rho', float('nan'))

    ndcg_learned = learned.get('mean_ndcg_20', float('nan'))
    ndcg_random = random.get('mean_ndcg_20', float('nan'))

    # Primary criterion
    primary_pass = rho_learned > 0 and rho_learned > rho_random
    # Secondary criterion
    secondary_pass = ndcg_learned > ndcg_random

    if primary_pass:
        status = 'PASS'
        rationale = (f'ρ_learned={rho_learned:.4f} > 0 and > ρ_random={rho_random:.4f}. '
                     f'ρ_error={rho_error:.4f} for reference.')
    elif rho_learned > 0:
        status = 'WEAK_PASS'
        rationale = (f'ρ_learned={rho_learned:.4f} > 0 but not > ρ_random={rho_random:.4f}. '
                     f'Signal exists but may not be reliable.')
    else:
        status = 'FAIL'
        rationale = (f'ρ_learned={rho_learned:.4f} <= 0. '
                     f'No evidence of zero-shot transfer.')

    return {
        'status': status,
        'primary_pass': primary_pass,
        'secondary_pass': secondary_pass,
        'rho_learned': rho_learned,
        'rho_random': rho_random,
        'rho_error': rho_error,
        'ndcg_learned': ndcg_learned,
        'ndcg_random': ndcg_random,
        'rationale': rationale,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 8: Generalization / Zero-Shot Transfer Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Workflow:
  1. evaluate (Stage A):
     Reconstructs tum_fr1_desk and tum_fr2_xyz pipelines, collects ground-truth
     oracle utility U*, evaluates frozen Phase 4 TwoHeadMLP predictions and baselines
     across n=5 seeds, and writes stage_a_results.json, seed_*.json, and prediction_metrics.csv.

  2. process (Stage B/C & Freeze):
     Aggregates selection and regret metrics across all 5 budgets, generates publication
     figures (fig12-fig15), evaluates Gates 8A-8D, writes generalization_summary.md,
     and computes SHA256 checksums in manifest.json.

  3. all:
     Executes evaluate followed by process.
        """
    )
    parser.add_argument(
        '--stage', type=str, default='evaluate',
        choices=['evaluate', 'process', 'all', 'A'],
        help="Stage to execute: 'evaluate' (or 'A', live GPU evaluation), 'process' (post-process & freeze), 'all' (both)"
    )
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available)')
    args = parser.parse_args()

    # Map legacy alias
    stage = 'evaluate' if args.stage == 'A' else args.stage

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("PHASE 8: GENERALIZATION / ZERO-SHOT TRANSFER")
    print(f"  Stage:  {stage} (raw arg: {args.stage})")
    print(f"  Device: {device}")
    print("=" * 70)
    print()

    # Gate 8A: Protocol integrity check
    validate_no_leakage()
    assert len(SEEDS) == N_SEEDS == 5, f"Expected {N_SEEDS} seeds, found {len(SEEDS)}"
    for seed in SEEDS:
        ckpt = get_checkpoint_path(seed)
        assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
    assert os.path.exists(get_normalizer_path()), f"Missing normalizer: {get_normalizer_path()}"
    print("Gate 8A (Protocol Integrity): PASS")
    print()

    if stage in ('evaluate', 'all'):
        result_a = run_stage_a(device=device)
        gate_status = result_a['gate_8b']['status']
        if gate_status == 'FAIL':
            print("\n⚠ Gate 8B FAIL — model shows no zero-shot transfer signal.")
            print("  Phase 9 should address robust utility representation.")
            return

    if stage in ('process', 'all'):
        print("\n" + "=" * 70)
        print("RUNNING POST-PROCESSING, FIGURES GENERATION & FREEZE MANIFEST")
        print("=" * 70)
        from experiments.process_phase8_results import main as process_main
        process_main()


if __name__ == '__main__':
    main()
