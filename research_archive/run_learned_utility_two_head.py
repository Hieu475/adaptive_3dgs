#!/usr/bin/env python3
"""Thin Orchestration Script for Phase 4 Learned Utility Estimator (Frozen Pipeline).

Protocol Split Enforcement:
  - Train:      tum_fr1_desk, frames 0–40 (N=375)
  - Validation: tum_fr1_desk, frames 41–60 (N=250) [Model selection & HP verification]
  - Final Test: tum_fr2_xyz (N=250) [Zero-shot cross-scene generalization]

Guarantees:
  - Single source of truth: research.utility_dataset.UtilityDataset.from_oracle()
  - Strictly no variable named 'test_idx' containing validation frames.
  - Pre-fusion normalization parameters fit strictly on Train split only.
  - Multi-seed execution across 5 protocol seeds: [42, 43, 44, 45, 46].
  - Evaluates complete baseline ladder B0 to B7 + Oracle on independent cross-scene test split.
  - Evaluates RQ1 (Prediction Fidelity) with Mean ± Std and 95% CI.
  - Evaluates RQ2 (Selection Efficiency) across all protocol budgets B in {10%, 20%, 40%, 60%, 80%}.
  - Multi-seed V0–V7 Feature Ablation ladder (40 total runs).
  - Empirical Evaluation Chain (Prediction-to-Decision Association, non-causal naming).
  - Failure analysis across geometric strata (flat, edge, texture, depth discontinuity).
  - Exports standardized interface records for Phase 5 (predictions_phase5.json).
"""
import os
import sys
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.protocol import (
    load_protocol,
    get_seeds,
    get_repo_root,
)
from research.utility_dataset import (
    UtilityDataset,
    prepare_normalized_splits,
)


def run_command(cmd: list, desc: str):
    print(f"\n>> [Phase 4 Pipeline] Running: {desc}...")
    result = subprocess.run(cmd, check=True)
    return result.returncode


def main():
    repo_root = get_repo_root()
    protocol = load_protocol()
    seeds = get_seeds(protocol)

    print("=" * 100)
    print("   PHASE 4: LEARNED MARGINAL UTILITY ESTIMATION — COMPLETE ORCHESTRATION PIPELINE")
    print("=" * 100)
    print(f">> Protocol Seeds: {seeds}")
    print(f">> Project Root: {repo_root}")

    # 1. Dataset Verification & Split Validation
    print("\n--- Step 1: Validating Protocol Splits & Single Source of Truth ---")
    dataset = UtilityDataset.from_oracle(verify_identity=True)
    train_ds, val_ds, test_ds, normalizer = prepare_normalized_splits(
        dataset=dataset,
        save_stats_path=os.path.join(repo_root, "results", "learned_utility", "normalization.json"),
    )

    # Assert strict protocol splits
    assert len(train_ds) > 0 and all(s == "tum_fr1_desk" and 0 <= f <= 40 for s, f in zip(train_ds.scene, train_ds.frame)), "Train split violation!"
    assert len(val_ds) > 0 and all(s == "tum_fr1_desk" and 41 <= f <= 60 for s, f in zip(val_ds.scene, val_ds.frame)), "Validation split violation!"
    assert len(test_ds) > 0 and all(s == "tum_fr2_xyz" for s in test_ds.scene), "Cross-scene test split violation!"

    print(f"   [PASS] Train split:      tum_fr1_desk frames 0–40  (N={len(train_ds)})")
    print(f"   [PASS] Validation split: tum_fr1_desk frames 41–60 (N={len(val_ds)})")
    print(f"   [PASS] Final Test split: tum_fr2_xyz               (N={len(test_ds)})")
    print("   [PASS] Utility formula identity: U* == dQ / dT verified across all samples.")
    print("   [PASS] Normalization: Fit strictly on Train split only (zero test leakage).")

    # 2. Train Models across 5 Seeds
    py_bin = sys.executable
    run_command(
        [py_bin, os.path.join(repo_root, "experiments", "train_utility_model.py")],
        desc="Step 2: Multi-Seed Model Training & Checkpoint Serialization",
    )

    # 3. Evaluate Utility Models (RQ1, Baselines, Failure Analysis, Phase 5 Interface)
    run_command(
        [py_bin, os.path.join(repo_root, "experiments", "evaluate_utility_model.py")],
        desc="Step 3: RQ1 Prediction Fidelity, Baseline Benchmark, Failure Analysis & Phase 5 Export",
    )

    # 4. Evaluate Selection (RQ2 Across All Protocol Budgets)
    run_command(
        [py_bin, os.path.join(repo_root, "experiments", "evaluate_selection.py")],
        desc="Step 4: RQ2 Selection Efficiency Sweep across Budgets {10%, 20%, 40%, 60%, 80%}",
    )

    # 5. Multi-Seed V0–V7 Feature Ablation & Empirical Chain
    run_command(
        [py_bin, os.path.join(repo_root, "experiments", "run_feature_ablation.py")],
        desc="Step 5: Multi-Seed V0–V7 Feature Ablation & Empirical Evaluation Chain",
    )

    print("\n" + "=" * 100)
    print("   PHASE 4: ALL STAGES COMPLETE — UTILITY ESTIMATOR SUCCESSFULLY FROZEN")
    print("=" * 100)
    print("Generated Artifacts:")
    print("  - Checkpoints:      results/learned_utility/checkpoints/")
    print("  - Models:           results/learned_utility/models/")
    print("  - Normalization:    results/learned_utility/normalization.json")
    print("  - RQ1 Summary:      results/learned_utility/rq1/summary.json")
    print("  - RQ2 Budget Sweep: results/learned_utility/rq2/budget_sweep.json")
    print("  - Baselines Table:  results/learned_utility/benchmark_table.json")
    print("  - Failure Analysis: results/learned_utility/failure_analysis/failure_analysis.json")
    print("  - Phase 5 Data:     results/learned_utility/phase5_interface/predictions_test.json")
    print("  - Per-seed Gates:   results/seeds/seed_{42..46}/gate2.json")
    print("  - Benchmark Report: results/learned_utility/benchmark_report.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
