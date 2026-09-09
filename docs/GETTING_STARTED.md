# Getting Started: Adaptive 3D Gaussian Splatting

This guide walks through setting up the environment, running the full test suite (417/417 PASS), and reproducing the confirmatory experimental benchmarks.

---

## 1. Prerequisites

- **OS**: Linux (Ubuntu 20.04 / 22.04 LTS recommended)
- **Python**: 3.10+
- **PyTorch**: >= 2.0 with CUDA support
- **CUDA Toolkit**: >= 11.8

Install required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Running the Confirmatory Test Suite

The repository contains 417 rigorous unit, regression, and invariance tests covering Gaussian state stores, counterfactual oracle evaluations, selective optimization schedulers, and Phase 6 rank stability audits:

```bash
pytest -q
```
**Expected Output**: `417 passed in ~8s (100% PASS)`

---

## 3. Reproducing Scientific Gates & Frozen Artifacts

Every benchmark output has a deterministic Single Source of Truth under `results/`:

```bash
# 1. Gate 1: Measurability, Headroom, and Group Non-Additivity
python3 experiments/run_gate1_headroom.py

# 2. Phase 4: Two-Head Learned Utility Model (5 Seeds)
python3 experiments/train_utility_model.py
python3 experiments/eval_utility_model.py
python3 experiments/eval_selection.py

# 3. Phase 5: Controlled Budget Benchmark & Online Trajectory
python3 experiments/run_phase5_budget_benchmark.py --seeds 42 43 44 45 46
python3 experiments/run_phase5_online_trajectory.py --n-frames 25 --budget-ms 15.0

# 4. Phase 6: Candidate Pool Coverage Audit & Rank Stability (Case B Proof)
python3 experiments/run_phase6_rank_stability.py

# 5. Phase 6: 8-Variant Architecture Ladder Ablation
python3 experiments/run_phase6_ablation.py --seed 42

# 6. Phase 6: 5-Policy Oracle Decomposition & Normalized Regret
python3 experiments/run_phase6_oracle_gap.py

# 7. Phase 6: Runtime Profiling Breakdown (T_P6 Pipeline Stages)
python3 experiments/run_phase6_runtime_profile.py
```

---

## 4. Single Source of Truth References

- **Phase 6 Authoritative Manifest**: `results/phase6_context_utility/manifest.json`
- **Rank Stability Analysis**: `results/phase6_context_utility/rank_stability_analysis.json`
- **Ablation Ladder Summary**: `results/phase6_context_utility/ablation/ablation_summary.json`
- **Runtime Breakdown**: `results/phase6_context_utility/runtime_breakdown.json`

