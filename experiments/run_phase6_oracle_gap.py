#!/usr/bin/env python3
"""Phase 6: Oracle Gap & Selection Regret Analysis (P1.2 & P1.3).

Computes across seeds and budget levels:
  1. Oracle Gap:
       Gap_P4 = Q(Oracle_Static) - Q(P4_Learned)
       Gap_P6 = Q(Oracle_Cond)   - Q(P6_Adaptive)
  2. Absolute Selection Regret:
       Regret(policy) = Q(S*) - Q(S_policy)
  3. Normalized Selection Regret:
       NormRegret(policy) = [Q(S*) - Q(S_policy)] / [Q(S*) - Q(∅)]

Outputs:
  results/phase6_context_utility/oracle_gap_and_regret.json
"""
import os
import sys
import json
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
benchmark_path = os.path.join(
    repo_root, "results", "phase6_context_utility", "selection", "selection_benchmark_reformed_tum_fr2_xyz.json"
)
oracle_decomp_path = os.path.join(
    repo_root, "results", "phase6_context_utility", "oracle_benchmark", "oracle_benchmark_decomposition_seed_42.json"
)
output_path = os.path.join(
    repo_root, "results", "phase6_context_utility", "oracle_gap_and_regret.json"
)

# 1. Decomposition Benchmark (30% & 60% Oracle Comparison)
decomp_summary = {}
if os.path.exists(oracle_decomp_path):
    with open(oracle_decomp_path, "r") as f:
        decomp_data = json.load(f)
    b_res = decomp_data.get("benchmark_results", {})
    for b_level, p_dict in b_res.items():
        q_oracle_static = p_dict["Oracle Static"]["delta_q_realized"]
        q_oracle_cond = p_dict["Oracle Conditional"]["delta_q_realized"]
        q_p4 = p_dict["Phase 4 Learned"]["delta_q_realized"]
        q_p6 = p_dict["Phase 6 Adaptive"]["delta_q_realized"]
        q_heur = p_dict["Static Heuristic"]["delta_q_realized"]

        gap_p4 = q_oracle_static - q_p4
        gap_p6 = q_oracle_cond - q_p6

        regret_p4 = q_oracle_cond - q_p4
        regret_p6 = q_oracle_cond - q_p6
        regret_heur = q_oracle_cond - q_heur

        norm_regret_p4 = regret_p4 / max(q_oracle_cond, 1e-9)
        norm_regret_p6 = regret_p6 / max(q_oracle_cond, 1e-9)
        norm_regret_heur = regret_heur / max(q_oracle_cond, 1e-9)

        decomp_summary[b_level] = {
            "q_oracle_static": float(q_oracle_static),
            "q_oracle_cond": float(q_oracle_cond),
            "oracle_context_advantage": float(q_oracle_cond - q_oracle_static),
            "gap_p4": float(gap_p4),
            "gap_p6": float(gap_p6),
            "regret": {
                "phase4_learned": float(regret_p4),
                "phase6_adaptive": float(regret_p6),
                "static_heuristic": float(regret_heur),
            },
            "normalized_regret": {
                "phase4_learned": float(norm_regret_p4),
                "phase6_adaptive": float(norm_regret_p6),
                "static_heuristic": float(norm_regret_heur),
            },
            "regret_reduction_vs_heuristic_pct": float((norm_regret_heur - norm_regret_p6) / max(norm_regret_heur, 1e-6) * 100.0),
        }

# 2. Multi-Seed Full Sweep Regret Analysis
with open(benchmark_path, "r") as f:
    bench_data = json.load(f)

per_seed_results = bench_data.get("per_seed_results", [])
sweep_by_budget = {}

for seed_res in per_seed_results:
    seed = seed_res.get("seed")
    for item in seed_res.get("relative_sweep", []):
        bp = item["budget_pct_str"]
        pol = item["policy"]
        if bp not in sweep_by_budget:
            sweep_by_budget[bp] = {}
        if pol not in sweep_by_budget[bp]:
            sweep_by_budget[bp][pol] = {"abs_regret": [], "rel_regret": [], "actual_dq": []}
        sweep_by_budget[bp][pol]["abs_regret"].append(item.get("regret_abs", 0.0))
        sweep_by_budget[bp][pol]["rel_regret"].append(item.get("regret_rel", 0.0))
        sweep_by_budget[bp][pol]["actual_dq"].append(item.get("actual_delta_q", 0.0))

multi_seed_regret_summary = {}
for bp in sorted(sweep_by_budget.keys(), key=float):
    multi_seed_regret_summary[bp] = {}
    for pol, stats in sweep_by_budget[bp].items():
        multi_seed_regret_summary[bp][pol] = {
            "mean_actual_dq": float(np.mean(stats["actual_dq"])),
            "mean_abs_regret": float(np.mean(stats["abs_regret"])),
            "std_abs_regret": float(np.std(stats["abs_regret"])),
            "mean_norm_regret": float(np.mean(stats["rel_regret"])),
            "std_norm_regret": float(np.std(stats["rel_regret"])),
            "n_seeds": len(stats["actual_dq"]),
        }

full_output = {
    "oracle_decomposition_benchmark": decomp_summary,
    "multi_seed_sweep_regret": multi_seed_regret_summary,
}

with open(output_path, "w") as f:
    json.dump(full_output, f, indent=2)

print("=" * 75)
print("  PHASE 6: ORACLE GAP & REGRET ANALYSIS (P1.2, P1.3)")
print("=" * 75)
print("\n-- Oracle 5-Policy Decomposition --")
for b_level, d in decomp_summary.items():
    print(f"\nBudget {b_level}:")
    print(f"  Oracle Static Q:     {d['q_oracle_static']:.4e}")
    print(f"  Oracle Conditional Q:{d['q_oracle_cond']:.4e} (Context Advantage = {d['oracle_context_advantage']:+.2e})")
    print(f"  Gap P4:              {d['gap_p4']:.4e}")
    print(f"  Gap P6:              {d['gap_p6']:.4e}")
    print(f"  Norm Regret P4:      {d['normalized_regret']['phase4_learned']:.1%}")
    print(f"  Norm Regret P6:      {d['normalized_regret']['phase6_adaptive']:.1%}")
    print(f"  Norm Regret Heur:    {d['normalized_regret']['static_heuristic']:.1%}")
    print(f"  P6 Regret Reduction vs Heuristic: {d['regret_reduction_vs_heuristic_pct']:.1f}%")

print("\n-- Multi-Seed Selection Regret (5 Seeds Mean) --")
print(f"{'Budget (ms)':12s} | {'Policy':18s} | {'Mean Realized Q':16s} | {'Normalized Regret':18s}")
print("-" * 72)
for bp in sorted(sweep_by_budget.keys(), key=float):
    for pol in ["heuristic", "phase4_learned", "phase6_adaptive"]:
        if pol in multi_seed_regret_summary[bp]:
            st = multi_seed_regret_summary[bp][pol]
            print(f"{bp:12s} | {pol:18s} | {st['mean_actual_dq']:.4e}         | {st['mean_norm_regret']:.1%}")

print(f"\n[Saved] Artifact: {output_path}")
