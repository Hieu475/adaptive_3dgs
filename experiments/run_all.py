"""Master Research Experiment Runner (R8 through R20).

Executes the complete experimental suite:
    1. R8/R15: Importance Validation & Error Coverage@K
    2. R9: Component Independence Correlation Matrix
    3. R10: Importance Normalization Ablation
    4. R11: Optimization Policy Benchmark (Quality vs Compute)
    5. R12: Densification Convergence Trajectories (t_90, t_95)
    6. R13: Closed-Loop Budget Controller & Latency Sweep
    7. R14: Per-Stage Pipeline Latency Profiling
    8. R16: Attribution Pixel Top-K Ablation
    9. R17: Oracle Utility & Marginal Value Framework
    10. R18: Uncertainty & Temporal Hysteresis Ablation
    11. R19: Matched-Budget Benchmark Matrix (Measured Compute)
    12. R20: Failure Mode Diagnostic Suite

Generates:
    - results/master/summary.json
    - results/master/summary.md
"""
import os
import sys
import subprocess
import argparse
import time
import json

EXPERIMENT_SCRIPTS = [
    ('R8/R15 Importance Validation', 'experiments/run_importance_validation.py'),
    ('R9 Component Independence', 'experiments/run_component_independence.py'),
    ('R10 Normalization Ablation', 'experiments/run_normalization_ablation.py'),
    ('R11 Policy Benchmark', 'experiments/run_policy_benchmark.py'),
    ('R12 Densification Convergence', 'experiments/run_densification_convergence.py'),
    ('R13 Budget Controller Sweep', 'experiments/run_budget_sweep.py'),
    ('R14 Pipeline Profiler', 'benchmarks/profile_pipeline.py'),
    ('R16 Top-K Attribution Ablation', 'experiments/run_attribution_topk_ablation.py'),
    ('R17 Oracle Utility Experiment', 'experiments/run_oracle_utility.py'),
    ('R18 Importance Formulation Ablation', 'experiments/run_oracle_ablation.py'),
    ('R19 Matched-Budget Benchmark', 'experiments/run_matched_budget.py'),
    ('R20 Failure Case Diagnostics', 'experiments/run_failure_analysis.py'),
]


def main():
    parser = argparse.ArgumentParser(description="Master Research Experiment Runner (R8-R20)")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=8, help='Number of frames per experiment')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    total_start = time.time()
    print("=" * 80)
    print("       ADAPTIVE 3D GAUSSIAN SPLATTING — COMPLETE EXPERIMENTAL SUITE (R8-R20)")
    print("=" * 80)
    print(f"Device: {args.device} | Frames per experiment: {args.frames}\n")

    summary_records = []

    for idx, (title, script_path) in enumerate(EXPERIMENT_SCRIPTS, 1):
        print("\n" + "#" * 80)
        print(f"[{idx}/{len(EXPERIMENT_SCRIPTS)}] RUNNING: {title} ({script_path})")
        print("#" * 80)
        
        full_script = os.path.join(project_root, script_path)
        cmd = [sys.executable, full_script, '--device', args.device, '--frames', str(args.frames)]
        
        start_t = time.time()
        ret = subprocess.run(cmd, cwd=project_root)
        elapsed = time.time() - start_t
        
        status = "PASSED" if ret.returncode == 0 else f"FAILED (code {ret.returncode})"
        summary_records.append({
            'milestone': title,
            'script': script_path,
            'status': status,
            'duration_s': round(elapsed, 2)
        })
        
        if ret.returncode != 0:
            print(f"\n❌ FAILED: {title} with exit code {ret.returncode}")
            sys.exit(ret.returncode)
        else:
            print(f"\n✅ COMPLETED: {title} in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"       ALL RESEARCH EXPERIMENTS (R8-R20) COMPLETED IN {total_elapsed:.1f}s")
    print("=" * 80)
    
    # Save Master Summary
    master_dir = os.path.join(project_root, 'results', 'master')
    os.makedirs(master_dir, exist_ok=True)
    
    summary_json_path = os.path.join(master_dir, 'summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump({
            'total_duration_s': round(total_elapsed, 2),
            'device': args.device,
            'frames_per_run': args.frames,
            'experiments': summary_records
        }, f, indent=2)
        
    summary_md_path = os.path.join(master_dir, 'summary.md')
    with open(summary_md_path, 'w') as f:
        f.write("# Master Research Experimental Suite Summary (R8–R20)\n\n")
        f.write(f"- **Total Execution Time**: {total_elapsed:.1f}s\n")
        f.write(f"- **Device**: `{args.device}`\n")
        f.write(f"- **Frames / Experiment**: `{args.frames}`\n\n")
        f.write("| Milestone | Script | Status | Duration (s) |\n")
        f.write("|:---|:---|:---:|:---:|\n")
        for rec in summary_records:
            f.write(f"| {rec['milestone']} | `{rec['script']}` | {rec['status']} | {rec['duration_s']} |\n")
        f.write("\n")
        
    print(f"\nMaster summaries saved to:")
    print(f"  - {summary_json_path}")
    print(f"  - {summary_md_path}")


if __name__ == '__main__':
    main()
