"""Master Research Experiment Runner.

Executes all research benchmarks (R8 through R16):
    1. R8/R15: Importance Validation & Error Coverage@K
    2. R9: Component Independence Correlation Matrix
    3. R10: Importance Normalization Ablation
    4. R11: Optimization Policy Benchmark (Quality vs Compute)
    5. R12: Densification Convergence Trajectories (t_90, t_95)
    6. R13: Closed-Loop Budget Controller & Latency Sweep
    7. R14: Per-Stage Pipeline Latency Profiling
    8. R15/R16: Attribution Pixel Top-K Ablation
"""
import os
import sys
import subprocess
import argparse
import time

EXPERIMENT_SCRIPTS = [
    ('R8/R15 Importance Validation', 'experiments/run_importance_validation.py'),
    ('R9 Component Independence', 'experiments/run_component_independence.py'),
    ('R10 Normalization Ablation', 'experiments/run_normalization_ablation.py'),
    ('R11 Policy Benchmark', 'experiments/run_policy_benchmark.py'),
    ('R12 Densification Convergence', 'experiments/run_densification_convergence.py'),
    ('R13 Budget Controller Sweep', 'experiments/run_budget_sweep.py'),
    ('R14 Pipeline Profiler', 'benchmarks/profile_pipeline.py'),
    ('R15 Top-K Attribution Ablation', 'experiments/run_attribution_topk_ablation.py'),
]


def main():
    parser = argparse.ArgumentParser(description="Master Research Experiment Runner")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--frames', type=int, default=10, help='Number of frames per experiment')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    total_start = time.time()
    print("=" * 80)
    print("       ADAPTIVE 3D GAUSSIAN SPLATTING — MASTER EXPERIMENTAL SUITE")
    print("=" * 80)
    print(f"Device: {args.device} | Frames per experiment: {args.frames}\n")

    for idx, (title, script_path) in enumerate(EXPERIMENT_SCRIPTS, 1):
        print("\n" + "#" * 80)
        print(f"[{idx}/{len(EXPERIMENT_SCRIPTS)}] RUNNING: {title} ({script_path})")
        print("#" * 80)
        
        full_script = os.path.join(project_root, script_path)
        cmd = [sys.executable, full_script, '--device', args.device, '--frames', str(args.frames)]
        
        start_t = time.time()
        ret = subprocess.run(cmd, cwd=project_root)
        elapsed = time.time() - start_t
        
        if ret.returncode != 0:
            print(f"\n❌ FAILED: {title} with exit code {ret.returncode}")
            sys.exit(ret.returncode)
        else:
            print(f"\n✅ COMPLETED: {title} in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"       ALL RESEARCH EXPERIMENTS COMPLETED IN {total_elapsed:.1f}s")
    print("=" * 80)
    print("All JSON results saved to results/ and figure CSVs saved to results/figures/")


if __name__ == '__main__':
    main()
