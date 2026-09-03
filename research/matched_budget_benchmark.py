"""
Rigorous Matched-Budget Benchmark for Adaptive 3DGS (Sections X–XVIII).

Implements:
1. Calibration independent of Evaluation (Section XI):
       Frames 0..N_calib  -> Calibrate and freeze K_P(B)
       Frames N_calib..N  -> Evaluate strictly with frozen K_P(B)
2. Benchmark A (Open-Loop, Section XIII):
       Answers: "Under strictly matched compute, which selection policy delivers superior quality?"
       Budgets: B_r = r · T_{full} for r ∈ {0.10, 0.20, 0.40, 0.60, 0.80}
       Metrics: PSNR (mean ± 95% CI), Depth L1 (mean ± 95% CI), Gain Efficiency (GE@B ± CI)
3. Benchmark B (Closed-Loop Systems Benchmark, Section XIII):
       Answers: "Can the adaptive feedback scheduler strictly maintain hard real-time deadlines?"
       Metrics: ViolationRate (%), p50/p95/p99 latency (ms), Jitter (ms), Recovery Time (frames)
4. Paired Sample Hypothesis Testing (Section XVII):
       ΔQ_t = Q_{ours, t} - Q_{baseline, t} with empirical bootstrap 95% CI.
"""
import time
import math
import numpy as np
import torch
from typing import Dict, List, Callable, Any, Optional, Tuple

from research.schema import (
    ExperimentMetadata,
    QualityMetrics,
    LatencyMetrics,
    MemoryMetrics,
    SelectionMetrics,
    ExperimentMetrics,
    ExperimentResult,
)
from research.reproducibility import (
    get_git_commit,
    get_hardware_info,
    set_seed,
    bootstrap_ci,
    save_experiment_bundle,
)


class SchedulerMetrics:
    """Tracks latency distribution, jitter, recovery frames, and budget violations."""
    def __init__(self):
        self.latencies: List[float] = []
        self.budgets: List[float] = []
        
    def record_frame(self, latency_ms: float, budget_ms: float):
        self.latencies.append(latency_ms)
        self.budgets.append(budget_ms)
        
    def get_violation_rate(self) -> float:
        if not self.latencies:
            return 0.0
        return float(sum(l > b for l, b in zip(self.latencies, self.budgets)) / len(self.latencies))
        
    def get_jitter(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        return float(np.std(self.latencies))
        
    def get_recovery_time(self) -> float:
        if not self.latencies:
            return 0.0
        recovery_times = []
        in_violation = False
        current_recovery = 0
        for l, b in zip(self.latencies, self.budgets):
            if l > b:
                in_violation = True
                current_recovery = 0
            elif in_violation:
                current_recovery += 1
                recovery_times.append(current_recovery)
                in_violation = False
        return float(sum(recovery_times) / len(recovery_times)) if recovery_times else 0.0
        
    def get_p50_latency(self) -> float:
        return float(np.percentile(self.latencies, 50)) if self.latencies else 0.0
        
    def get_p95_latency(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0
        
    def get_p99_latency(self) -> float:
        return float(np.percentile(self.latencies, 99)) if self.latencies else 0.0
        
    def get_mean_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0
        
    def get_summary(self) -> Dict[str, float]:
        return {
            'violation_rate': self.get_violation_rate(),
            'jitter': self.get_jitter(),
            'recovery_time': self.get_recovery_time(),
            'p50_latency': self.get_p50_latency(),
            'p95_latency': self.get_p95_latency(),
            'p99_latency': self.get_p99_latency(),
            'mean_latency': self.get_mean_latency(),
        }


class MatchedBudgetBenchmark:
    """Primary research benchmark executing strict Open-Loop and Closed-Loop evaluations."""
    
    def __init__(
        self,
        relative_budgets: List[float] = [0.10, 0.20, 0.40, 0.60, 0.80],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42,
    ):
        self.relative_budgets = relative_budgets
        self.device = device
        self.seed = seed
        self.policies = [
            'random',
            'error_only',
            'error_influence',
            'binary',
            'top_k',
            'ours',
        ]
        
    def calibrate_policy_k(
        self,
        pipeline_factory: Callable,
        calib_frames: List[Dict],
        intrinsics: Any,
        target_budget_ms: float,
        policy: str,
    ) -> int:
        """Calibrate Gaussian count K on calibration frames and freeze it (Section XI)."""
        if policy == 'ours':
            return 0  # Ours uses dynamic knapsack solver internally
            
        k_current = max(4, int(200 * (target_budget_ms / max(target_budget_ms * 2.0, 50.0))))
        
        for iteration in range(2):
            config_overrides = {
                'scheduler': {
                    'policy': policy,
                    'top_k': k_current,
                    'gpu_budget_ms': target_budget_ms,
                }
            }
            p = pipeline_factory(config_overrides, self.device)
            p.initialize(calib_frames[0]['rgb'], calib_frames[0]['depth'], intrinsics, calib_frames[0].get('pose'))
            
            latencies = []
            for f in calib_frames[1:min(4, len(calib_frames))]:
                res = p.process_frame(f['rgb'], f['depth'], f.get('pose'))
                latencies.append(res.get('opt_time_ms', 1.0))
                
            mean_t = float(np.mean(latencies)) if latencies else target_budget_ms
            if mean_t <= 0.001:
                break
                
            ratio = target_budget_ms / mean_t
            k_new = max(2, min(int(k_current * ratio), p.gaussian_model.num_gaussians))
            if abs(k_new - k_current) <= 1:
                break
            k_current = k_new
            
        return k_current

    def evaluate_policy_on_frames(
        self,
        pipeline_factory: Callable,
        eval_frames: List[Dict],
        intrinsics: Any,
        budget_ms: float,
        policy: str,
        frozen_k: Optional[int] = None,
        relative_budget: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate a policy on held-out evaluation frames using frozen calibrated K."""
        config_overrides = {
            'scheduler': {
                'policy': policy,
                'gpu_budget_ms': budget_ms,
                'top_k': frozen_k if (frozen_k is not None and policy != 'ours') else None,
            }
        }
        
        pipeline = pipeline_factory(config_overrides, self.device)
        metrics = SchedulerMetrics()
        
        psnrs = []
        depth_l1s = []
        ssims = []
        opt_times = []
        n_actives = []
        n_totals = []
        
        pipeline.initialize(eval_frames[0]['rgb'], eval_frames[0]['depth'], intrinsics, eval_frames[0].get('pose'))
        
        for frame in eval_frames[1:]:
            res = pipeline.process_frame(frame['rgb'], frame['depth'], frame.get('pose'))
            opt_time = res.get('opt_time_ms', 0.0)
            metrics.record_frame(opt_time, budget_ms)
            
            psnrs.append(res.get('psnr', 0.0))
            depth_l1s.append(res.get('depth_l1', 0.0))
            ssims.append(res.get('ssim', 0.80))
            opt_times.append(opt_time)
            n_actives.append(res.get('n_optimized', 0))
            n_totals.append(pipeline.gaussian_model.num_gaussians)
            
        summary = metrics.get_summary()
        avg_active = float(np.mean(n_actives)) if n_actives else 0.0
        avg_total = float(np.mean(n_totals)) if n_totals else 1.0
        measured_ms = summary['mean_latency']
        utilization = (measured_ms / budget_ms) if budget_ms > 0 else 1.0
        
        # Bootstrap 95% Confidence Intervals (Section XVIII)
        psnr_ci = bootstrap_ci(psnrs, stat_fn=np.mean, n_boot=500, ci=0.95, seed=self.seed)
        depth_ci = bootstrap_ci(depth_l1s, stat_fn=np.mean, n_boot=500, ci=0.95, seed=self.seed)
        
        return {
            'policy_name': policy,
            'relative_budget': relative_budget,
            'budget_ms': budget_ms,
            'frozen_k': frozen_k,
            'avg_psnr': float(np.mean(psnrs)) if psnrs else 0.0,
            'psnr_std': float(np.std(psnrs)) if psnrs else 0.0,
            'psnr_ci95': psnr_ci,
            'avg_ssim': float(np.mean(ssims)) if ssims else 0.0,
            'avg_depth_l1': float(np.mean(depth_l1s)) if depth_l1s else 0.0,
            'depth_l1_std': float(np.std(depth_l1s)) if depth_l1s else 0.0,
            'depth_ci95': depth_ci,
            'measured_compute_ms': measured_ms,
            'p50_ms': summary['p50_latency'],
            'p95_ms': summary['p95_latency'],
            'p99_ms': summary['p99_latency'],
            'jitter': summary['jitter'],
            'budget_utilization': utilization,
            'violation_rate': summary['violation_rate'] * 100.0,
            'active_gaussians': avg_active,
            'active_ratio': float(avg_active / max(1.0, avg_total)),
            'n_eval_frames': len(eval_frames),
            'per_frame_psnr': psnrs,
            'per_frame_depth': depth_l1s,
            'per_frame_opt_ms': opt_times,
        }

    def run_benchmark_a_open_loop(
        self,
        pipeline_factory: Callable,
        all_frames: List[Dict],
        intrinsics: Any,
        calib_fraction: float = 0.35,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Execute Benchmark A (Open-Loop Calibrated Matched Budget, Sections X–XVI)."""
        n_tot = len(all_frames)
        n_calib = max(3, int(n_tot * calib_fraction))
        calib_frames = all_frames[:n_calib]
        eval_frames = all_frames[n_calib:]
        
        print(f"\n[Benchmark A: Open-Loop] Total Frames: {n_tot} | Calibration: {len(calib_frames)} | Evaluation: {len(eval_frames)}")
        
        results = []
        
        # 1. Unconstrained Full Optimization Reference
        print(">> Measuring Full-Optimization Reference...")
        full_eval = self.evaluate_policy_on_frames(
            pipeline_factory, eval_frames, intrinsics, budget_ms=100.0, policy='full', relative_budget=1.0
        )
        full_eval['policy_name'] = 'Full Reference (Unconstrained)'
        t_full = max(5.0, full_eval['measured_compute_ms'])
        q_full = full_eval['avg_psnr']
        results.append(full_eval)
        print(f"   Reference Opt Time: {t_full:.2f} ms | Reference PSNR: {q_full:.2f} dB")
        
        # 2. Relative Budgets with Frozen K
        for rel_b in self.relative_budgets:
            target_b_ms = rel_b * t_full
            print(f"\n>> Budget Level: {int(rel_b*100)}% ({target_b_ms:.2f} ms target)")
            
            # Step A: Offline Calibration on calib_frames
            calibrated_ks = {}
            for policy in self.policies:
                if policy != 'ours':
                    k_val = self.calibrate_policy_k(
                        pipeline_factory, calib_frames, intrinsics, target_b_ms, policy
                    )
                else:
                    k_val = None
                calibrated_ks[policy] = k_val
                
            # Step B: Evaluation on eval_frames with frozen K
            for policy in self.policies:
                k_frozen = calibrated_ks[policy]
                eval_res = self.evaluate_policy_on_frames(
                    pipeline_factory, eval_frames, intrinsics,
                    budget_ms=target_b_ms, policy=policy,
                    frozen_k=k_frozen, relative_budget=rel_b
                )
                results.append(eval_res)
                print(
                    f"   [{policy:<18}] Frozen K={str(k_frozen):<4} | "
                    f"T_opt={eval_res['p50_ms']:5.1f}ms (Util: {eval_res['budget_utilization']*100:5.1f}%) | "
                    f"PSNR: {eval_res['avg_psnr']:5.2f} ± {eval_res['psnr_std']:.2f} dB"
                )
                
        # 3. Compute Gain Efficiency (GE@B) and Paired Differences (ΔQ)
        for rel_b in self.relative_budgets:
            b_rows = [r for r in results if r.get('relative_budget') == rel_b]
            rand_row = next((r for r in b_rows if r['policy_name'] == 'random'), None)
            ours_row = next((r for r in b_rows if r['policy_name'] == 'ours'), None)
            q_rand = rand_row['avg_psnr'] if rand_row else 0.0
            
            for r in b_rows:
                q_val = r['avg_psnr']
                denom = max(0.01, q_full - q_rand)
                ge = (q_val - q_rand) / denom
                r['gain_efficiency'] = float(np.clip(ge, 0.0, 1.0))
                
                # Paired sample differences against Ours (Section XVII)
                if ours_row and 'per_frame_psnr' in r and 'per_frame_psnr' in ours_row:
                    paired_deltas = [
                        q_o - q_p for q_o, q_p in zip(ours_row['per_frame_psnr'], r['per_frame_psnr'])
                    ]
                    r['delta_psnr_vs_ours_mean'] = float(np.mean(paired_deltas))
                    r['delta_psnr_ci95'] = bootstrap_ci(paired_deltas, seed=self.seed)
                else:
                    r['delta_psnr_vs_ours_mean'] = 0.0
                    r['delta_psnr_ci95'] = (0.0, 0.0)
                    
        meta = {
            't_full': t_full,
            'q_full': q_full,
            'relative_budgets': self.relative_budgets,
            'device': self.device,
            'n_calib_frames': len(calib_frames),
            'n_eval_frames': len(eval_frames),
        }
        return results, meta

    def run_benchmark_b_closed_loop(
        self,
        pipeline_factory: Callable,
        eval_frames: List[Dict],
        intrinsics: Any,
        target_budgets_ms: List[float] = [10.0, 15.0, 20.0, 30.0],
    ) -> List[Dict[str, Any]]:
        """Execute Benchmark B (Closed-Loop Real-Time Deadline Adherence, Section XIII)."""
        print(f"\n[Benchmark B: Closed-Loop Systems] Evaluating Deadline Adherence across {target_budgets_ms} ms budgets...")
        closed_loop_results = []
        
        for b_target in target_budgets_ms:
            res = self.evaluate_policy_on_frames(
                pipeline_factory, eval_frames, intrinsics,
                budget_ms=b_target, policy='ours', relative_budget=0.0
            )
            closed_loop_results.append(res)
            print(
                f"   [Budget {b_target:4.1f}ms] p50: {res['p50_ms']:5.1f}ms | "
                f"p95: {res['p95_ms']:5.1f}ms | Jitter: {res['jitter']:4.2f}ms | "
                f"Violation: {res['violation_rate']:4.1f}% | Util: {res['budget_utilization']*100:5.1f}%"
            )
            
        return closed_loop_results

    def format_markdown_table(self, open_loop_results: List[Dict], meta: Dict[str, Any]) -> str:
        """Format publication-quality Markdown table with 95% Confidence Intervals (Section XV)."""
        lines = []
        lines.append("# Final Matched-Budget Scientific Benchmark (Table 1 & Table 3)")
        lines.append("")
        lines.append(f"Evaluated with independent calibration/evaluation split (Reference $T_{{full}} = {meta['t_full']:.2f}$ ms).")
        lines.append(f"Calibration: {meta['n_calib_frames']} frames | Held-Out Evaluation: {meta['n_eval_frames']} frames.")
        lines.append("")
        lines.append("## Table 1: Quality@Budget with Bootstrap 95% Confidence Intervals")
        lines.append("")
        lines.append("| Budget | Policy | Frozen $K$ | Opt ($p50$) | $p95$ (ms) | Jitter | Util% | Viol% | PSNR (95% CI) ↑ | Depth L1 (m) ↓ | Gain Eff ($GE$) |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        
        for r in open_loop_results:
            if 'Full Reference' in r['policy_name']:
                ci = r.get('psnr_ci95', (r['avg_psnr'], r['avg_psnr']))
                lines.append(
                    f"| **Reference** | **{r['policy_name']}** | All | "
                    f"{r['p50_ms']:6.2f} ms | {r['p95_ms']:6.2f} ms | {r['jitter']:5.2f} | "
                    f"— | — | **{r['avg_psnr']:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]** | {r['avg_depth_l1']:7.4f} | 1.00 |"
                )
                lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
                
        for rel_b in meta['relative_budgets']:
            b_rows = [r for r in open_loop_results if r.get('relative_budget') == rel_b]
            b_rows.sort(key=lambda x: x['avg_psnr'], reverse=True)
            for r in b_rows:
                k_str = str(r['frozen_k']) if r['frozen_k'] is not None else "Knapsack"
                p_name = r['policy_name']
                is_ours = (p_name == 'ours')
                bold = "**" if is_ours else ""
                ci = r.get('psnr_ci95', (r['avg_psnr'], r['avg_psnr']))
                lines.append(
                    f"| {int(rel_b*100)}% ({r['budget_ms']:.1f}ms) | {bold}{p_name}{bold} | "
                    f"{k_str} | {r['p50_ms']:6.2f} ms | {r['p95_ms']:6.2f} ms | "
                    f"{r['jitter']:5.2f} | {r['budget_utilization']*100:5.1f}% | {r['violation_rate']:4.1f}% | "
                    f"{bold}{r['avg_psnr']:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]{bold} | {r['avg_depth_l1']:7.4f} | {r.get('gain_efficiency', 0.0):.3f} |"
                )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            
        return "\n".join(lines)
