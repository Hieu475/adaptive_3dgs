"""
Matched-budget benchmark for comparing 3DGS policies under identical compute constraints.
"""
import torch
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

class SchedulerMetrics:
    """
    Metrics for tracking budget scheduler performance.
    """
    def __init__(self):
        self.latencies = []
        self.budgets = []
        
    def record_frame(self, latency_ms: float, budget_ms: float):
        """Record the measured latency and target budget for a frame."""
        self.latencies.append(latency_ms)
        self.budgets.append(budget_ms)
        
    def get_violation_rate(self) -> float:
        """Calculate the rate of frames that exceeded the budget."""
        if not self.latencies:
            return 0.0
        violations = sum(l > b for l, b in zip(self.latencies, self.budgets))
        return violations / len(self.latencies)
        
    def get_jitter(self) -> float:
        """Calculate the standard deviation of latency."""
        if len(self.latencies) < 2:
            return 0.0
        return float(np.std(self.latencies))
        
    def get_recovery_time(self) -> float:
        """Calculate the average number of frames to return to budget after a violation."""
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
                if current_recovery > 0:
                    recovery_times.append(current_recovery)
                    in_violation = False
                    
        if not recovery_times:
            return 0.0
        return sum(recovery_times) / len(recovery_times)
        
    def get_p95_latency(self) -> float:
        """Calculate the 95th percentile latency."""
        if not self.latencies:
            return 0.0
        return float(np.percentile(self.latencies, 95))
        
    def get_summary(self) -> Dict[str, float]:
        """Get a summary dictionary of all metrics."""
        return {
            'violation_rate': self.get_violation_rate(),
            'jitter': self.get_jitter(),
            'recovery_time': self.get_recovery_time(),
            'p95_latency': self.get_p95_latency(),
            'mean_latency': float(np.mean(self.latencies)) if self.latencies else 0.0
        }

class MatchedBudgetBenchmark:
    """
    Primary benchmark for comparing policies under identical compute budgets.
    """
    def __init__(self, budget_levels_ms=[2.0, 5.0, 10.0, 20.0], device='cpu'):
        self.budget_levels_ms = budget_levels_ms
        self.device = device
        self.policies = ['full', 'random', 'error_only', 'error_influence', 'binary', 'top_k', 'ours']
        
    def run_single_budget(self, pipeline_factory: Callable, frames: List, intrinsics: Any, budget_ms: float, policy: str, **kwargs) -> Dict:
        """
        Run a single budget configuration for a specific policy.
        """
        config_overrides = {
            'scheduler': {
                'policy': policy,
                'gpu_budget_ms': budget_ms
            }
        }
        
        pipeline = pipeline_factory(config_overrides, self.device)
        metrics = SchedulerMetrics()
        
        psnrs = []
        depth_l1s = []
        opt_times = []
        
        # Initialize pipeline with the first frame
        pipeline.initialize(frames[0]['rgb'], frames[0]['depth'], intrinsics, frames[0].get('pose'))
        
        is_constrained = (policy != 'full')
        
        for frame in frames[1:]:
            res = pipeline.process_frame(frame['rgb'], frame['depth'], frame.get('pose'))
            opt_time = res.get('opt_time_ms', 0)
            metrics.record_frame(opt_time, budget_ms)
            
            psnrs.append(res.get('psnr', 0))
            depth_l1s.append(res.get('depth_l1', 0))
            opt_times.append(opt_time)
            
        summary = metrics.get_summary()
        
        return {
            'avg_psnr': np.mean(psnrs) if psnrs else 0.0,
            'avg_depth_l1': np.mean(depth_l1s) if depth_l1s else 0.0,
            'measured_compute_ms': np.mean(opt_times) if opt_times else 0.0,
            'budget_ms': budget_ms,
            'n_frames': len(frames),
            'budget_violation_rate': summary['violation_rate'],
            'jitter': summary['jitter'],
            'recovery_frames': summary['recovery_time'],
            'policy_name': policy,
            'budget_constrained': "Yes" if is_constrained else "No (Upper Bound)"
        }
        
    def run_full_matrix(self, pipeline_factory: Callable, frames: List, intrinsics: Any) -> List[Dict]:
        """
        Run all policies across all budget levels.
        """
        results = []
        for policy in self.policies:
            for budget in self.budget_levels_ms:
                res = self.run_single_budget(pipeline_factory, frames, intrinsics, budget, policy)
                results.append(res)
        return results
        
    def compute_quality_at_budget(self, results_matrix: List[Dict]) -> Dict:
        """
        Group and rank results by budget level.
        """
        ranked_results = {}
        for budget in self.budget_levels_ms:
            budget_res = [r for r in results_matrix if r['budget_ms'] == budget]
            # Rank budget-constrained policies by PSNR
            budget_res.sort(key=lambda x: (x['budget_constrained'] != 'No (Upper Bound)', x['avg_psnr']), reverse=True)
            ranked_results[budget] = budget_res
        return ranked_results
        
    def format_table(self, results_matrix: List[Dict]) -> str:
        """
        Format the results matrix into a markdown table.
        """
        ranked = self.compute_quality_at_budget(results_matrix)
        
        lines = []
        lines.append("| Budget (ms) | Policy | Budget Constrained? | PSNR ↑ | Depth L1 ↓ | Violations ↓ | Jitter ↓ | Compute (ms) |")
        lines.append("|-------------|--------|---------------------|--------|------------|--------------|----------|--------------|")
        
        for budget in self.budget_levels_ms:
            for res in ranked[budget]:
                lines.append(
                    f"| {budget:11.1f} | {res['policy_name']:15} | {res['budget_constrained']:19} | "
                    f"{res['avg_psnr']:6.2f} | {res['avg_depth_l1']:10.4f} | "
                    f"{res['budget_violation_rate'] * 100:10.1f}% | "
                    f"{res['jitter']:8.2f} | {res['measured_compute_ms']:12.2f} |"
                )
        return "\n".join(lines)
