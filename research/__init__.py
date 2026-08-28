"""
Adaptive 3D Gaussian Splatting Research Package.
"""
from .gaussian_repr import GaussianModel
from .pipeline import OnlineReconstructionPipeline
from .scheduler import BudgetScheduler, OptimizationPolicy, estimate_gaussian_costs
from .importance import GaussianImportanceEstimator
from .attribution import (
    render_with_attribution,
    compute_gaussian_statistics,
    compute_projected_area,
    normalize_importance_components,
)
from .importance_diagnostics import (
    compute_full_diagnostics,
    format_diagnostics_report,
)
from .benchmark_policies import (
    run_policy_experiment,
    run_full_policy_ablation_matrix,
    format_benchmark_table,
)
from .benchmark_densification import (
    run_densification_experiment,
    run_full_densification_ablation,
    format_densification_table,
)
from .benchmark_budgets import (
    run_budget_experiment,
    run_full_budget_matrix,
    format_budget_table,
)
from .evaluation import (
    generate_table_1_main_benchmark,
    generate_table_2_ablation_study,
    generate_ascii_pareto_curve,
    generate_tier_distribution_chart,
    generate_hypothesis_verification_summary,
)
from .oracle_utility import OracleUtilityExperiment
from .uncertainty import GaussianUncertaintyEstimator
from .matched_budget_benchmark import MatchedBudgetBenchmark, SchedulerMetrics
from .failure_analysis import FailureCaseAnalyzer, FailureType, format_failure_analysis_report

__all__ = [
    'GaussianModel',
    'OnlineReconstructionPipeline',
    'BudgetScheduler',
    'OptimizationPolicy',
    'estimate_gaussian_costs',
    'GaussianImportanceEstimator',
    'render_with_attribution',
    'compute_gaussian_statistics',
    'compute_projected_area',
    'normalize_importance_components',
    'compute_full_diagnostics',
    'format_diagnostics_report',
    'run_policy_experiment',
    'run_full_policy_ablation_matrix',
    'format_benchmark_table',
    'run_densification_experiment',
    'run_full_densification_ablation',
    'format_densification_table',
    'run_budget_experiment',
    'run_full_budget_matrix',
    'format_budget_table',
    'generate_table_1_main_benchmark',
    'generate_table_2_ablation_study',
    'generate_ascii_pareto_curve',
    'generate_tier_distribution_chart',
    'generate_hypothesis_verification_summary',
    # v3.0 additions
    'OracleUtilityExperiment',
    'GaussianUncertaintyEstimator',
    'MatchedBudgetBenchmark',
    'SchedulerMetrics',
    'FailureCaseAnalyzer',
    'FailureType',
    'format_failure_analysis_report',
]
