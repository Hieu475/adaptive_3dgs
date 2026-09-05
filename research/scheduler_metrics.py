"""Phase 5 Scheduler Metrics & Diagnostics.

Provides standardized evaluation metrics for compute-budgeted Gaussian scheduling:
  - Oracle Selection Efficiency (OSE) with scientific hygiene (NaN for non-positive denominator)
  - Absolute and Relative Regret, and Selection Regret
  - Policy Efficiency (Realized Gain / Actual Cost in ms)
  - Cost Calibration: Absolute Error, MAPE_C, R2_C, and Hard Budget Violation (V_B)
  - Selection Churn and retention tracking across sequential frames
  - Memory overhead tracking (GPU VRAM / RAM)
  - Multi-seed statistical tests (95% Bootstrap CI, Wilcoxon Signed-Rank, Cohen's d)
"""
from typing import Dict, List, Optional, Tuple, Union, Set, Any
import numpy as np
import torch
from scipy.stats import wilcoxon


def compute_ose(delta_q_learned: float, delta_q_oracle: float) -> Optional[float]:
    """Computes Oracle Selection Efficiency: OSE(B) = Delta Q_learned(B) / Delta Q_oracle(B).
    
    Scientific Hygiene Rule (Point 18, Phase 5 IV/V):
      When Delta Q_oracle <= 0, denominator has no valid scientific reference.
      Returns None (JSON null / float('nan')) instead of an unprincipled fallback.
      OSE is interpreted as efficiency relative to oracle-guided reference policy,
      NOT as fraction of a combinatorial global optimum.
    """
    if delta_q_oracle is None or np.isnan(delta_q_oracle) or delta_q_oracle <= 0.0:
        return None
    return float(delta_q_learned / delta_q_oracle)


def compute_regret(
    delta_q_oracle: float,
    delta_q_learned: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Computes Absolute and Relative Regret (Point 19).
    
    Regret_abs = Delta Q_oracle - Delta Q_learned
    Regret_rel = (Delta Q_oracle - Delta Q_learned) / (|Delta Q_oracle| + eps)
    """
    regret_abs = float(delta_q_oracle - delta_q_learned)
    denom = abs(delta_q_oracle) + eps
    regret_rel = float(regret_abs / denom)
    return {
        "regret_abs": regret_abs,
        "regret_rel": regret_rel,
    }


def compute_selection_regret(delta_q_oracle: float, delta_q_policy: float) -> float:
    """Computes Selection Regret: SelectionRegret(B) = Q(S*_B) - Q(S_B)."""
    return float(delta_q_oracle - delta_q_policy)


def compute_policy_efficiency(
    delta_q_realized: float,
    actual_cost_ms: float,
    eps: float = 1e-6,
) -> float:
    """Computes Policy Efficiency (Point 20): Quality gain per millisecond of compute.
    
    Efficiency(B) = Delta Q_realized / C_actual
    Directly answers: how much reconstruction quality is created per ms of compute?
    """
    return float(delta_q_realized / max(eps, actual_cost_ms))


def compute_cost_metrics(
    actual_cost_ms: float,
    predicted_cost_ms: float,
    scheduled_cost_ms: float,
    budget_ms: float,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Computes cost error, MAPE_C, and hard budget violations (Points 8, 9).
    
    Cost Error = C_actual - C_pred
    MAPE_C = |C_actual - C_pred| / (C_actual + eps)
    V_B_sched = max(0, C_sched - B)
    V_B_wall = max(0, C_actual - B)
    """
    cost_error = float(actual_cost_ms - predicted_cost_ms)
    mape_c = float(abs(cost_error) / max(eps, actual_cost_ms))
    budget_violation_ms = float(max(0.0, actual_cost_ms - budget_ms))
    is_violation = bool(actual_cost_ms > budget_ms + 1e-4)
    scheduled_violation_ms = float(max(0.0, scheduled_cost_ms - budget_ms))
    is_scheduled_violation = bool(scheduled_cost_ms > budget_ms + 1e-4)

    return {
        "cost_error_ms": cost_error,
        "mape_c": mape_c,
        "budget_violation_ms": budget_violation_ms,
        "is_violation": is_violation,
        "scheduled_violation_ms": scheduled_violation_ms,
        "is_scheduled_violation": is_scheduled_violation,
    }


def compute_cost_calibration_metrics(
    actual_costs: np.ndarray,
    predicted_costs: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Computes MAE_C, MAPE_C, and R2_C across paired cost observations."""
    acts = np.asarray(actual_costs, dtype=np.float64)
    preds = np.asarray(predicted_costs, dtype=np.float64)
    if len(acts) == 0:
        return {"mae_c": 0.0, "mape_c": 0.0, "r2_c": 0.0}

    errors = np.abs(acts - preds)
    mae_c = float(np.mean(errors))
    mape_c = float(np.mean(errors / (np.abs(acts) + eps)) * 100.0)

    ss_res = np.sum((acts - preds) ** 2)
    ss_tot = np.sum((acts - np.mean(acts)) ** 2)
    r2_c = float(1.0 - (ss_res / (ss_tot + eps))) if ss_tot > eps else 0.0

    return {
        "mae_c": mae_c,
        "mape_c": mape_c,
        "r2_c": r2_c,
    }


def compute_selection_churn(
    current_selected_ids: Union[Set[int], List[int]],
    previous_selected_ids: Union[Set[int], List[int]],
) -> float:
    """Computes Selection Churn between sequential frames (Point 27).
    
    Churn_t = 1 - |S_t cap S_{t-1}| / |S_t cup S_{t-1}|
    Measures scheduling stability across frames.
    """
    s_curr = set(current_selected_ids)
    s_prev = set(previous_selected_ids)

    if not s_curr and not s_prev:
        return 0.0
    union_len = len(s_curr | s_prev)
    if union_len == 0:
        return 0.0
    inter_len = len(s_curr & s_prev)
    return float(1.0 - (inter_len / union_len))


def compute_extended_churn(
    current_selected_ids: Union[Set[int], List[int]],
    previous_selected_ids: Union[Set[int], List[int]],
) -> Dict[str, Any]:
    """Computes detailed churn, retained count, and new count (Section XX)."""
    s_curr = set(current_selected_ids)
    s_prev = set(previous_selected_ids)
    
    churn_val = compute_selection_churn(s_curr, s_prev)
    retained = len(s_curr & s_prev)
    new_sel = len(s_curr - s_prev)
    total_sel = len(s_curr)

    return {
        "selection_churn": churn_val,
        "selected_count": total_sel,
        "retained_count": retained,
        "new_selected_count": new_sel,
    }


def compute_memory_overhead(device: Optional[Union[str, torch.device]] = None) -> Dict[str, float]:
    """Measures current and peak GPU VRAM allocation in Megabytes (Point 28)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
        max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        return {
            "allocated_mb": float(allocated_mb),
            "reserved_mb": float(reserved_mb),
            "max_allocated_mb": float(max_allocated_mb),
        }
    return {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
    }


def bootstrap_ci_95(
    data: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes empirical 95% bootstrap confidence interval (Point 21)."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    alpha = (1.0 - ci) / 2.0 * 100.0
    boot_means = []
    rng = np.random.default_rng(seed)
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100.0 - alpha))


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Computes paired Cohen's d effect size on n paired observations."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    diff = g1 - g2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    if std_diff < 1e-8:
        return 0.0
    return float(mean_diff / std_diff)


def paired_wilcoxon_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Runs paired Wilcoxon signed-rank test on n paired observations (alternative: group1 > group2)."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    if len(g1) < 5 or np.allclose(g1, g2):
        return 0.0, 0.5
    try:
        res = wilcoxon(g1, g2, alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 0.5
