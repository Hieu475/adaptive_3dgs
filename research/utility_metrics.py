"""Scientific Metrics for Learned Utility Estimation (RQ1 & RQ2).

Addresses:
  - RQ1 (Prediction Fidelity):
      * Spearman rho(U*, U_hat)
      * Pearson r(U*, U_hat)
      * MAE(Delta Q_hat, Delta Q)
      * MAE(Delta T_hat, Delta T)
      * MAE(U_hat, U*)
      * Calibration (Quantile Calibration Error / Regression Slope)
  - RQ2 (Selection & Reconstruction Efficacy):
      * NDCG@k
      * Overlap@k (Jaccard / Set Intersection)
      * Regret@k (Gain_oracle - Gain_pred)
      * OSE@k (Optimization Selection Efficiency: Gain_pred / Gain_oracle)
      * Realized Delta Q@k
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Spearman correlation with guard against constant arrays."""
    if len(x) < 3 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    r_val = float(r) if not np.isnan(r) else 0.0
    p_val = float(p) if not np.isnan(p) else 1.0
    return r_val, p_val


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Pearson correlation with guard against constant arrays."""
    if len(x) < 3 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    r_val = float(r) if not np.isnan(r) else 0.0
    p_val = float(p) if not np.isnan(p) else 1.0
    return r_val, p_val


def compute_ndcg_at_k(pred_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k."""
    if k <= 0 or len(pred_scores) == 0:
        return 0.0
    k_eval = min(k, len(pred_scores))
    p_idx = np.argsort(-pred_scores)[:k_eval]
    i_idx = np.argsort(-true_scores)[:k_eval]
    
    min_val = min(0.0, float(np.min(true_scores)))
    rel = true_scores - min_val
    
    discounts = np.log2(np.arange(2, k_eval + 2))
    dcg = np.sum(rel[p_idx] / discounts)
    idcg = np.sum(rel[i_idx] / discounts)
    return float(dcg / (idcg + 1e-8)) if idcg > 0 else 1.0


def compute_calibration_metrics(pred_vals: np.ndarray, true_vals: np.ndarray, n_bins: int = 5) -> Dict[str, float]:
    """Computes quantile-based Expected Calibration Error (ECE) and linear calibration slope."""
    if len(pred_vals) < n_bins:
        return {"calibration_ece": 0.0, "calibration_slope": 1.0}

    # Binned quantile calibration error
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(pred_vals, quantiles)
    ece = 0.0
    
    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        mask = (pred_vals >= low) & (pred_vals <= high) if b == n_bins - 1 else (pred_vals >= low) & (pred_vals < high)
        if np.sum(mask) > 0:
            bin_pred_mean = np.mean(pred_vals[mask])
            bin_true_mean = np.mean(true_vals[mask])
            bin_weight = np.sum(mask) / len(pred_vals)
            ece += bin_weight * np.abs(bin_pred_mean - bin_true_mean)

    # Linear calibration slope: y = slope * x + intercept
    if np.std(pred_vals) > 1e-7:
        slope = float(np.cov(pred_vals, true_vals)[0, 1] / (np.var(pred_vals) + 1e-8))
    else:
        slope = 0.0

    return {
        "calibration_ece": float(ece),
        "calibration_slope": float(slope),
    }


def evaluate_rq1_prediction(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    pred_q: Optional[np.ndarray] = None,
    true_q: Optional[np.ndarray] = None,
    pred_t: Optional[np.ndarray] = None,
    true_t: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluates RQ1 (Prediction Fidelity): s_i(t) -> U_i*."""
    rho, p_val = safe_spearmanr(pred_u, oracle_u)
    r_val, _ = safe_pearsonr(pred_u, oracle_u)
    mae_u = float(np.mean(np.abs(pred_u - oracle_u)))
    
    metrics = {
        "spearman_rho": float(rho),
        "spearman_pval": float(p_val),
        "pearson_r": float(r_val),
        "mae_utility": mae_u,
    }

    if pred_q is not None and true_q is not None:
        metrics["mae_delta_q"] = float(np.mean(np.abs(pred_q - true_q)))
    if pred_t is not None and true_t is not None:
        metrics["mae_delta_t"] = float(np.mean(np.abs(pred_t - true_t)))

    calib = compute_calibration_metrics(pred_u, oracle_u)
    metrics.update(calib)
    return metrics


PROTOCOL_BUDGETS: Tuple[float, ...] = (0.10, 0.20, 0.40, 0.60, 0.80)


def compute_confidence_interval_95(std: float, n: int) -> float:
    """Computes half-width of 95% confidence interval: 1.96 * std / sqrt(n)."""
    if n <= 1 or np.isnan(std):
        return 0.0
    return float(1.96 * std / np.sqrt(n))


def rank_candidates(scores: np.ndarray) -> np.ndarray:
    """Returns candidate indices sorted descending by score."""
    return np.argsort(-scores)


def select_candidates(
    utility: Union[np.ndarray, torch.Tensor],
    cost: Union[np.ndarray, torch.Tensor],
    budget: float,
) -> Tuple[List[int], float]:
    """Greedy budget-constrained candidate selection for Phase 4 & Phase 5:
    
    argmax_{S} sum_{i in S} utility_i  s.t.  sum_{i in S} cost_i <= budget
    
    Sorts candidates descending by utility score, and selects items while
    cumulative cost <= budget.
    Returns (selected_indices, realized_cost).
    """
    if isinstance(utility, torch.Tensor):
        utility = utility.detach().cpu().numpy()
    if isinstance(cost, torch.Tensor):
        cost = cost.detach().cpu().numpy()

    ranked = rank_candidates(utility)
    selected = []
    cur_cost = 0.0
    for idx in ranked:
        c = float(cost[idx])
        if cur_cost + c <= budget + 1e-7:
            selected.append(int(idx))
            cur_cost += c
    return selected, cur_cost


select_under_budget = select_candidates


def evaluate_rq2_selection(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    delta_q: np.ndarray,
    costs: Optional[np.ndarray] = None,
    k_fractions: Tuple[float, ...] = PROTOCOL_BUDGETS,
) -> Dict[str, float]:
    """Evaluates RQ2 (Selection Efficiency): U_hat_i -> S_B across all protocol budgets."""
    n = len(pred_u)
    pred_ranks = rank_candidates(pred_u)
    oracle_ranks = rank_candidates(oracle_u)
    
    results = {}
    total_cost = float(costs.sum()) if costs is not None else float(n)

    for frac in k_fractions:
        pct_label = f"{int(frac * 100)}pct"
        k = max(1, int(n * frac))
        
        # 1. Top-k ranking selection
        top_pred = set(pred_ranks[:k].tolist())
        top_ora = set(oracle_ranks[:k].tolist())
        overlap = len(top_pred & top_ora) / k
        
        gain_pred = float(delta_q[list(top_pred)].sum())
        gain_ora = float(delta_q[list(top_ora)].sum())
        
        ose = float(gain_pred / (gain_ora + 1e-8)) if gain_ora > 0 else 1.0
        regret = float(gain_ora - gain_pred)
        ndcg = compute_ndcg_at_k(pred_u, oracle_u, k)
        
        results[f"ndcg_{pct_label}"] = float(ndcg)
        results[f"overlap_{pct_label}"] = float(overlap)
        results[f"ose_{pct_label}"] = float(ose)
        results[f"regret_{pct_label}"] = float(regret)
        results[f"realized_delta_q_{pct_label}"] = float(gain_pred)
        results[f"oracle_delta_q_{pct_label}"] = float(gain_ora)

        # 2. Budget-constrained selection (when costs provided)
        if costs is not None:
            budget_val = frac * total_cost
            sel_pred, c_pred = select_under_budget(pred_u, costs, budget_val)
            sel_ora, c_ora = select_under_budget(oracle_u, costs, budget_val)

            gain_bg_pred = float(delta_q[sel_pred].sum()) if len(sel_pred) > 0 else 0.0
            gain_bg_ora = float(delta_q[sel_ora].sum()) if len(sel_ora) > 0 else 0.0
            ose_bg = float(gain_bg_pred / (gain_bg_ora + 1e-8)) if gain_bg_ora > 0 else 1.0

            results[f"budget_ose_{pct_label}"] = float(ose_bg)
            results[f"budget_realized_delta_q_{pct_label}"] = float(gain_bg_pred)
            results[f"budget_oracle_delta_q_{pct_label}"] = float(gain_bg_ora)
            results[f"budget_cost_{pct_label}"] = float(c_pred)
        
    return results


def evaluate_utility_complete(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    delta_q: np.ndarray,
    pred_q: Optional[np.ndarray] = None,
    true_q: Optional[np.ndarray] = None,
    pred_t: Optional[np.ndarray] = None,
    true_t: Optional[np.ndarray] = None,
    costs: Optional[np.ndarray] = None,
    k_fractions: Tuple[float, ...] = PROTOCOL_BUDGETS,
) -> Dict[str, float]:
    """Full evaluation encompassing both RQ1 and RQ2 metrics."""
    res_rq1 = evaluate_rq1_prediction(
        pred_u=pred_u,
        oracle_u=oracle_u,
        pred_q=pred_q,
        true_q=true_q,
        pred_t=pred_t,
        true_t=true_t,
    )
    res_rq2 = evaluate_rq2_selection(
        pred_u=pred_u,
        oracle_u=oracle_u,
        delta_q=delta_q,
        costs=costs,
        k_fractions=k_fractions,
    )
    combined = dict(res_rq1)
    combined.update(res_rq2)
    return combined


def analyze_utility_prediction_failures(
    pred_u: np.ndarray,
    oracle_u: np.ndarray,
    X_features: np.ndarray,
    feature_names: List[str],
    strata: List[str],
    gaussian_ids: Optional[List[int]] = None,
    threshold_quantile: float = 0.10,
) -> Dict[str, Any]:
    """Diagnoses systematic failure modes of the learned utility estimator.
    
    Identifies:
      1. Over-predicted Gaussians: U_hat_i >> U_i* (top residual quantile)
      2. Under-predicted Gaussians: U_hat_i << U_i* (bottom residual quantile)
      3. Geometry stratum breakdown: flat, edge, texture, depth_discontinuity
      4. Feature profile anomalies: which Gaussian states drive systematic misprediction.
    """
    if isinstance(pred_u, torch.Tensor):
        pred_u = pred_u.detach().cpu().numpy()
    if isinstance(oracle_u, torch.Tensor):
        oracle_u = oracle_u.detach().cpu().numpy()
    if isinstance(X_features, torch.Tensor):
        X_features = X_features.detach().cpu().numpy()

    n = len(pred_u)
    residuals = pred_u - oracle_u
    abs_residuals = np.abs(residuals)

    q_high = (1.0 - threshold_quantile) * 100.0
    q_low = threshold_quantile * 100.0
    cutoff_high = float(np.percentile(residuals, q_high))
    cutoff_low = float(np.percentile(residuals, q_low))

    over_idx = np.where(residuals >= cutoff_high)[0].tolist()
    under_idx = np.where(residuals <= cutoff_low)[0].tolist()
    calib_idx = np.where((residuals > cutoff_low) & (residuals < cutoff_high))[0].tolist()

    # Strata breakdown
    unique_strata = sorted(list(set(strata)))
    strata_analysis = {}

    for st in unique_strata:
        st_indices = [i for i, s in enumerate(strata) if s == st]
        n_st = len(st_indices)
        if n_st == 0:
            continue

        st_over = [i for i in st_indices if i in over_idx]
        st_under = [i for i in st_indices if i in under_idx]
        st_res = residuals[st_indices]

        strata_analysis[st] = {
            "total_samples": n_st,
            "fraction_of_dataset": float(n_st / n),
            "n_over_predicted": len(st_over),
            "n_under_predicted": len(st_under),
            "over_prediction_rate": float(len(st_over) / n_st),
            "under_prediction_rate": float(len(st_under) / n_st),
            "mean_residual": float(np.mean(st_res)),
            "std_residual": float(np.std(st_res)),
            "mae_utility": float(np.mean(np.abs(st_res))),
            "mean_oracle_utility": float(np.mean(oracle_u[st_indices])),
            "mean_predicted_utility": float(np.mean(pred_u[st_indices])),
        }

    # Feature profile comparison across regimes
    feature_profiles = {}
    for f_idx, f_name in enumerate(feature_names):
        f_vals = X_features[:, f_idx]
        mean_all = float(np.mean(f_vals))
        std_all = float(np.std(f_vals)) + 1e-7

        mean_over = float(np.mean(f_vals[over_idx])) if len(over_idx) > 0 else mean_all
        mean_under = float(np.mean(f_vals[under_idx])) if len(under_idx) > 0 else mean_all
        mean_calib = float(np.mean(f_vals[calib_idx])) if len(calib_idx) > 0 else mean_all

        # Standardized z-shift
        z_shift_over = (mean_over - mean_all) / std_all
        z_shift_under = (mean_under - mean_all) / std_all

        feature_profiles[f_name] = {
            "mean_overall": mean_all,
            "mean_over_predicted": mean_over,
            "mean_under_predicted": mean_under,
            "mean_calibrated": mean_calib,
            "z_shift_over_predicted": float(z_shift_over),
            "z_shift_under_predicted": float(z_shift_under),
        }

    # Identify primary drivers of misprediction
    sorted_by_over_shift = sorted(
        feature_profiles.items(),
        key=lambda x: abs(x[1]["z_shift_over_predicted"]),
        reverse=True,
    )
    sorted_by_under_shift = sorted(
        feature_profiles.items(),
        key=lambda x: abs(x[1]["z_shift_under_predicted"]),
        reverse=True,
    )

    key_over_drivers = [
        {"feature": name, "z_shift": data["z_shift_over_predicted"]}
        for name, data in sorted_by_over_shift[:3]
    ]
    key_under_drivers = [
        {"feature": name, "z_shift": data["z_shift_under_predicted"]}
        for name, data in sorted_by_under_shift[:3]
    ]

    return {
        "n_samples": n,
        "threshold_quantile": threshold_quantile,
        "cutoff_over_predicted": cutoff_high,
        "cutoff_under_predicted": cutoff_low,
        "counts": {
            "n_over_predicted": len(over_idx),
            "n_under_predicted": len(under_idx),
            "n_calibrated": len(calib_idx),
        },
        "strata_analysis": strata_analysis,
        "feature_profiles": feature_profiles,
        "systematic_drivers": {
            "over_prediction_drivers": key_over_drivers,
            "under_prediction_drivers": key_under_drivers,
        },
    }
