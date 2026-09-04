"""Unit tests for Phase 4 Utility Metrics (RQ1 & RQ2)."""
import pytest
import numpy as np
import torch

from research.utility_metrics import (
    PROTOCOL_BUDGETS,
    safe_spearmanr,
    safe_pearsonr,
    compute_ndcg_at_k,
    compute_calibration_metrics,
    compute_confidence_interval_95,
    rank_candidates,
    select_candidates,
    select_under_budget,
    evaluate_rq1_prediction,
    evaluate_rq2_selection,
    evaluate_utility_complete,
)


def test_protocol_budgets_definition():
    assert PROTOCOL_BUDGETS == (0.10, 0.20, 0.40, 0.60, 0.80)


def test_safe_correlations():
    # Identical monotonic arrays
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    r_rho, p_rho = safe_spearmanr(x, y)
    assert np.isclose(r_rho, 1.0)
    assert p_rho < 0.05

    r_p, _ = safe_pearsonr(x, y)
    assert np.isclose(r_p, 1.0)

    # Constant array guard
    c = np.array([2.0, 2.0, 2.0, 2.0])
    r_c, _ = safe_spearmanr(c, y[:4])
    assert r_c == 0.0


def test_ndcg_computation():
    true_scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    pred_perfect = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    pred_reverse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    ndcg_perf = compute_ndcg_at_k(pred_perfect, true_scores, k=3)
    assert np.isclose(ndcg_perf, 1.0)

    ndcg_rev = compute_ndcg_at_k(pred_reverse, true_scores, k=3)
    assert ndcg_rev < 1.0


def test_select_candidates_budget_constraint():
    utility = np.array([0.9, 0.8, 0.7, 0.1])
    costs = np.array([20.0, 15.0, 10.0, 5.0])
    budget = 30.0

    # Highest utility is idx 0 (cost 20), then idx 1 (cost 15, but 20+15=35 > 30), then idx 2 (cost 10, 20+10=30 <= 30)
    selected, real_cost = select_candidates(utility, costs, budget)
    assert selected == [0, 2]
    assert real_cost == 30.0
    assert real_cost <= budget

    # Works identically with torch tensors
    u_t = torch.from_numpy(utility)
    c_t = torch.from_numpy(costs)
    sel_t, real_c_t = select_candidates(u_t, c_t, budget)
    assert sel_t == [0, 2]
    assert real_c_t == 30.0

    # Alias check
    assert select_under_budget is select_candidates


def test_confidence_interval_95():
    ci = compute_confidence_interval_95(std=0.05, n=5)
    assert np.isclose(ci, 1.96 * 0.05 / np.sqrt(5))


def test_rq1_prediction_evaluation():
    pred_u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    oracle_u = np.array([0.12, 0.18, 0.31, 0.42, 0.49])
    pred_q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    true_q = np.array([1.05, 1.95, 3.02, 4.01, 4.98])
    pred_t = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    true_t = np.array([51.0, 59.0, 71.0, 79.0, 91.0])

    m = evaluate_rq1_prediction(
        pred_u=pred_u,
        oracle_u=oracle_u,
        pred_q=pred_q,
        true_q=true_q,
        pred_t=pred_t,
        true_t=true_t,
    )
    assert "spearman_rho" in m
    assert m["spearman_rho"] > 0.95
    assert "mae_delta_q" in m
    assert "mae_delta_t" in m
    assert "mae_utility" in m
    assert "calibration_ece" in m
    assert "calibration_slope" in m


def test_rq2_selection_evaluation_all_budgets():
    n = 100
    rng = np.random.default_rng(42)
    oracle_u = rng.uniform(0.01, 1.0, size=n)
    pred_u = oracle_u + rng.normal(0, 0.05, size=n)
    delta_q = oracle_u * rng.uniform(10, 50, size=n)
    costs = rng.uniform(5, 20, size=n)

    m = evaluate_rq2_selection(
        pred_u=pred_u,
        oracle_u=oracle_u,
        delta_q=delta_q,
        costs=costs,
        k_fractions=PROTOCOL_BUDGETS,
    )

    for frac in PROTOCOL_BUDGETS:
        pct = f"{int(frac * 100)}pct"
        assert f"ndcg_{pct}" in m
        assert f"overlap_{pct}" in m
        assert f"ose_{pct}" in m
        assert f"regret_{pct}" in m
        assert f"realized_delta_q_{pct}" in m
        assert f"budget_ose_{pct}" in m
        assert f"budget_realized_delta_q_{pct}" in m


def test_analyze_utility_prediction_failures():
    from research.utility_metrics import analyze_utility_prediction_failures
    n = 50
    rng = np.random.default_rng(123)
    u_true = rng.uniform(0.01, 1.0, size=n)
    u_pred = u_true + rng.normal(0, 0.2, size=n)
    X = rng.normal(0, 1, size=(n, 11))
    feature_names = [f"feat_{i}" for i in range(11)]
    strata = [["flat", "edge", "texture", "depth_discontinuity"][i % 4] for i in range(n)]

    res = analyze_utility_prediction_failures(
        pred_u=u_pred,
        oracle_u=u_true,
        X_features=X,
        feature_names=feature_names,
        strata=strata,
        threshold_quantile=0.10,
    )
    assert res["n_samples"] == n
    assert "counts" in res
    assert res["counts"]["n_over_predicted"] > 0
    assert res["counts"]["n_under_predicted"] > 0
    assert "strata_analysis" in res
    for st in ["flat", "edge", "texture", "depth_discontinuity"]:
        assert st in res["strata_analysis"]
        assert "mae_utility" in res["strata_analysis"][st]
    assert "feature_profiles" in res
    assert len(res["feature_profiles"]) == 11
    assert "systematic_drivers" in res
