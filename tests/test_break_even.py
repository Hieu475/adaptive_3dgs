"""Tests for break-even ratio calculation.

The break-even point r* is defined as:
  r* = max{r : Speedup(r) >= 1.0}

This means r* is the LARGEST active ratio where selective optimization is still beneficial.
"""

from typing import Any, Dict, List
import pytest


def compute_break_even(results: List[Dict[str, Any]]) -> float:
    """Compute the break-even active ratio r* = max{r : Speedup(r) >= 1.0}.

    Args:
        results: List of dicts, each with keys 'active_ratio' and 'opt_speedup'.

    Returns:
        float: The largest active_ratio where opt_speedup >= 1.0, or 0.0 if no ratio
               achieves speedup >= 1.0.
    """
    valid_ratios = [
        r['active_ratio'] for r in results if r.get('opt_speedup', 0.0) >= 1.0
    ]
    if not valid_ratios:
        return 0.0
    return float(max(valid_ratios))


def test_simple_break_even():
    """ratios = [1.0, 0.75, 0.50, 0.25, 0.10], speedups = [0.91, 1.02, 1.26, 2.31, 6.14] -> r* = 0.75."""
    ratios = [1.0, 0.75, 0.50, 0.25, 0.10]
    speedups = [0.91, 1.02, 1.26, 2.31, 6.14]
    results = [{'active_ratio': r, 'opt_speedup': s} for r, s in zip(ratios, speedups)]
    r_star = compute_break_even(results)
    assert r_star == pytest.approx(0.75)


def test_all_speedup():
    """ratios = [1.0, 0.50, 0.10], speedups = [1.02, 1.72, 19.33] -> r* = 1.0."""
    ratios = [1.0, 0.50, 0.10]
    speedups = [1.02, 1.72, 19.33]
    results = [{'active_ratio': r, 'opt_speedup': s} for r, s in zip(ratios, speedups)]
    r_star = compute_break_even(results)
    assert r_star == pytest.approx(1.0)


def test_no_speedup():
    """ratios = [1.0, 0.50, 0.10], speedups = [0.5, 0.8, 0.9] -> r* = 0.0."""
    ratios = [1.0, 0.50, 0.10]
    speedups = [0.5, 0.8, 0.9]
    results = [{'active_ratio': r, 'opt_speedup': s} for r, s in zip(ratios, speedups)]
    r_star = compute_break_even(results)
    assert r_star == pytest.approx(0.0)


def test_monotonic_boundary():
    """ratios = [1.0, 0.50, 0.25, 0.10], speedups = [0.91, 0.95, 1.01, 3.0] -> r* = 0.25."""
    ratios = [1.0, 0.50, 0.25, 0.10]
    speedups = [0.91, 0.95, 1.01, 3.0]
    results = [{'active_ratio': r, 'opt_speedup': s} for r, s in zip(ratios, speedups)]
    r_star = compute_break_even(results)
    assert r_star == pytest.approx(0.25)


def test_empty_results():
    """Empty results list should return 0.0."""
    assert compute_break_even([]) == pytest.approx(0.0)


def test_exact_1_speedup():
    """Speedup exactly 1.0 should be included."""
    results = [
        {'active_ratio': 1.0, 'opt_speedup': 0.9},
        {'active_ratio': 0.8, 'opt_speedup': 1.0},
        {'active_ratio': 0.5, 'opt_speedup': 1.2},
    ]
    assert compute_break_even(results) == pytest.approx(0.8)


def test_unsorted_ratios():
    """Unsorted input should still find the maximum active_ratio with speedup >= 1.0."""
    results = [
        {'active_ratio': 0.10, 'opt_speedup': 3.0},
        {'active_ratio': 0.75, 'opt_speedup': 1.02},
        {'active_ratio': 0.50, 'opt_speedup': 1.26},
        {'active_ratio': 1.0, 'opt_speedup': 0.91},
    ]
    assert compute_break_even(results) == pytest.approx(0.75)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
