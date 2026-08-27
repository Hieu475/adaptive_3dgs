"""Tests for importance diagnostics module (R3 research validation)."""
import torch
import pytest


class TestSpearmanCorrelation:
    """Test Spearman rank correlation computation."""

    def test_perfect_positive(self):
        from research.importance_diagnostics import spearman_rank_correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - 1.0) < 0.01

    def test_perfect_negative(self):
        from research.importance_diagnostics import spearman_rank_correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0])
        rho = spearman_rank_correlation(x, y)
        assert abs(rho - (-1.0)) < 0.01

    def test_no_correlation(self):
        from research.importance_diagnostics import spearman_rank_correlation
        torch.manual_seed(42)
        x = torch.randn(100)
        y = torch.randn(100)
        rho = spearman_rank_correlation(x, y)
        assert abs(rho) < 0.3  # should be near zero

    def test_too_few_elements(self):
        from research.importance_diagnostics import spearman_rank_correlation
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        rho = spearman_rank_correlation(x, y)
        assert rho == 0.0


class TestPearsonCorrelation:
    """Test Pearson correlation computation."""

    def test_perfect_linear(self):
        from research.importance_diagnostics import pearson_correlation
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 3.0
        r = pearson_correlation(x, y)
        assert abs(r - 1.0) < 0.01

    def test_zero_variance(self):
        from research.importance_diagnostics import pearson_correlation
        x = torch.ones(10)
        y = torch.randn(10)
        r = pearson_correlation(x, y)
        assert r == 0.0  # zero variance in x


class TestComponentCorrelation:
    """Test pairwise component correlation matrix."""

    def test_independent_components(self):
        from research.importance_diagnostics import component_correlation_matrix
        torch.manual_seed(42)
        components = {
            'a': torch.randn(200),
            'b': torch.randn(200),
            'c': torch.randn(200),
        }
        corrs = component_correlation_matrix(components)
        # All should be near zero for independent random variables
        for key, rho in corrs.items():
            assert abs(rho) < 0.3, f"{key} correlation {rho} too high for independent vars"

    def test_correlated_components(self):
        from research.importance_diagnostics import component_correlation_matrix
        torch.manual_seed(42)
        x = torch.randn(200)
        components = {
            'a': x,
            'b': x + torch.randn(200) * 0.1,  # highly correlated with a
        }
        corrs = component_correlation_matrix(components)
        assert corrs['a__vs__b'] > 0.8  # should be highly correlated


class TestImportanceErrorCorrelation:
    """Test importance-error correlation measurement."""

    def test_well_correlated_importance(self):
        from research.importance_diagnostics import importance_error_correlation
        torch.manual_seed(42)
        # Importance tracks error well
        error = torch.rand(100)
        importance = error + torch.randn(100) * 0.05  # noisy version of error
        result = importance_error_correlation(importance, error, torch.zeros(100))
        assert result['importance_vs_color_err_spearman'] > 0.7

    def test_uncorrelated_importance(self):
        from research.importance_diagnostics import importance_error_correlation
        torch.manual_seed(42)
        importance = torch.rand(100)
        error = torch.rand(100)  # completely independent
        result = importance_error_correlation(importance, error, torch.zeros(100))
        assert abs(result['importance_vs_color_err_spearman']) < 0.3


class TestCalibration:
    """Test importance calibration binning."""

    def test_well_calibrated(self):
        from research.importance_diagnostics import importance_calibration
        # Importance = error (perfectly calibrated)
        importance = torch.linspace(0, 1, 100)
        error = importance.clone()
        result = importance_calibration(importance, error, n_bins=5)
        assert result['monotonicity_score'] >= 0.75  # should be monotonic

    def test_constant_importance(self):
        from research.importance_diagnostics import importance_calibration
        importance = torch.ones(50)
        error = torch.rand(50)
        result = importance_calibration(importance, error, n_bins=5)
        assert result['monotonicity_score'] == 0.0  # can't assess monotonicity

    def test_output_shapes(self):
        from research.importance_diagnostics import importance_calibration
        importance = torch.rand(100)
        error = torch.rand(100)
        result = importance_calibration(importance, error, n_bins=10)
        assert result['bin_edges'].shape == (11,)
        assert result['bin_mean_importance'].shape == (10,)
        assert result['bin_mean_error'].shape == (10,)
        assert result['bin_count'].shape == (10,)


class TestTierSeparation:
    """Test tier separation quality metrics."""

    def test_well_separated_tiers(self):
        from research.importance_diagnostics import tier_separation_quality
        # Tier A: high error, Tier C: low error
        importance = torch.cat([
            torch.ones(30) * 0.9,   # Tier A
            torch.ones(40) * 0.5,   # Tier B
            torch.ones(30) * 0.1,   # Tier C
        ])
        tiers = torch.cat([
            torch.zeros(30, dtype=torch.long),  # A
            torch.ones(40, dtype=torch.long),   # B
            torch.full((30,), 2, dtype=torch.long),  # C
        ])
        error = torch.cat([
            torch.ones(30) * 0.8,   # high error for tier A
            torch.ones(40) * 0.4,   # medium error for tier B
            torch.ones(30) * 0.1,   # low error for tier C
        ])
        result = tier_separation_quality(importance, tiers, error)
        assert result['correct_tier_ordering'] == 1.0
        assert result['fisher_discriminant_ratio'] > 1.0

    def test_poorly_separated_tiers(self):
        from research.importance_diagnostics import tier_separation_quality
        torch.manual_seed(42)
        N = 100
        importance = torch.rand(N)
        tiers = torch.randint(0, 3, (N,))
        error = torch.rand(N)  # random, no correlation with tiers
        result = tier_separation_quality(importance, tiers, error)
        # Fisher ratio should be low
        assert result['fisher_discriminant_ratio'] < 1.0


class TestFullDiagnostics:
    """Test the full diagnostics pipeline."""

    def test_runs_without_error(self):
        from research.importance_diagnostics import compute_full_diagnostics
        torch.manual_seed(42)
        N = 50
        importance = torch.rand(N)
        tiers = torch.randint(0, 4, (N,))
        gaussian_stats = {
            'color_error': torch.rand(N),
            'depth_error': torch.rand(N),
            'visibility': torch.rand(N),
            'screen_area': torch.rand(N),
            'visibility_mask': torch.randint(0, 2, (N,)).bool(),
        }
        result = compute_full_diagnostics(importance, tiers, gaussian_stats)

        assert 'basic_stats' in result
        assert 'component_correlations' in result
        assert 'importance_error_correlation' in result
        assert 'calibration' in result
        assert 'tier_quality' in result
        assert 'normalization_comparison' in result

    def test_report_formatting(self):
        from research.importance_diagnostics import (
            compute_full_diagnostics, format_diagnostics_report
        )
        torch.manual_seed(42)
        N = 50
        importance = torch.rand(N)
        tiers = torch.randint(0, 4, (N,))
        gaussian_stats = {
            'color_error': torch.rand(N),
            'depth_error': torch.rand(N),
            'visibility': torch.rand(N),
            'screen_area': torch.rand(N),
            'visibility_mask': torch.randint(0, 2, (N,)).bool(),
        }
        result = compute_full_diagnostics(importance, tiers, gaussian_stats)
        report = format_diagnostics_report(result)

        assert isinstance(report, str)
        assert 'IMPORTANCE DIAGNOSTICS REPORT' in report
        assert 'Spearman' in report
        assert 'Fisher' in report
        assert len(report) > 200
