"""Importance Diagnostics: Verify importance scores carry real information.

Research question (R3):
    Does continuous per-Gaussian importance actually predict
    which Gaussians contribute most to reconstruction error?

This module provides diagnostic tools to answer:
1. Do importance components (E_color, E_depth, V, S, ΔT) carry different information?
2. Does importance correlate with actual per-Gaussian reconstruction error?
3. How do different normalization strategies affect importance quality?
4. Are tier assignments meaningful?

Key metrics:
- Spearman rank correlation between importance and error
- Component independence (pairwise correlation matrix)
- Importance calibration (binned importance vs actual error)
- Tier separation quality (inter-tier vs intra-tier variance ratio)
"""
import torch
from typing import Dict, List, Tuple, Optional
import math


def spearman_rank_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute Spearman rank correlation between two 1D tensors.

    ρ_s = 1 - (6 Σ d²) / (n(n²-1))
    where d_i = rank(x_i) - rank(y_i)

    Args:
        x: (N,) first variable
        y: (N,) second variable

    Returns:
        Spearman correlation coefficient in [-1, 1]
    """
    assert x.shape == y.shape and x.ndim == 1
    n = x.shape[0]
    if n < 3:
        return 0.0

    # Compute ranks
    rank_x = _rank(x)
    rank_y = _rank(y)

    # Spearman formula
    d = rank_x - rank_y
    d_sq_sum = (d ** 2).sum().item()
    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n ** 2 - 1))
    return rho


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Compute ranks of elements (1-based, average ties)."""
    sorted_indices = torch.argsort(x)
    ranks = torch.zeros_like(x)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    return ranks


def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute Pearson correlation between two 1D tensors.

    r = cov(x,y) / (σ_x · σ_y)

    Args:
        x: (N,) first variable
        y: (N,) second variable

    Returns:
        Pearson r in [-1, 1]
    """
    assert x.shape == y.shape and x.ndim == 1
    n = x.shape[0]
    if n < 3:
        return 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    norm_x = x_centered.norm()
    norm_y = y_centered.norm()
    if norm_x < 1e-8 or norm_y < 1e-8:
        return 0.0
    return ((x_centered * y_centered).sum() / (norm_x * norm_y)).item()


def component_correlation_matrix(
    components: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute pairwise Spearman correlations between importance components.

    Answers: Do E_color, E_depth, V, S, ΔT carry different information?
    If all components are highly correlated, they're redundant.
    If they have low correlation, each provides unique signal.

    Args:
        components: dict mapping name -> (N,) tensor
            e.g. {'color_error': ..., 'depth_error': ..., 'visibility': ..., 'screen_area': ...}

    Returns:
        Dict mapping 'name_a__vs__name_b' -> Spearman ρ
    """
    names = sorted(components.keys())
    results = {}
    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if j <= i:
                continue
            rho = spearman_rank_correlation(
                components[name_a].float(),
                components[name_b].float(),
            )
            results[f'{name_a}__vs__{name_b}'] = rho
    return results


def importance_error_correlation(
    importance: torch.Tensor,
    per_gaussian_color_err: torch.Tensor,
    per_gaussian_depth_err: torch.Tensor,
) -> Dict[str, float]:
    """Measure how well importance predicts actual error.

    Core research diagnostic: if ρ(I, E) is high, our importance
    formula is capturing real error. If ρ ≈ 0, importance is useless.

    Args:
        importance: (N,) normalized importance scores
        per_gaussian_color_err: (N,) per-Gaussian color error
        per_gaussian_depth_err: (N,) per-Gaussian depth error

    Returns:
        Dict with correlation metrics
    """
    combined_err = per_gaussian_color_err + per_gaussian_depth_err

    return {
        'importance_vs_color_err_spearman': spearman_rank_correlation(
            importance, per_gaussian_color_err),
        'importance_vs_depth_err_spearman': spearman_rank_correlation(
            importance, per_gaussian_depth_err),
        'importance_vs_combined_err_spearman': spearman_rank_correlation(
            importance, combined_err),
        'importance_vs_color_err_pearson': pearson_correlation(
            importance, per_gaussian_color_err),
        'importance_vs_depth_err_pearson': pearson_correlation(
            importance, per_gaussian_depth_err),
        'importance_vs_combined_err_pearson': pearson_correlation(
            importance, combined_err),
    }


def importance_calibration(
    importance: torch.Tensor,
    actual_error: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute binned calibration: mean error within importance bins.

    If importance is well-calibrated:
    - High importance bins should have high mean error
    - Low importance bins should have low mean error
    - The relationship should be monotonically increasing

    Args:
        importance: (N,) importance scores
        actual_error: (N,) per-Gaussian error
        n_bins: number of bins

    Returns:
        Dict with:
            'bin_edges': (n_bins + 1,) bin boundaries
            'bin_mean_importance': (n_bins,) mean importance per bin
            'bin_mean_error': (n_bins,) mean actual error per bin
            'bin_count': (n_bins,) count per bin
            'monotonicity_score': float, fraction of adjacent bins
                                  where error increases with importance
    """
    N = importance.shape[0]
    device = importance.device

    # Create uniform bins over importance range
    i_min, i_max = importance.min(), importance.max()
    if i_max - i_min < 1e-8:
        # All importance values are the same
        return {
            'bin_edges': torch.linspace(0, 1, n_bins + 1, device=device),
            'bin_mean_importance': torch.zeros(n_bins, device=device),
            'bin_mean_error': torch.full((n_bins,), actual_error.mean().item(), device=device),
            'bin_count': torch.zeros(n_bins, dtype=torch.long, device=device),
            'monotonicity_score': 0.0,
        }

    bin_edges = torch.linspace(i_min.item(), i_max.item() + 1e-8, n_bins + 1, device=device)
    bin_mean_importance = torch.zeros(n_bins, device=device)
    bin_mean_error = torch.zeros(n_bins, device=device)
    bin_count = torch.zeros(n_bins, dtype=torch.long, device=device)

    # Assign bins
    bin_idx = torch.bucketize(importance, bin_edges[1:-1])  # 0..n_bins-1
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    for b in range(n_bins):
        mask = bin_idx == b
        count = mask.sum()
        bin_count[b] = count
        if count > 0:
            bin_mean_importance[b] = importance[mask].mean()
            bin_mean_error[b] = actual_error[mask].mean()

    # Monotonicity: fraction of adjacent non-empty bins where error increases
    non_empty = bin_count > 0
    non_empty_errors = bin_mean_error[non_empty]
    if len(non_empty_errors) < 2:
        mono_score = 0.0
    else:
        increases = (non_empty_errors[1:] >= non_empty_errors[:-1]).float()
        mono_score = increases.mean().item()

    return {
        'bin_edges': bin_edges,
        'bin_mean_importance': bin_mean_importance,
        'bin_mean_error': bin_mean_error,
        'bin_count': bin_count,
        'monotonicity_score': mono_score,
    }


def tier_separation_quality(
    importance: torch.Tensor,
    tiers: torch.Tensor,
    actual_error: torch.Tensor,
) -> Dict[str, float]:
    """Measure how well tiers separate Gaussians by error.

    Computes Fisher's discriminant ratio:
        J = inter-class variance / intra-class variance

    Higher J means tiers create meaningful groups with different
    error levels. Low J means tiers don't separate well.

    Also checks: does Tier A (high importance) actually have
    higher error than Tier C (low importance)?

    Args:
        importance: (N,) importance scores
        tiers: (N,) tier labels (0=A, 1=B, 2=C, 3=D)
        actual_error: (N,) per-Gaussian error

    Returns:
        Dict with separation metrics
    """
    results = {}
    grand_mean = actual_error.mean().item()
    grand_var = actual_error.var().item() if actual_error.numel() > 1 else 0.0

    inter_class_var = 0.0
    intra_class_var = 0.0
    total_count = 0

    tier_stats = {}
    for tier_val in range(4):  # A=0, B=1, C=2, D=3
        mask = tiers == tier_val
        count = mask.sum().item()
        tier_name = ['A', 'B', 'C', 'D'][tier_val]

        if count > 0:
            tier_err = actual_error[mask]
            tier_mean = tier_err.mean().item()
            tier_var = tier_err.var().item() if count > 1 else 0.0
            tier_imp_mean = importance[mask].mean().item()

            inter_class_var += count * (tier_mean - grand_mean) ** 2
            intra_class_var += count * tier_var
            total_count += count

            tier_stats[tier_name] = {
                'count': count,
                'mean_error': tier_mean,
                'mean_importance': tier_imp_mean,
            }
        else:
            tier_stats[tier_name] = {
                'count': 0,
                'mean_error': 0.0,
                'mean_importance': 0.0,
            }

    if total_count > 0:
        inter_class_var /= total_count
        intra_class_var /= total_count
    
    fisher_ratio = inter_class_var / (intra_class_var + 1e-8)

    # Check: Tier A error > Tier C error?
    tier_a_err = tier_stats.get('A', {}).get('mean_error', 0.0)
    tier_c_err = tier_stats.get('C', {}).get('mean_error', 0.0)
    correct_ordering = tier_a_err > tier_c_err

    results['fisher_discriminant_ratio'] = fisher_ratio
    results['correct_tier_ordering'] = float(correct_ordering)
    results['tier_stats'] = tier_stats
    results['inter_class_variance'] = inter_class_var
    results['intra_class_variance'] = intra_class_var

    return results


def compute_full_diagnostics(
    importance: torch.Tensor,
    tiers: torch.Tensor,
    gaussian_stats: Dict[str, torch.Tensor],
    normalization_method: str = 'zscore',
) -> Dict[str, any]:
    """Run all diagnostic checks in one call.

    This is the main entry point for R3 research validation.

    Args:
        importance: (N,) importance scores from importance estimator
        tiers: (N,) tier classifications
        gaussian_stats: output from compute_gaussian_statistics()
            Must contain: color_error, depth_error, visibility, screen_area
        normalization_method: 'raw', 'zscore', or 'robust'

    Returns:
        Comprehensive diagnostics dict
    """
    from .attribution import normalize_importance_components

    color_err = gaussian_stats['color_error']
    depth_err = gaussian_stats['depth_error']
    combined_err = color_err + depth_err

    # 1. Component correlation matrix
    components = {
        'color_error': color_err,
        'depth_error': depth_err,
        'visibility': gaussian_stats['visibility'],
        'screen_area': gaussian_stats['screen_area'],
    }
    comp_correlations = component_correlation_matrix(components)

    # 2. Importance-error correlation
    imp_err_corr = importance_error_correlation(importance, color_err, depth_err)

    # 3. Calibration
    calibration = importance_calibration(importance, combined_err)

    # 4. Tier separation
    tier_quality = tier_separation_quality(importance, tiers, combined_err)

    # 5. Normalization comparison
    norm_comparison = {}
    for method in ['raw', 'zscore', 'robust']:
        normalized = normalize_importance_components(components, method=method)
        # Quick check: what's the scale of each component after normalization?
        norm_comparison[method] = {
            name: {
                'mean': vals.mean().item(),
                'std': vals.std().item(),
                'min': vals.min().item(),
                'max': vals.max().item(),
            }
            for name, vals in normalized.items()
        }

    # 6. Basic statistics
    basic_stats = {
        'n_gaussians': importance.shape[0],
        'importance_mean': importance.mean().item(),
        'importance_std': importance.std().item(),
        'importance_min': importance.min().item(),
        'importance_max': importance.max().item(),
        'color_error_mean': color_err.mean().item(),
        'color_error_std': color_err.std().item(),
        'depth_error_mean': depth_err.mean().item(),
        'depth_error_std': depth_err.std().item(),
        'n_visible': gaussian_stats['visibility_mask'].sum().item(),
        'visibility_fraction': gaussian_stats['visibility_mask'].float().mean().item(),
    }

    return {
        'basic_stats': basic_stats,
        'component_correlations': comp_correlations,
        'importance_error_correlation': imp_err_corr,
        'calibration': {
            'monotonicity_score': calibration['monotonicity_score'],
            'bin_mean_importance': calibration['bin_mean_importance'].tolist(),
            'bin_mean_error': calibration['bin_mean_error'].tolist(),
            'bin_count': calibration['bin_count'].tolist(),
        },
        'tier_quality': {
            'fisher_discriminant_ratio': tier_quality['fisher_discriminant_ratio'],
            'correct_tier_ordering': tier_quality['correct_tier_ordering'],
            'tier_stats': tier_quality['tier_stats'],
        },
        'normalization_comparison': norm_comparison,
    }


def format_diagnostics_report(diagnostics: Dict) -> str:
    """Format diagnostics dict into a human-readable report string.

    Args:
        diagnostics: output from compute_full_diagnostics()

    Returns:
        Formatted multi-line string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  IMPORTANCE DIAGNOSTICS REPORT")
    lines.append("=" * 70)

    # Basic stats
    bs = diagnostics['basic_stats']
    lines.append(f"\n[Basic Statistics]")
    lines.append(f"  N Gaussians:      {bs['n_gaussians']}")
    lines.append(f"  N Visible:        {bs['n_visible']} ({bs['visibility_fraction']:.1%})")
    lines.append(f"  Importance:       {bs['importance_mean']:.4f} ± {bs['importance_std']:.4f}  "
                 f"[{bs['importance_min']:.4f}, {bs['importance_max']:.4f}]")
    lines.append(f"  Color Error:      {bs['color_error_mean']:.6f} ± {bs['color_error_std']:.6f}")
    lines.append(f"  Depth Error:      {bs['depth_error_mean']:.6f} ± {bs['depth_error_std']:.6f}")

    # Importance-error correlation
    iec = diagnostics['importance_error_correlation']
    lines.append(f"\n[Importance ↔ Error Correlation]")
    lines.append(f"  vs Color Error:   Spearman={iec['importance_vs_color_err_spearman']:.4f}  "
                 f"Pearson={iec['importance_vs_color_err_pearson']:.4f}")
    lines.append(f"  vs Depth Error:   Spearman={iec['importance_vs_depth_err_spearman']:.4f}  "
                 f"Pearson={iec['importance_vs_depth_err_pearson']:.4f}")
    lines.append(f"  vs Combined:      Spearman={iec['importance_vs_combined_err_spearman']:.4f}  "
                 f"Pearson={iec['importance_vs_combined_err_pearson']:.4f}")

    # Assess quality
    combined_rho = iec['importance_vs_combined_err_spearman']
    if combined_rho > 0.7:
        quality = "EXCELLENT ✓"
    elif combined_rho > 0.4:
        quality = "GOOD"
    elif combined_rho > 0.2:
        quality = "MODERATE — consider adjusting weights"
    else:
        quality = "POOR ✗ — importance does not predict error"
    lines.append(f"  Quality:          {quality}")

    # Component correlations
    cc = diagnostics['component_correlations']
    lines.append(f"\n[Component Independence (Spearman ρ)]")
    for pair, rho in sorted(cc.items()):
        independence = "independent" if abs(rho) < 0.3 else (
            "weakly correlated" if abs(rho) < 0.6 else "highly correlated")
        lines.append(f"  {pair}: {rho:+.4f}  ({independence})")

    # Calibration
    cal = diagnostics['calibration']
    lines.append(f"\n[Calibration]")
    lines.append(f"  Monotonicity Score: {cal['monotonicity_score']:.2f}")
    if cal['monotonicity_score'] > 0.8:
        lines.append(f"  → Well calibrated: higher importance ≈ higher error ✓")
    elif cal['monotonicity_score'] > 0.5:
        lines.append(f"  → Partially calibrated")
    else:
        lines.append(f"  → Poorly calibrated: importance does not track error ✗")

    # Tier quality
    tq = diagnostics['tier_quality']
    lines.append(f"\n[Tier Separation]")
    lines.append(f"  Fisher Ratio:     {tq['fisher_discriminant_ratio']:.4f}")
    lines.append(f"  Correct Ordering: {'Yes ✓' if tq['correct_tier_ordering'] else 'No ✗'} "
                 f"(Tier A error > Tier C error)")
    for tier_name, stats in sorted(tq['tier_stats'].items()):
        lines.append(f"  Tier {tier_name}: n={stats['count']:>6}  "
                     f"mean_err={stats['mean_error']:.6f}  "
                     f"mean_imp={stats['mean_importance']:.4f}")

    # Normalization comparison
    nc = diagnostics['normalization_comparison']
    lines.append(f"\n[Normalization Comparison]")
    for method, comp_stats in nc.items():
        lines.append(f"  {method}:")
        for comp_name, s in comp_stats.items():
            lines.append(f"    {comp_name:>15}: mean={s['mean']:+.4f}  std={s['std']:.4f}  "
                         f"range=[{s['min']:.4f}, {s['max']:.4f}]")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
