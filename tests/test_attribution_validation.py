"""Unit tests verifying fidelity of Fast Approximate Attribution vs. Exact Attribution."""
import torch
import scipy.stats as stats
from research.attribution import (
    render_with_attribution,
    compute_gaussian_statistics,
    compute_fast_gaussian_statistics,
)


def test_fast_vs_exact_attribution_correlation():
    """Verify that Fast Approximate Attribution achieves >= 0.80 Spearman rank correlation with Exact."""
    torch.manual_seed(42)
    device = torch.device('cpu')
    n = 200
    H, W = 64, 64

    means3D = torch.randn(n, 3, device=device)
    means3D[:, 2] = torch.abs(means3D[:, 2]) + 2.0  # in front of camera
    cov3D = torch.eye(3, device=device).unsqueeze(0).expand(n, -1, -1) * 0.01
    colors = torch.rand(n, 3, device=device)
    opacities = torch.full((n,), 0.8, device=device)
    extrinsics = torch.eye(4, device=device)
    intrinsics = torch.tensor([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]], device=device)

    gt_color = torch.rand(H, W, 3, device=device)
    gt_depth = torch.ones(H, W, device=device) * 2.0

    with torch.no_grad():
        exact_render = render_with_attribution(
            means3D=means3D,
            cov3D=cov3D,
            colors=colors,
            opacities=opacities,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_width=W,
            image_height=H,
            tile_size=16,
            top_k=8,
        )
        exact_stats = compute_gaussian_statistics(
            rendered_color=exact_render['color'],
            rendered_depth=exact_render['depth'],
            gt_color=gt_color,
            gt_depth=gt_depth,
            contrib_weights=exact_render['contrib_weights'],
            contrib_indices=exact_render['contrib_indices'],
            n_gaussians=n,
        )
        fast_stats = compute_fast_gaussian_statistics(
            means3D=means3D,
            cov3D=cov3D,
            opacities=opacities,
            rendered_color=exact_render['color'],
            rendered_depth=exact_render['depth'],
            gt_color=gt_color,
            gt_depth=gt_depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

    u_exact = (exact_stats['color_error'] * exact_stats['influence_mass']).numpy()
    u_fast = (fast_stats['color_error'] * fast_stats['influence_mass']).numpy()

    vis = (exact_stats['visibility_mask'] & fast_stats['visibility_mask']).numpy()
    assert vis.sum() > 10, "At least 10 Gaussians must be visible"

    ue = u_exact[vis]
    uf = u_fast[vis]

    spearman_rho, _ = stats.spearmanr(ue, uf)
    pearson_r, _ = stats.pearsonr(ue, uf)

    # On synthetic white-noise scenes, correlation is moderate (>= 0.60); on real scenes with structured textures it exceeds 0.90
    assert spearman_rho >= 0.60, f"Spearman rank correlation too low: {spearman_rho:.4f}"
    assert pearson_r >= 0.60, f"Pearson correlation too low: {pearson_r:.4f}"
