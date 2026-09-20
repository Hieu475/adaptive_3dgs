"""Validate the newly implemented real Spherical Harmonics color evaluation.

These tests don't need GPU/real data — they check the math is self-consistent:
degree-0 behavior is unchanged, degree>=1 actually varies with viewing
direction (i.e. it's no longer a dead no-op), and the DC-only limit still
recovers plain diffuse color when the rest-coefficients are all zero.
"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from research.gaussian_repr import GaussianModel


def _make_model(sh_degree, n=8):
    torch.manual_seed(0)
    m = GaussianModel(sh_degree=sh_degree, device='cpu')
    pts = torch.randn(n, 3)
    m.initialize_from_points(pts, initial_scale=0.01)
    return m


def test_degree0_unchanged_by_directions():
    m = _make_model(sh_degree=0)
    dirs = torch.nn.functional.normalize(torch.randn(m.num_gaussians, 3), dim=-1)
    c_no_dir = m.get_colors()
    c_with_dir = m.get_colors(dirs)
    assert torch.allclose(c_no_dir, c_with_dir), "degree=0 must ignore directions"


def test_degree1_varies_with_direction():
    m = _make_model(sh_degree=1)
    with torch.no_grad():
        m._features_rest.copy_(torch.randn_like(m._features_rest) * 0.5)
    dirs_a = torch.nn.functional.normalize(torch.randn(m.num_gaussians, 3), dim=-1)
    dirs_b = torch.nn.functional.normalize(torch.randn(m.num_gaussians, 3), dim=-1)
    c_a = m.get_colors(dirs_a)
    c_b = m.get_colors(dirs_b)
    assert not torch.allclose(c_a, c_b), "degree>=1 must be view-dependent (was a dead no-op before this fix)"


def test_degree1_zero_rest_matches_dc_only():
    m = _make_model(sh_degree=1)
    with torch.no_grad():
        m._features_rest.zero_()
    dirs = torch.nn.functional.normalize(torch.randn(m.num_gaussians, 3), dim=-1)
    c = m.get_colors(dirs)
    dc_only = torch.sigmoid(GaussianModel._SH_C0 * m._features_dc[:, 0, :])
    assert torch.allclose(c, dc_only, atol=1e-5), "with zero rest-coeffs, output must equal DC*SH_C0 through sigmoid"


def test_output_in_valid_range():
    for deg in (0, 1, 2, 3):
        m = _make_model(sh_degree=deg)
        dirs = torch.nn.functional.normalize(torch.randn(m.num_gaussians, 3), dim=-1)
        c = m.get_colors(dirs)
        assert (c >= 0).all() and (c <= 1).all(), f"colors out of [0,1] range for degree={deg}"
        assert c.shape == (m.num_gaussians, 3)
