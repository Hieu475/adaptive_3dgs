import torch

from src.tracking.map_tracker import _se3_exp


def test_exp_identity_and_nonzero_rotation_gradient_at_zero():
    xi = torch.zeros(6, requires_grad=True)
    T = _se3_exp(xi)
    assert torch.allclose(T, torch.eye(4), atol=1e-6)
    (T[:3, :3] * torch.randn(3, 3)).sum().backward()
    assert xi.grad[:3].abs().sum() > 0  # the plain norm-based exp has exactly zero rotation gradient here


def test_exp_matches_rotation_about_z():
    T = _se3_exp(torch.tensor([0., 0., 0.5, 0., 0., 0.]))
    c, s = torch.cos(torch.tensor(0.5)), torch.sin(torch.tensor(0.5))
    assert torch.allclose(T[:2, :2], torch.stack([torch.stack([c, -s]), torch.stack([s, c])]), atol=1e-5)
