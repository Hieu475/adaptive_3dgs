import torch
from research.tracker import DenseRGBDTracker, _se3_exp


def test_se3_exp_identity():
    T = _se3_exp(torch.zeros(6))
    assert torch.allclose(T, torch.eye(4), atol=1e-6)


def test_identical_frames_give_near_zero_motion():
    torch.manual_seed(0)
    tr = DenseRGBDTracker(n_levels=1, n_iters=(5,))
    rgb = torch.rand(60, 80, 3)
    depth = torch.full((60, 80), 2.0) + 0.01 * torch.randn(60, 80)
    K = torch.tensor([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]])
    tr.reset(rgb, depth, torch.eye(4))
    tr._K = K
    T = tr.track(rgb, depth, K)
    assert (T - torch.eye(4)).norm() < 0.05  # near-identity for identical frames
