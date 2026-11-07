import torch

from src.core.gaussian_repr import build_scaling_matrix, covariance_from_rotation_scale, quaternion_to_rotation_matrix


def test_elementwise_covariance_matches_bmm_and_gradients():
    torch.manual_seed(0)
    q = torch.randn(64, 4, dtype=torch.float64, requires_grad=True)
    s = torch.rand(64, 3, dtype=torch.float64).add(0.1).requires_grad_(True)
    R = quaternion_to_rotation_matrix(q / q.norm(dim=-1, keepdim=True))
    RS = torch.bmm(R, build_scaling_matrix(s))
    ref = torch.bmm(RS, RS.transpose(1, 2))
    fast = covariance_from_rotation_scale(R, s)
    assert torch.allclose(ref, fast, atol=1e-12)
    w = torch.randn_like(ref)
    g_ref = torch.autograd.grad((ref * w).sum(), [q, s], retain_graph=True)
    g_fast = torch.autograd.grad((fast * w).sum(), [q, s])
    for a, b in zip(g_ref, g_fast):
        assert torch.allclose(a, b, atol=1e-10)


def test_separable_ssim_matches_direct_value_and_gradient():
    from src.core.losses import ssim_loss
    torch.manual_seed(1)
    gt = torch.rand(40, 48, 3)
    a = torch.rand(40, 48, 3, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    ld = ssim_loss(a, gt, implementation="direct")
    ls = ssim_loss(b, gt, implementation="separable")
    assert abs(float(ld) - float(ls)) < 1e-6
    ld.backward(); ls.backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-7)


def test_dense_adam_matches_selective_adam_when_all_active():
    from src.core.selective_optimizer import SelectiveAdam
    torch.manual_seed(2)
    base = torch.randn(50, 3)
    pa = torch.nn.Parameter(base.clone()); pb = torch.nn.Parameter(base.clone())
    oa = SelectiveAdam([{"params": [pa], "lr": 0.01}]); ob = SelectiveAdam([{"params": [pb], "lr": 0.01}])
    idx = torch.arange(50)
    for _ in range(4):
        g = torch.randn(50, 3)
        pa.grad = g.clone(); pb.grad = g.clone()
        oa.step(active_idx=idx); ob.step(active_idx=idx, dense=True)
    assert torch.allclose(pa, pb, atol=1e-6)
