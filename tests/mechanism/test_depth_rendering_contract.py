import torch

from research.background_cache import FrozenBackgroundCache
from research.gaussian_repr import GaussianModel
from research.rasterizer import render


def _single_gaussian(opacity=0.25):
    means = torch.tensor([[0.0, 0.0, 2.0]])
    covariance = torch.eye(3).unsqueeze(0) * 0.01
    colors = torch.tensor([[0.8, 0.2, 0.1]])
    opacities = torch.tensor([opacity])
    camera = torch.eye(4)
    intrinsics = torch.tensor([[20.0, 0.0, 8.0],
                               [0.0, 20.0, 8.0],
                               [0.0, 0.0, 1.0]])
    return means, covariance, colors, opacities, camera, intrinsics


def test_reference_renderer_returns_expected_and_accumulated_depth_separately():
    means, cov, colors, opacities, camera, intrinsics = _single_gaussian(opacity=0.25)
    result = render(means, cov, colors, opacities, camera, intrinsics,
                    image_width=16, image_height=16, backend="reference")
    center = (8, 8)
    alpha = result["alpha"][center]
    assert alpha == torch.tensor(0.25)
    assert result["depth"][center] == torch.tensor(2.0)
    assert result["depth_accumulated"][center] == torch.tensor(0.5)


def test_cached_active_and_frozen_depth_composites_as_expected_depth():
    model = GaussianModel(sh_degree=0, device="cpu")
    points = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    model.initialize_from_points(points, rgb, initial_scale=0.2)
    with torch.no_grad():
        model._opacity.fill_(torch.logit(torch.tensor(0.5)))
    intrinsics = torch.tensor([[20.0, 0.0, 8.0],
                               [0.0, 20.0, 8.0],
                               [0.0, 0.0, 1.0]])
    camera = torch.eye(4)
    active = torch.tensor([True, False])
    cache = FrozenBackgroundCache(device="cpu")
    cache.build_cache(model, ~active, camera, intrinsics, 16, 16, backend="reference")
    subset = model.get_optimization_subset(active)
    result = cache.composite_with_active(subset, camera, intrinsics, 16, 16,
                                         backend="reference")
    assert torch.isfinite(result["depth"]).all()
    assert torch.isfinite(result["depth_accumulated"]).all()
    assert result["depth"][8, 8] > 0
    assert result["alpha"][8, 8] > 0
