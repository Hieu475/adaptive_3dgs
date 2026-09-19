"""Unit tests for Step 1: Age-Aware Guaranteed Warm-up & Coverage-based Densification Throttling."""
import pytest
import torch
import math

from research.scheduler import BudgetScheduler, OptimizationPolicy
from research.gaussian_repr import GaussianModel
from research.pipeline import OnlineReconstructionPipeline


def test_coverage_based_densification_throttling():
    """Verify that compute_max_new_gaussians throttles map growth when coverage >= 90%."""
    scheduler = BudgetScheduler(
        gpu_budget_ms=15.0,
        budget_allocation={'optimize': 0.50, 'densify': 0.20, 'render': 0.20, 'memory': 0.10},
        cost_densify_us=2.0,
    )
    # Budget = 15.0 * 1000 * 0.20 = 3000 us. Base cap = 3000 / 2.0 = 1500.
    base_cap = scheduler.compute_max_new_gaussians(n_error_pixels=10000)
    assert base_cap == 1500

    # Low coverage (<80%): full exploration cap
    cap_low = scheduler.compute_max_new_gaussians(n_error_pixels=10000, current_coverage=0.50)
    assert cap_low == 1500

    # High coverage (>=90%): throttled by 80% (factor = 0.20)
    cap_high = scheduler.compute_max_new_gaussians(n_error_pixels=10000, current_coverage=0.95)
    assert cap_high == int(1500 * 0.20)  # 300

    # Intermediate coverage (85%): linear ramp between 1.0 and 0.20 -> factor = 0.60
    cap_mid = scheduler.compute_max_new_gaussians(n_error_pixels=10000, current_coverage=0.85)
    expected_mid = int(1500 * 0.60)
    assert abs(cap_mid - expected_mid) <= 1


def test_allocate_adaptive_micro_steps_warmup():
    """Verify that newly spawned Gaussians (update_count < 3) receive guaranteed warmup micro-steps."""
    scheduler = BudgetScheduler(
        gpu_budget_ms=10.0,
        budget_allocation={'optimize': 0.50, 'densify': 0.20, 'render': 0.20, 'memory': 0.10},
    )
    # Total optimize budget = 5000 us
    N = 100
    base_costs = torch.full((N,), 2.0)
    step_costs = torch.full((N,), 0.5)

    # 10 brand-new Gaussians with 0 updates, 90 mature Gaussians with >= 5 updates
    update_counts = torch.full((N,), 5, dtype=torch.long)
    update_counts[:10] = 0  # Brand new

    # Importance scores: suppose mature have higher error than newly spawned primitives
    importance = torch.rand(N) * 0.5
    importance[:10] = 0.01  # Newly spawned have near-zero error

    visibility_mask = torch.ones(N, dtype=torch.bool)

    # Allocate adaptive micro-steps with warm-up enabled
    k_alloc = scheduler.allocate_adaptive_micro_steps(
        importance_scores=importance,
        base_costs=base_costs,
        step_costs=step_costs,
        max_k=3,
        update_counts=update_counts,
        visibility_mask=visibility_mask,
        warmup_steps=3,
        warmup_budget_ratio=0.20,
        warmup_k=2,
    )

    # All 10 brand new Gaussians must receive guaranteed warmup_k = 2 micro-steps
    # despite having near-zero importance!
    assert (k_alloc[:10] == 2).all(), f"Expected all warmup candidates to get K=2, got {k_alloc[:10]}"

    # Total cost must strictly obey budget constraint
    item_costs = torch.where(k_alloc > 0, base_costs + k_alloc.float() * step_costs, 0.0)
    total_spent = item_costs.sum().item()
    assert total_spent <= 5000.0 + 1e-5


def test_select_by_policy_warmup():
    """Verify select_by_policy includes warmup candidates within warmup budget slice."""
    scheduler = BudgetScheduler(
        gpu_budget_ms=5.0,
        budget_allocation={'optimize': 0.50, 'densify': 0.20, 'render': 0.20, 'memory': 0.10},
    )
    # budget = 2500 us. Cost per item = 2.0 us -> ~1250 items max
    N = 200
    cost_estimates = torch.full((N,), 2.0)
    importance = torch.rand(N)
    importance[:15] = 0.001  # Very low error

    update_counts = torch.full((N,), 10, dtype=torch.long)
    update_counts[:15] = 0  # Brand new

    mask = scheduler.select_by_policy(
        policy="ours",
        importance_scores=importance,
        cost_estimates=cost_estimates,
        update_counts=update_counts,
        warmup_steps=3,
        warmup_budget_ratio=0.20,
    )

    # All 15 brand new candidates must be selected in the 20% budget slice
    assert mask[:15].all(), "Brand new Gaussians should be guaranteed selection in warmup slice"


def test_gaussian_model_update_counts_property():
    """Verify update_counts property on GaussianModel is synchronized with StateStore."""
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.randn(20, 3)
    model.initialize_from_points(points)

    assert model.update_counts.shape[0] == 20
    assert (model.update_counts == 0).all()

    # Add 10 new Gaussians
    new_params = {'xyz': torch.randn(10, 3)}
    model.add_gaussians(new_params)

    assert model.num_gaussians == 30
    assert model.update_counts.shape[0] == 30
    assert (model.update_counts == 0).all()


def test_pipeline_warmup_and_coverage_lifecycle():
    """Integration test: Verify pipeline tracks coverage, warms up Gaussians, and increments update_counts."""
    config = {
        'rendering': {'image_width': 64, 'image_height': 64, 'tile_size': 16, 'use_fast_attribution': True, 'backend': 'reference'},
        'scheduler': {
            'gpu_budget_ms': 30.0,
            'policy': 'ours',
            'enable_warmup': True,
            'warmup_steps': 3,
            'warmup_budget_ratio': 0.20,
            'warmup_k': 2,
        },
        'densification': {
            'enable_coverage_throttling': True,
            'throttle_coverage_threshold': 0.90,
            'throttle_factor': 0.20,
            'max_new_per_frame': 50,
        },
        'training': {
            'n_micro_steps': 3,
            'use_adaptive_k': True,
        }
    }
    pipeline = OnlineReconstructionPipeline(config=config, device='cpu')

    H, W = 64, 64
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    intrinsics = torch.tensor([[50.0, 0, 32.0], [0, 50.0, 32.0], [0, 0, 1.0]])

    pipeline.initialize(rgb, depth, intrinsics)
    n_initial = pipeline.gaussian_model.num_gaussians
    assert n_initial > 0
    assert (pipeline.gaussian_model.update_counts == 0).all()

    # Process frame 1
    metrics1 = pipeline.process_frame(rgb, depth)
    assert 'coverage' in metrics1
    assert 'n_warmup' in metrics1
    assert 'n_warmup_optimized' in metrics1

    # After processing frame 1, Gaussians that were optimized must have update_counts == 1
    opt_mask1 = pipeline._last_optimize_mask
    assert opt_mask1.any()
    assert (pipeline.gaussian_model.update_counts[opt_mask1] >= 1).all()

    # Process frame 2
    metrics2 = pipeline.process_frame(rgb, depth)
    assert metrics2['frame'] == 2
