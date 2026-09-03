"""Unit and Integration Tests for Persistent Gaussian State Store and Normalization."""
import pytest
import torch

from research.state_store import GaussianStateStore
from research.importance import robust_normalize, GaussianImportanceEstimator


def test_create_and_persistent_identity():
    """Verify unique persistent IDs and immutable identity across tensor operations."""
    store = GaussianStateStore(device='cpu')
    ids1 = store.create(count=10, frame_idx=0)
    assert len(ids1) == 10
    assert torch.equal(ids1, torch.arange(10))
    
    ids2 = store.create(count=5, frame_idx=1)
    assert len(ids2) == 5
    assert torch.equal(ids2, torch.arange(10, 15))
    assert store.num_gaussians == 15


def test_lineage_tracking_densification():
    """Verify parent-child lineage registration during densification (Section IV)."""
    store = GaussianStateStore(device='cpu')
    parent_ids = store.create(count=4, frame_idx=0)  # IDs 0, 1, 2, 3
    
    # Densify: parent at index 1 (ID 1) spawns 2 children
    child_ids = store.register_densification(
        parent_indices=torch.tensor([1]), n_children_per_parent=2, frame_idx=3
    )
    assert len(child_ids) == 2
    assert torch.equal(child_ids, torch.tensor([4, 5]))
    
    # Query lineage
    p_info = store.get_lineage(1)
    assert p_info['persistent_id'] == 1
    assert p_info['parent_id'] == -1
    assert p_info['children'] == [4, 5]
    assert p_info['status'] == 'active'
    
    # Check child 4 lineage
    c_info = store.get_lineage(4)
    assert c_info['persistent_id'] == 4
    assert c_info['parent_id'] == 1
    assert c_info['creation_frame'] == 3


def test_pruning_remap_and_archive():
    """Verify index compaction preserves persistent IDs and archives pruned Gaussians."""
    store = GaussianStateStore(device='cpu')
    store.create(count=5, frame_idx=0)  # IDs 0, 1, 2, 3, 4
    
    # Keep only indices 0, 2, 4 (prune 1 and 3)
    keep_mask = torch.tensor([True, False, True, False, True])
    pruned_ids = store.remap_after_pruning(keep_mask)
    
    assert torch.equal(pruned_ids, torch.tensor([1, 3]))
    assert store.num_gaussians == 3
    # Surviving IDs should still be 0, 2, 4
    assert torch.equal(store.persistent_ids, torch.tensor([0, 2, 4]))
    
    # Query pruned registry
    p1 = store.get_lineage(1)
    assert p1['status'] == 'pruned'
    p0 = store.get_lineage(0)
    assert p0['status'] == 'active'


def test_frame_ema_updates():
    """Verify identity-preserving EMA updates for errors and visibility."""
    store = GaussianStateStore(device='cpu')
    store.create(count=3, frame_idx=0)
    
    rgb_err = torch.tensor([1.0, 0.5, 0.2])
    depth_err = torch.tensor([0.8, 0.4, 0.1])
    opt_mask = torch.tensor([True, False, True])
    
    store.update_frame(
        frame_idx=1,
        rgb_errors=rgb_err,
        depth_errors=depth_err,
        optimized_mask=opt_mask,
        ema_decay=0.8,
    )
    
    assert store.ages[0].item() == 1
    assert store.last_update_frames[0].item() == 1
    assert store.last_update_frames[1].item() == 0  # not optimized
    assert abs(store.ema_rgb[0].item() - (0.8 * 0.0 + 0.2 * 1.0)) < 1e-4


def test_robust_normalization_freezing():
    """Verify robust normalization eliminates outlier distortion and supports frozen stats (Section V)."""
    # Create tensor with extreme outlier
    data = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 100.0])
    
    norm, p5, p95 = robust_normalize(data, p_low=0.1, p_high=0.9)
    # The normal values should span a meaningful range, not be crushed to 0 by 100.0
    assert norm[0].item() < norm[4].item()
    assert norm[-1].item() == 1.0  # clipped at 1.0
    
    # Test freezing normalization stats
    estimator = GaussianImportanceEstimator()
    estimator.freeze_normalization(p5=0.2, p95=0.8)
    assert estimator.fixed_norm_stats == (0.2, 0.8)
    
    test_tensor = torch.tensor([0.2, 0.5, 0.8, 1.2])
    n_test, _, _ = robust_normalize(test_tensor, fixed_stats=estimator.fixed_norm_stats)
    assert abs(n_test[0].item() - 0.0) < 1e-4
    assert abs(n_test[1].item() - 0.5) < 1e-4
    assert abs(n_test[2].item() - 1.0) < 1e-4
    assert abs(n_test[3].item() - 1.0) < 1e-4  # clipped


def test_state_identity_survives_densification():
    """Verify that existing Gaussians retain their persistent_id and state signals after densification."""
    from research.gaussian_repr import GaussianModel
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i), float(i)] for i in range(5)])
    model.initialize_from_points(points)
    
    # Check initial persistent IDs
    assert torch.equal(model.persistent_ids, torch.tensor([0, 1, 2, 3, 4]))
    
    # Store initial state for Gaussian with persistent_id=3
    orig_pos_3 = model.positions[3].clone()
    model.state_store.ema_rgb[3] = 0.77
    model.state_store.ema_depth[3] = 0.33
    
    # Densify: add 3 new Gaussians
    new_params = {
        'xyz': torch.tensor([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0], [12.0, 12.0, 12.0]]),
    }
    model.add_gaussians(new_params, frame_idx=2)
    
    assert model.num_gaussians == 8
    assert torch.equal(model.persistent_ids, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
    # Gaussian 3 must be untouched
    assert torch.equal(model.positions[3], orig_pos_3)
    assert abs(model.state_store.ema_rgb[3].item() - 0.77) < 1e-5
    assert abs(model.state_store.ema_depth[3].item() - 0.33) < 1e-5
    assert model.state_store.persistent_ids[3].item() == 3


def test_state_identity_survives_pruning():
    """Verify that when tensor indices shift due to pruning/compaction, persistent_id and state track the survivor."""
    from research.gaussian_repr import GaussianModel
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i), float(i)] for i in range(5)])
    model.initialize_from_points(points)
    
    # Target Gaussian: persistent_id = 3, initially at index 3
    target_id = 3
    target_pos = model.positions[target_id].clone()
    model.state_store.ema_rgb[target_id] = 0.88
    model.state_store.ema_depth[target_id] = 0.44
    model.state_store.temporal_drift[target_id] = 0.12
    
    # Prune indices 0, 1, 2 (keep indices 3 and 4)
    prune_mask = torch.tensor([True, True, True, False, False])
    model.prune_gaussians(prune_mask)
    keep_mask = model.compact()
    
    # Model now has 2 Gaussians: former index 3 is now index 0!
    assert model.num_gaussians == 2
    assert model.persistent_ids[0].item() == target_id
    assert torch.equal(model.positions[0], target_pos)
    assert abs(model.state_store.ema_rgb[0].item() - 0.88) < 1e-5
    assert abs(model.state_store.ema_depth[0].item() - 0.44) < 1e-5
    assert abs(model.state_store.temporal_drift[0].item() - 0.12) < 1e-5
    
    # Pruned IDs should be archived
    assert model.state_store.get_lineage(0)['status'] == 'pruned'
    assert model.state_store.get_lineage(1)['status'] == 'pruned'
    assert model.state_store.get_lineage(2)['status'] == 'pruned'
    assert model.state_store.get_lineage(3)['status'] == 'active'


def test_state_identity_survives_clone():
    """Verify that cloned Gaussians inherit parent lineage without modifying parent persistent state."""
    from research.gaussian_repr import GaussianModel
    from research.densification import importance_driven_densification
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i), float(i)] for i in range(4)])
    model.initialize_from_points(points)
    
    # Give candidate Gaussian at index 2 a high gradient and high importance
    model._xyz.grad = torch.zeros_like(model._xyz)
    model._xyz.grad[2] = torch.tensor([0.01, 0.01, 0.01])
    importance_scores = torch.tensor([0.1, 0.2, 0.95, 0.3])
    
    # Run importance-driven densification (clones index 2)
    importance_driven_densification(
        model, importance_scores, high_importance_threshold=0.8, gradient_threshold=0.0002
    )
    
    # Should now have 5 Gaussians
    assert model.num_gaussians == 5
    child_id = 4
    parent_id = 2
    assert model.persistent_ids[child_id].item() == child_id
    
    # Verify parent and child lineage
    parent_lineage = model.state_store.get_lineage(parent_id)
    child_lineage = model.state_store.get_lineage(child_id)
    assert parent_lineage['children'] == [child_id]
    assert child_lineage['parent_id'] == parent_id


def test_temporal_features_follow_identity():
    """Verify that multi-frame temporal signals track persistent identities even across index-shifting compaction."""
    from research.gaussian_repr import GaussianModel
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i), float(i)] for i in range(4)])
    model.initialize_from_points(points)
    
    # Frame 1 update
    model.state_store.update_frame(
        frame_idx=1,
        rgb_errors=torch.tensor([0.1, 0.2, 1.0, 0.4]),
        depth_errors=torch.tensor([0.05, 0.1, 0.5, 0.2]),
        ema_decay=0.8
    )
    # At frame 1: EMA for ID 2 should be (1 - 0.8) * 1.0 = 0.2
    assert abs(model.state_store.ema_rgb[2].item() - 0.2) < 1e-4
    
    # Frame 2 update
    model.state_store.update_frame(
        frame_idx=2,
        rgb_errors=torch.tensor([0.1, 0.2, 1.0, 0.4]),
        depth_errors=torch.tensor([0.05, 0.1, 0.5, 0.2]),
        ema_decay=0.8
    )
    # At frame 2: EMA for ID 2 should be 0.8 * 0.2 + 0.2 * 1.0 = 0.36
    assert abs(model.state_store.ema_rgb[2].item() - 0.36) < 1e-4
    
    # Now prune indices 0 and 1! Keep indices 2 and 3
    model.prune_gaussians(torch.tensor([True, True, False, False]))
    model.compact()
    
    # Index of ID 2 is now 0!
    assert model.persistent_ids[0].item() == 2
    assert abs(model.state_store.ema_rgb[0].item() - 0.36) < 1e-4
    
    # Frame 3 update for surviving Gaussians (now size 2: ID 2 and ID 3)
    model.state_store.update_frame(
        frame_idx=3,
        rgb_errors=torch.tensor([1.0, 0.4]),
        depth_errors=torch.tensor([0.5, 0.2]),
        ema_decay=0.8
    )
    # EMA for ID 2 (at index 0) should be 0.8 * 0.36 + 0.2 * 1.0 = 0.488
    assert abs(model.state_store.ema_rgb[0].item() - 0.488) < 1e-4

