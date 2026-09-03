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
