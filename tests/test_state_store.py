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


def test_temporal_state_follows_persistent_id():
    """Verify Phase 5: position_drift and temporal features follow persistent_id across index compaction.

    Example from Phase 5:
      frame t:
        Gaussian persistent_id=512 (e.g. index k) has position_drift=0.03
      frame t+1:
        pruning/compaction shifts tensor index
        persistent_id is still 512
        position_drift must continue from 0.03, NOT reset to zero.
    """
    from research.gaussian_repr import GaussianModel
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i), float(i)] for i in range(10)])
    model.initialize_from_points(points)

    target_idx = 4
    target_id = model.persistent_ids[target_idx].item()
    
    # Set temporal signals at frame t
    model.state_store.temporal_drift[target_idx] = 0.03
    model.state_store.ages[target_idx] = 10
    model.state_store.ema_rgb[target_idx] = 0.45
    model.state_store.ema_depth[target_idx] = 0.25

    # Prune indices 0, 1, 2, 3 so target shifts from index 4 to index 0
    prune_mask = torch.tensor([True, True, True, True, False, False, False, False, False, False])
    model.prune_gaussians(prune_mask)
    model.compact()

    # Target is now at index 0
    assert model.persistent_ids[0].item() == target_id
    assert abs(model.state_store.temporal_drift[0].item() - 0.03) < 1e-6, "temporal_drift must NOT reset to zero"
    assert model.state_store.ages[0].item() == 10
    assert abs(model.state_store.ema_rgb[0].item() - 0.45) < 1e-6
    assert abs(model.state_store.ema_depth[0].item() - 0.25) < 1e-6


def test_create_unique_ids():
    """Invariant A: ID uniqueness and monotonic progression across batches."""
    store = GaussianStateStore(device='cpu')
    ids1 = store.create(count=10, frame_idx=0)
    assert len(ids1) == 10
    assert torch.equal(ids1, torch.arange(0, 10))
    
    ids2 = store.create(count=20, frame_idx=1)
    assert len(ids2) == 20
    assert torch.equal(ids2, torch.arange(10, 30))
    
    # ID uniqueness across all active Gaussians
    all_ids = store.persistent_ids.cpu().numpy()
    assert len(all_ids) == len(set(all_ids))
    assert store._next_id == 30


def test_create_parent_lineage():
    """Verify root Gaussians have parent_id == -1 and metadata recorded."""
    store = GaussianStateStore(device='cpu')
    ids = store.create(count=5, frame_idx=0)
    assert torch.all(store.parent_ids == -1)
    for g_id in ids.tolist():
        meta = store.get_lineage(g_id)
        assert meta['parent_id'] == -1
        assert meta['creation_frame'] == 0
        assert meta['children'] == []
        assert meta['status'] == 'active'


def test_update_ema_formula():
    """Verify exact formula: EMA_t = beta * EMA_{t-1} + (1 - beta) * x_t."""
    store = GaussianStateStore(device='cpu')
    store.create(count=1, frame_idx=0)
    assert store.ema_rgb[0].item() == 0.0
    
    # x_1 = 1.0, beta = 0.9 => EMA_1 = 0.9 * 0 + 0.1 * 1.0 = 0.1
    store.update_frame(frame_idx=1, rgb_errors=torch.tensor([1.0]), ema_decay=0.9)
    assert abs(store.ema_rgb[0].item() - 0.1) < 1e-6
    
    # x_2 = 1.0 => EMA_2 = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
    store.update_frame(frame_idx=2, rgb_errors=torch.tensor([1.0]), ema_decay=0.9)
    assert abs(store.ema_rgb[0].item() - 0.19) < 1e-6


def test_age_semantics_exact():
    """Verify age semantic: age_i(t) = t - t_{creation, i}."""
    store = GaussianStateStore(device='cpu')
    # Created at frame 10
    store.create(count=3, frame_idx=10)
    assert torch.all(store.ages == 0)
    
    # Frame 11
    store.update_frame(frame_idx=11)
    assert torch.all(store.ages == 1)
    
    # Frame 12
    store.update_frame(frame_idx=12)
    assert torch.all(store.ages == 2)


def test_last_update_frame_selective():
    """Verify last_update_frame is modified ONLY for Gaussians in optimized_mask."""
    store = GaussianStateStore(device='cpu')
    store.create(count=4, frame_idx=5)
    assert torch.all(store.last_update_frames == 5)
    
    # Optimize indices 0 and 2 at frame 10
    opt_mask = torch.tensor([True, False, True, False])
    store.update_frame(frame_idx=10, optimized_mask=opt_mask)
    
    assert store.last_update_frames[0].item() == 10
    assert store.last_update_frames[1].item() == 5  # untouched
    assert store.last_update_frames[2].item() == 10
    assert store.last_update_frames[3].item() == 5  # untouched


def test_staleness_calculation():
    """Verify staleness_i(t) = t - last_update_frame_i."""
    store = GaussianStateStore(device='cpu')
    store.create(count=2, frame_idx=90)
    
    # Optimize index 0 at frame 93
    store.update_frame(frame_idx=93, optimized_mask=torch.tensor([True, False]))
    assert store.last_update_frames[0].item() == 93
    assert store.last_update_frames[1].item() == 90
    
    # Query staleness at frame 100
    stale = store.get_staleness(frame_idx=100)
    assert stale[0].item() == 7   # 100 - 93 = 7
    assert stale[1].item() == 10  # 100 - 90 = 10


def test_position_drift_calculation():
    """Verify d_i(t) = ||mu_i(t) - mu_i(t-1)||_2."""
    store = GaussianStateStore(device='cpu')
    store.create(count=2, frame_idx=0)
    
    pos_0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    store.update_frame(frame_idx=0, positions=pos_0)
    assert torch.all(store.position_drift == 0.0)
    
    # Frame 1: index 0 moves (3, 4, 0) => drift = 5.0; index 1 stationary
    pos_1 = torch.tensor([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
    store.update_frame(frame_idx=1, positions=pos_1)
    assert abs(store.position_drift[0].item() - 5.0) < 1e-5
    assert abs(store.position_drift[1].item() - 0.0) < 1e-5


def test_residual_drift_ema_calculation():
    """Verify residual drift r_i(t) = |e_i(t) - e_i(t-1)| and its EMA."""
    store = GaussianStateStore(device='cpu')
    store.create(count=1, frame_idx=0)
    
    # Frame 0: error = 0.5
    store.update_frame(frame_idx=0, rgb_errors=torch.tensor([0.5]))
    assert store.residual_drift_ema[0].item() == 0.0
    
    # Frame 1: error = 0.8 => residual drift = |0.8 - 0.5| = 0.3
    # EMA with decay=0.9 => 0.9 * 0 + 0.1 * 0.3 = 0.03
    store.update_frame(frame_idx=1, rgb_errors=torch.tensor([0.8]), ema_decay=0.9)
    assert abs(store.residual_drift_ema[0].item() - 0.03) < 1e-5


def test_reordering_preserves_state():
    """Invariant B: State tracks persistent ID, NOT tensor index under permutation."""
    store = GaussianStateStore(device='cpu')
    store.create(count=4, frame_idx=0)  # IDs 0, 1, 2, 3
    
    # Set distinct state signals for each Gaussian
    store.ema_rgb = torch.tensor([10.0, 11.0, 12.0, 13.0])
    store.ema_depth = torch.tensor([20.0, 21.0, 22.0, 23.0])
    store.ages = torch.tensor([1, 2, 3, 4])
    
    # Permutation: [3, 0, 2, 1]
    perm = torch.tensor([3, 0, 2, 1])
    store.reorder(perm)
    
    # New order of persistent IDs must be [3, 0, 2, 1]
    assert torch.equal(store.persistent_ids, torch.tensor([3, 0, 2, 1]))
    # State values must follow their respective IDs
    assert torch.equal(store.ema_rgb, torch.tensor([13.0, 10.0, 12.0, 11.0]))
    assert torch.equal(store.ema_depth, torch.tensor([23.0, 20.0, 22.0, 21.0]))
    assert torch.equal(store.ages, torch.tensor([4, 1, 3, 2]))


def test_state_matrix_alignment_and_factors():
    """Verify get_state_matrix() aligns with protocol state factors."""
    store = GaussianStateStore(device='cpu')
    store.create(count=3, frame_idx=0)
    
    matrix = store.get_state_matrix()
    required_keys = [
        'persistent_id', 'parent_id', 'age', 'creation_frame', 'last_update_frame',
        'staleness', 'ema_rgb', 'ema_depth', 'ema_influence', 'ema_visibility',
        'uncertainty', 'uncertainty_var', 'temporal_drift', 'position_drift',
        'residual_drift_ema', 'gradient_ema', 'tier'
    ]
    for k in required_keys:
        assert k in matrix, f"Missing state factor key: {k}"
        assert len(matrix[k]) == 3


def test_densification_lineage_preserves_persistent_parent_id():
    """Verify child parent_id stores parent's PERSISTENT ID, not tensor index."""
    store = GaussianStateStore(device='cpu')
    store.create(count=5, frame_idx=0)  # IDs 0, 1, 2, 3, 4
    
    # Prune index 0 so indices shift! Surviving: [1, 2, 3, 4]
    store.remap_after_pruning(torch.tensor([False, True, True, True, True]))
    # Now index 0 has persistent_id 1; index 1 has persistent_id 2
    assert store.persistent_ids[1].item() == 2
    
    # Densify parent at tensor index 1 (persistent_id = 2)
    child_ids = store.register_densification(
        parent_indices=torch.tensor([1]), n_children_per_parent=2, frame_idx=5
    )
    
    # Parent ID of children must be 2, NOT tensor index 1!
    c0_lineage = store.get_lineage(child_ids[0].item())
    c1_lineage = store.get_lineage(child_ids[1].item())
    assert c0_lineage['parent_id'] == 2, "Child parent_id must be persistent ID (2), not index (1)"
    assert c1_lineage['parent_id'] == 2


def test_child_state_initialization_policy():
    """Verify densification policy: 'fresh' (clean state) vs 'inherit' (parent EMA)."""
    store = GaussianStateStore(device='cpu')
    store.create(count=2, frame_idx=0)
    store.ema_rgb[0] = 0.85
    store.ema_depth[0] = 0.42
    
    # 1. Fresh policy (protocol default)
    c_fresh = store.register_densification(
        parent_indices=torch.tensor([0]), n_children_per_parent=1, frame_idx=2, policy='fresh'
    )
    c_fresh_idx = (store.persistent_ids == c_fresh[0]).nonzero(as_tuple=True)[0].item()
    assert store.ema_rgb[c_fresh_idx].item() == 0.0
    assert store.ema_depth[c_fresh_idx].item() == 0.0
    
    # 2. Inherit policy
    c_inherit = store.register_densification(
        parent_indices=torch.tensor([0]), n_children_per_parent=1, frame_idx=3, policy='inherit'
    )
    c_inherit_idx = (store.persistent_ids == c_inherit[0]).nonzero(as_tuple=True)[0].item()
    assert abs(store.ema_rgb[c_inherit_idx].item() - 0.85) < 1e-5
    assert abs(store.ema_depth[c_inherit_idx].item() - 0.42) < 1e-5


def test_snapshot_restore_persistence():
    """Verify state_dict() and load_state_dict() restore state completely and isolate trials."""
    store = GaussianStateStore(device='cpu')
    store.create(count=3, frame_idx=0)
    store.ema_rgb = torch.tensor([0.1, 0.2, 0.3])
    store.ages = torch.tensor([5, 6, 7])
    
    # Save snapshot
    snap = store.state_dict()
    
    # Mutate store (simulating trial optimization)
    store.create(count=2, frame_idx=1)
    store.ema_rgb = torch.tensor([9.9, 9.9, 9.9, 9.9, 9.9])
    store.ages = torch.tensor([99, 99, 99, 99, 99])
    
    # Restore from snapshot
    store.load_state_dict(snap)
    
    assert store.num_gaussians == 3
    assert torch.equal(store.persistent_ids, torch.tensor([0, 1, 2]))
    assert torch.equal(store.ema_rgb, torch.tensor([0.1, 0.2, 0.3]))
    assert torch.equal(store.ages, torch.tensor([5, 6, 7]))
    assert store._next_id == 3


def test_state_store_determinism():
    """Verify identical sequence of operations yields bitwise identical persistent state."""
    def run_sequence():
        s = GaussianStateStore(device='cpu')
        s.create(count=5, frame_idx=0)
        s.update_frame(
            frame_idx=1,
            rgb_errors=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            depth_errors=torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]),
            optimized_mask=torch.tensor([True, False, True, False, True])
        )
        s.remap_after_pruning(torch.tensor([True, False, True, True, False]))
        s.register_densification(parent_indices=torch.tensor([0]), n_children_per_parent=2, frame_idx=2)
        return s.state_dict()
        
    s1 = run_sequence()
    s2 = run_sequence()
    
    assert s1['_next_id'] == s2['_next_id']
    assert torch.equal(s1['persistent_ids'], s2['persistent_ids'])
    assert torch.equal(s1['parent_ids'], s2['parent_ids'])
    assert torch.equal(s1['ema_rgb'], s2['ema_rgb'])
    assert torch.equal(s1['last_update_frames'], s2['last_update_frames'])
    assert s1['_id_to_metadata'] == s2['_id_to_metadata']


def test_visibility_count_and_ema():
    """Point A: Verify raw visibility_count is preserved and ema_visibility tracks it."""
    store = GaussianStateStore(device='cpu')
    store.create(count=3, frame_idx=0)
    
    # Frame 0: pixel counts [120, 45, 0]
    store.update_frame(frame_idx=0, visibility_count=torch.tensor([120.0, 45.0, 0.0]), ema_decay=0.9)
    assert abs(store.visibility_count[0].item() - 120.0) < 1e-5
    assert abs(store.visibility_count[1].item() - 45.0) < 1e-5
    assert abs(store.visibility_count[2].item() - 0.0) < 1e-5
    # EMA: 0.9 * 0 + 0.1 * count
    assert abs(store.ema_visibility[0].item() - 12.0) < 1e-5
    assert abs(store.ema_visibility[1].item() - 4.5) < 1e-5
    
    # Frame 1: pixel counts [150, 50, 10]
    store.update_frame(frame_idx=1, visibility_count=torch.tensor([150.0, 50.0, 10.0]), ema_decay=0.9)
    # EMA: 0.9 * 12.0 + 0.1 * 150 = 10.8 + 15.0 = 25.8
    assert abs(store.ema_visibility[0].item() - 25.8) < 1e-4
    
    # Check state matrix contains both
    matrix = store.get_state_matrix()
    assert 'visibility_count' in matrix
    assert 'ema_visibility' in matrix
    assert abs(matrix['visibility_count'][0].item() - 150.0) < 1e-5


def test_temporal_drift_alias_unification():
    """Point B: Verify position_drift and temporal_drift refer to the same underlying tensor."""
    store = GaussianStateStore(device='cpu')
    store.create(count=2, frame_idx=0)
    
    # Mutating position_drift directly changes temporal_drift
    store.position_drift = torch.tensor([0.05, 0.12])
    assert torch.equal(store.temporal_drift, store.position_drift)
    assert abs(store.temporal_drift[0].item() - 0.05) < 1e-5
    
    # Mutating via legacy setter
    store.temporal_drift = torch.tensor([0.20, 0.35])
    assert torch.equal(store.position_drift, torch.tensor([0.20, 0.35]))
    
    # State matrix contains both pointing to the same data
    matrix = store.get_state_matrix()
    assert torch.equal(matrix['position_drift'], matrix['temporal_drift'])


def test_observed_update_frequency():
    """3-FIX-4: Verify true observed update frequency f_i(t) = updates_i / max(1, t - t_creation + 1)."""
    store = GaussianStateStore(device='cpu')
    store.create(count=2, frame_idx=0)
    
    # Frame 1: optimize Gaussian 0 only
    store.update_frame(frame_idx=1, optimized_mask=torch.tensor([True, False]))
    # Frame 2: optimize Gaussian 0 only
    store.update_frame(frame_idx=2, optimized_mask=torch.tensor([True, False]))
    # Frame 3: optimize neither
    store.update_frame(frame_idx=3, optimized_mask=torch.tensor([False, False]))
    # Frame 4: optimize Gaussian 0 and Gaussian 1
    store.update_frame(frame_idx=4, optimized_mask=torch.tensor([True, True]))
    
    freq = store.get_update_frequency(frame_idx=4)
    # Gaussian 0: 3 updates over 5 frames (0,1,2,3,4) => 3/5 = 0.6
    assert abs(freq[0].item() - 0.60) < 1e-5
    # Gaussian 1: 1 update over 5 frames => 1/5 = 0.2
    assert abs(freq[1].item() - 0.20) < 1e-5
    
    # Check that update_frequency is in get_state_matrix()
    matrix = store.get_state_matrix()
    assert 'update_frequency' in matrix
    assert abs(matrix['update_frequency'][0].item() - 0.60) < 1e-5





