"""Integration Tests for GaussianStateStore with GaussianModel and Oracle Isolation."""
import pytest
import torch
import copy

from research.gaussian_repr import GaussianModel, GaussianState
from research.state_store import GaussianStateStore


def test_model_lifecycle_pipeline():
    """Scenario 2.17: create -> update -> prune -> densify -> update.
    
    Initial: G = [A, B, C, D]
    Prune: remove B
    Densify: C -> C1, C2
    Optimize: A, C, D, C1, C2
    """
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([
        [0.0, 0.0, 0.0],  # A (ID 0)
        [1.0, 1.0, 1.0],  # B (ID 1)
        [2.0, 2.0, 2.0],  # C (ID 2)
        [3.0, 3.0, 3.0],  # D (ID 3)
    ])
    model.initialize_from_points(points)
    assert model.num_gaussians == 4
    assert torch.equal(model.persistent_ids, torch.tensor([0, 1, 2, 3]))
    
    # Frame 0: Update state store with distinct initial signals
    model.state_store.update_frame(
        frame_idx=0,
        rgb_errors=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        depth_errors=torch.tensor([0.01, 0.02, 0.03, 0.04]),
    )
    assert abs(model.state_store.ema_rgb[2].item() - 0.03) < 1e-4
    
    # Frame 1: Prune B (index 1, ID 1)
    prune_mask = torch.tensor([False, True, False, False])
    model.prune_gaussians(prune_mask)
    model.compact()
    
    # After compact: surviving are [A (ID 0), C (ID 2), D (ID 3)]
    assert model.num_gaussians == 3
    assert torch.equal(model.persistent_ids, torch.tensor([0, 2, 3]))
    # C is now at index 1, its state must remain intact
    assert abs(model.state_store.ema_rgb[1].item() - 0.03) < 1e-4
    assert model.state_store.get_lineage(1)['status'] == 'pruned'
    
    # Frame 2: Densify C (tensor index 1, persistent_id 2) -> 2 children (C1, C2)
    # Add new gaussians at C's position
    c_pos = model.positions[1].clone()
    new_params = {
        'xyz': torch.stack([c_pos + 0.01, c_pos - 0.01]),
    }
    model.add_gaussians(new_params, parent_indices=torch.tensor([1, 1]), frame_idx=2)
    
    # Model should now have 5 Gaussians: [A(0), C(2), D(3), C1(4), C2(5)]
    assert model.num_gaussians == 5
    assert torch.equal(model.persistent_ids, torch.tensor([0, 2, 3, 4, 5]))
    
    # Lineage check: C1 (ID 4) and C2 (ID 5) must have parent_id == 2
    c1_meta = model.state_store.get_lineage(4)
    c2_meta = model.state_store.get_lineage(5)
    assert c1_meta['parent_id'] == 2
    assert c2_meta['parent_id'] == 2
    
    # Frame 3: Optimize all surviving Gaussians
    opt_mask = torch.ones(5, dtype=torch.bool)
    model.state_store.update_frame(
        frame_idx=3,
        rgb_errors=torch.tensor([0.15, 0.25, 0.35, 0.45, 0.55]),
        optimized_mask=opt_mask,
    )
    assert torch.all(model.state_store.last_update_frames == 3)
    assert model.state_store.ages[0].item() == 3  # created at 0
    assert model.state_store.ages[3].item() == 1  # created at 2 (C1)
    assert model.state_store.ages[4].item() == 1  # created at 2 (C2)


def test_model_reorder_integration():
    """Scenario 2.18: GaussianModel.reorder() permutes state in lockstep with geometry."""
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), float(i)*2, float(i)*3] for i in range(5)])
    model.initialize_from_points(points)
    
    # Set unique state per ID
    for i in range(5):
        model.state_store.ema_rgb[i] = float(i * 10)
        model.state_store.ages[i] = i
        
    perm = torch.tensor([4, 2, 0, 3, 1])
    orig_pos_4 = model.positions[4].clone()
    orig_pos_0 = model.positions[0].clone()
    
    # Reorder model
    model.reorder(perm)
    
    # Check that positions moved to new indices
    assert torch.equal(model.positions[0], orig_pos_4)
    assert torch.equal(model.positions[2], orig_pos_0)
    
    # Check that state store followed identically
    assert torch.equal(model.persistent_ids, torch.tensor([4, 2, 0, 3, 1]))
    assert model.state_store.ema_rgb[0].item() == 40.0
    assert model.state_store.ema_rgb[2].item() == 0.0
    assert model.state_store.ages[0].item() == 4
    assert model.state_store.ages[2].item() == 0


def test_oracle_snapshot_restore_isolation():
    """Scenario 2.23 & 2.24: Oracle trial does not contaminate persistent state store."""
    from research.pipeline import OnlineReconstructionPipeline
    from research.oracle_utility import OracleUtilityExperiment
    
    config = {
        'gaussian': {'sh_degree': 0, 'initial_opacity': 0.5, 'max_gaussians': 1000},
        'rendering': {'tile_size': 16, 'image_width': 32, 'image_height': 32},
        'scheduler': {'gpu_budget_ms': 20.0, 'policy': 'budget_aware'},
    }
    pipeline = OnlineReconstructionPipeline(config=config, device='cpu')
    rgb = torch.zeros(32, 32, 3)
    depth = torch.ones(32, 32)
    intr = torch.eye(3)
    pose = torch.eye(4)
    pipeline.initialize(rgb=rgb, depth=depth, intrinsics=intr, pose=pose)
    
    oracle = OracleUtilityExperiment(pipeline=pipeline, n_samples=5, n_opt_steps=2, seed=42)
    
    # Capture initial baseline state
    orig_num_g = pipeline.gaussian_model.num_gaussians
    orig_ids = pipeline.gaussian_model.state_store.persistent_ids.clone()
    orig_next_id = pipeline.gaussian_model.state_store._next_id
    pipeline.gaussian_model.state_store.ema_rgb[0] = 0.555
    
    # Snapshot
    snap = oracle.snapshot_state()
    
    # Mutate pipeline and state store (as would happen during trial intervention)
    pipeline.gaussian_model.state_store.ema_rgb[0] = 9.999
    pipeline.gaussian_model.state_store.create(count=10, frame_idx=99)
    pipeline.gaussian_model._xyz.data[0] += 5.0
    
    # Restore
    oracle.restore_state(snap)
    
    # Verify exact restore
    assert pipeline.gaussian_model.num_gaussians == orig_num_g
    assert torch.equal(pipeline.gaussian_model.state_store.persistent_ids, orig_ids)
    assert pipeline.gaussian_model.state_store._next_id == orig_next_id
    assert abs(pipeline.gaussian_model.state_store.ema_rgb[0].item() - 0.555) < 1e-6


def test_add_gaussians_dimension_validation():
    """Point C: Verify explicit parent mapping and dimension validation in add_gaussians."""
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), 0.0, 0.0] for i in range(4)])
    model.initialize_from_points(points)
    assert model.num_gaussians == 4
    
    # Case 1: 2 parents each spawn 2 children => 4 new Gaussians
    new_params_4 = {
        'xyz': torch.tensor([[10.0, 0.0, 0.0], [10.1, 0.0, 0.0], [11.0, 0.0, 0.0], [11.1, 0.0, 0.0]])
    }
    # parent indices [0, 1] with n_children_per_parent=2 (2 * 2 == 4)
    model.add_gaussians(new_params_4, parent_indices=torch.tensor([0, 1]), n_children_per_parent=2, frame_idx=1)
    assert model.num_gaussians == 8
    assert model.state_store.num_gaussians == 8
    
    # Check parent IDs
    assert model.state_store.get_lineage(4)['parent_id'] == 0
    assert model.state_store.get_lineage(5)['parent_id'] == 0
    assert model.state_store.get_lineage(6)['parent_id'] == 1
    assert model.state_store.get_lineage(7)['parent_id'] == 1
    
    # Case 2: Dimension mismatch error check
    new_params_3 = {'xyz': torch.zeros(3, 3)}
    # 2 parents with n_children=1 != 3 new Gaussians => must raise ValueError
    with pytest.raises(ValueError) as excinfo:
        model.add_gaussians(new_params_3, parent_indices=torch.tensor([0, 1]), n_children_per_parent=1)
    assert "Dimension mismatch in add_gaussians" in str(excinfo.value)


def test_gaussian_model_state_separation_contract():
    """Point D: Verify separation of concerns between GaussianModel._state and StateStore."""
    model = GaussianModel(sh_degree=0, device='cpu')
    points = torch.tensor([[float(i), 0.0, 0.0] for i in range(5)])
    model.initialize_from_points(points)
    
    # GaussianModel._state manages execution/rendering flags
    model._state[1] = GaussianState.FROZEN
    model._state[3] = GaussianState.PRUNED
    
    # GaussianStateStore manages research tiers and identity independently
    model.state_store.tiers[1] = 0  # Priority Tier A
    model.state_store.tiers[3] = 3  # Priority Tier D
    
    # Compact prunes the PRUNED Gaussian from both
    model.compact()
    
    # Surviving are indices 0, 1, 2, 4 (total 4)
    assert model.num_gaussians == 4
    assert model.state_store.num_gaussians == 4
    assert model._state[1] == GaussianState.FROZEN
    assert model.state_store.tiers[1] == 0  # Tier preserved
    # PRUNED Gaussian (ID 3) archived in StateStore registry
    assert model.state_store.get_lineage(3)['status'] == 'pruned'

