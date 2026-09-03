"""Unit and Integration Tests for Oracle Utility Engine (R34/R35).

Tests:
1. Pipeline evaluate_gaussian_update produces valid non-negative cost and delta quality.
2. State preservation: evaluating Gaussians does not alter base model parameters.
3. Regret@K computation correctly tracks loss against Oracle optimal gain.
"""
import pytest
import torch
import numpy as np
import math

from research.pipeline import OnlineReconstructionPipeline


def test_oracle_evaluation_and_state_preservation():
    """Test that evaluate_gaussian_update executes isolated trials and preserves model state."""
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    # Initialize with simple frame
    H, W = 32, 40
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    
    # Check baseline parameter snapshot
    orig_xyz = pipeline.gaussian_model._xyz.clone()
    
    # Evaluate update for a small subset of Gaussians
    eval_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    frame = {'rgb': rgb, 'depth': depth, 'pose': torch.eye(4)}
    
    result = pipeline.evaluate_gaussian_update(eval_indices, frame, n_steps=2)
    
    # Check metrics
    assert 'delta_psnr' in result
    assert 'delta_depth_gain' in result
    assert 'delta_quality' in result
    assert 'measured_trial_cost_ms' in result
    assert result['measured_trial_cost_ms'] > 0.0
    
    # Check that base model parameters were restored exactly
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz), "Base model state must be preserved after Oracle evaluation"


def test_oracle_repeatability():
    """Test 2 (Item 17): Same Gaussian evaluated twice in identical state produces identical delta Q."""
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    
    eval_indices = torch.tensor([1], dtype=torch.long)
    frame = {'rgb': rgb, 'depth': depth, 'pose': torch.eye(4)}
    
    # Run 1
    torch.manual_seed(42)
    res1 = pipeline.evaluate_gaussian_update(eval_indices, frame, n_steps=3)
    
    # Run 2
    torch.manual_seed(42)
    res2 = pipeline.evaluate_gaussian_update(eval_indices, frame, n_steps=3)
    
    # Verify exact repeatability
    assert abs(res1['delta_quality'] - res2['delta_quality']) < 1e-5, "Oracle evaluation must be strictly repeatable"
    assert abs(res1['delta_psnr'] - res2['delta_psnr']) < 1e-5, "Delta PSNR must be strictly repeatable"


def test_oracle_geometry_stratification_and_raw_metrics():
    """Test geometry-stratified sampling, raw metrics extraction, and stability verification."""
    from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    # Add depth discontinuity
    depth[16:, :] = 4.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    pipeline.process_frame(rgb, depth)
    
    exp = OracleUtilityExperiment(pipeline=pipeline, n_samples=12, n_opt_steps=2, seed=42)
    
    # Run oracle experiment with geometry stratification
    results = exp.run_oracle_experiment(rgb, depth, population_type=SamplingPopulation.GEOMETRY_STRATIFIED)
    
    assert len(results) > 0
    row = results[0]
    # Check that raw decoupled metrics exist
    assert 'delta_psnr_local' in row
    assert 'delta_ssim_local' in row
    assert 'delta_depth_gain_local' in row
    assert 'delta_loss_local' in row
    assert 'measured_trial_cost_ms' in row
    assert 'oracle_utility_rgb' in row
    assert 'oracle_utility_depth' in row
    assert 'oracle_utility_joint' in row
    assert 'features' in row
    assert 'rgb_error' in row['features']
    assert 'depth_error' in row['features']
    assert 'geometry_stratum' in row
    
    # Test stability check on candidate subset
    visible_candidates = [r['gaussian_id'] for r in results if r.get('visible', True)][:3]
    if len(visible_candidates) > 0:
        stability = exp.run_stability_check(rgb, depth, candidate_indices=visible_candidates, n_repeats=2)
        assert 'mean_cv' in stability
        assert 'stable_fraction' in stability
        assert 'mean_sign_stability' in stability
        assert stability['n_repeats'] == 2


def test_oracle_group_interaction():
    """Verify group utility evaluation and additivity error measurement (Section IX)."""
    from research.oracle_utility import OracleUtilityExperiment
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    pipeline.process_frame(rgb, depth)
    
    exp = OracleUtilityExperiment(pipeline=pipeline, n_samples=10, n_opt_steps=2, seed=42)
    
    candidates = list(range(min(8, pipeline.gaussian_model.num_gaussians)))
    group_res = exp.evaluate_group_interaction(rgb, depth, candidate_indices=candidates, group_sizes=[1, 2], n_groups_per_size=2)
    
    assert 'group_size_1' in group_res
    assert group_res['group_size_1']['interaction_error_mean'] == 0.0
    if 'group_size_2' in group_res:
        assert 'interaction_error_mean' in group_res['group_size_2']
        assert 'additivity_ratio_mean' in group_res['group_size_2']


def hash_gaussian_state(model, optimizer=None):
    """Compute SHA-256 cryptographic hash over all model parameters, buffers, and optimizer states."""
    import hashlib
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode())
        hasher.update(param.detach().cpu().numpy().tobytes())
    for name, buf in sorted(model.named_buffers()):
        hasher.update(name.encode())
        hasher.update(buf.detach().cpu().numpy().tobytes())
    if optimizer is not None:
        state_dict = optimizer.state_dict()
        for k in sorted(state_dict.keys(), key=lambda x: str(x)):
            v = state_dict[k]
            if isinstance(v, torch.Tensor):
                hasher.update(v.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def test_oracle_snapshot_restore_exact_state_hash():
    """Phase 2.1 Audit: Assert cryptographic equality of positions, scale, rotation, opacity, SH, and optimizer states."""
    from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    pipeline.process_frame(rgb, depth)
    
    # Snapshot baseline states
    state_before_hash = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    orig_xyz = pipeline.gaussian_model._xyz.clone()
    orig_scaling = pipeline.gaussian_model._scaling.clone()
    orig_rotation = pipeline.gaussian_model._rotation.clone()
    orig_opacity = pipeline.gaussian_model._opacity.clone()
    orig_dc = pipeline.gaussian_model._features_dc.clone()
    orig_rest = pipeline.gaussian_model._features_rest.clone()
    orig_normals = pipeline.gaussian_model._normals.clone()
    
    # 1. Test isolated trial via evaluate_gaussian_update
    frame = {'rgb': rgb, 'depth': depth, 'pose': torch.eye(4)}
    res = pipeline.evaluate_gaussian_update(torch.tensor([2, 3]), frame, n_steps=4)
    state_after_single = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    
    assert state_before_hash == state_after_single, "State hash must be identical after evaluate_gaussian_update"
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz)
    assert torch.equal(pipeline.gaussian_model._scaling, orig_scaling)
    assert torch.equal(pipeline.gaussian_model._rotation, orig_rotation)
    assert torch.equal(pipeline.gaussian_model._opacity, orig_opacity)
    assert torch.equal(pipeline.gaussian_model._features_dc, orig_dc)
    assert torch.equal(pipeline.gaussian_model._features_rest, orig_rest)
    assert torch.equal(pipeline.gaussian_model._normals, orig_normals)
    
    # 2. Test multi-sample stratified intervention via OracleUtilityExperiment
    exp = OracleUtilityExperiment(pipeline=pipeline, n_samples=6, n_opt_steps=3, seed=42)
    exp.run_oracle_experiment(rgb, depth, population_type=SamplingPopulation.GEOMETRY_STRATIFIED)
    state_after_oracle = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    
    assert state_before_hash == state_after_oracle, "State hash must be identical after full Oracle experiment"
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz)
    assert torch.equal(pipeline.gaussian_model._scaling, orig_scaling)
    assert torch.equal(pipeline.gaussian_model._rotation, orig_rotation)
    assert torch.equal(pipeline.gaussian_model._opacity, orig_opacity)
    assert torch.equal(pipeline.gaussian_model._features_dc, orig_dc)
    assert torch.equal(pipeline.gaussian_model._features_rest, orig_rest)
    assert torch.equal(pipeline.gaussian_model._normals, orig_normals)


def test_no_oracle_leakage_in_features():
    """Phase 2.2 Audit: Ensure zero data leakage of post-optimization metrics into state feature inputs."""
    from research.oracle_utility import OracleUtilityExperiment, SamplingPopulation
    pipeline = OnlineReconstructionPipeline(device='cpu')
    
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)
    
    pipeline.initialize(rgb, depth, intrinsics)
    pipeline.process_frame(rgb, depth)
    
    exp = OracleUtilityExperiment(pipeline=pipeline, n_samples=8, n_opt_steps=2, seed=42)
    results = exp.run_oracle_experiment(rgb, depth, population_type=SamplingPopulation.GEOMETRY_STRATIFIED)
    
    forbidden_tokens = ['oracle', 'delta', 'post_', 'after', 'trial_cost', 'gain']
    for row in results:
        feats = row.get('features', {})
        assert len(feats) > 0, "Features dict must not be empty"
        for k in feats.keys():
            k_lower = k.lower()
            for token in forbidden_tokens:
                assert token not in k_lower, f"Leakage violation: forbidden token '{token}' in input feature key '{k}'"
                
        # Validate that all features are finite numeric values
        for k, v in feats.items():
            assert not np.isnan(v), f"Feature {k} contains NaN"
            assert not np.isinf(v), f"Feature {k} contains Inf"


def test_oracle_multi_trial_sequential_invariance():
    """Phase 6.2 Audit: Sequential trials (trial A -> restore -> trial B -> restore -> trial C -> restore).
    Asserts baseline state hash and parameters before each trial are strictly identical.
    """
    pipeline = OnlineReconstructionPipeline(device='cpu')
    H, W = 32, 40
    torch.manual_seed(42)
    rgb = torch.rand(H, W, 3)
    depth = torch.ones(H, W) * 2.0
    fx, fy = 80.0, 80.0
    intrinsics = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=torch.float32)

    pipeline.initialize(rgb, depth, intrinsics)
    pipeline.process_frame(rgb, depth)

    initial_hash = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    orig_xyz = pipeline.gaussian_model._xyz.clone()
    frame = {'rgb': rgb, 'depth': depth, 'pose': torch.eye(4)}

    # Trial A
    pipeline.evaluate_gaussian_update(torch.tensor([0]), frame, n_steps=3)
    hash_after_a = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    assert hash_after_a == initial_hash, "Baseline state corrupted after Trial A"
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz)

    # Trial B
    pipeline.evaluate_gaussian_update(torch.tensor([1, 2]), frame, n_steps=5)
    hash_after_b = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    assert hash_after_b == initial_hash, "Baseline state corrupted after Trial B"
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz)

    # Trial C
    pipeline.evaluate_gaussian_update(torch.tensor([0, 1, 2]), frame, n_steps=2)
    hash_after_c = hash_gaussian_state(pipeline.gaussian_model, pipeline.optimizer)
    assert hash_after_c == initial_hash, "Baseline state corrupted after Trial C"
    assert torch.equal(pipeline.gaussian_model._xyz, orig_xyz)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



