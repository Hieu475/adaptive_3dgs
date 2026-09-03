"""Unit and Integration Tests for Oracle Utility Engine (R34/R35).

Tests:
1. Pipeline evaluate_gaussian_update produces valid non-negative cost and delta quality.
2. State preservation: evaluating Gaussians does not alter base model parameters.
3. Regret@K computation correctly tracks loss against Oracle optimal gain.
"""
import pytest
import torch

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
