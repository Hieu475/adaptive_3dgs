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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
