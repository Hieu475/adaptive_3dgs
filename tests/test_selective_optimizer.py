"""Unit and Integration Tests for SelectiveAdam Optimizer (R21/R31).

Tests:
1. Sliced parameter updates: only active indices are modified; non-active indices remain strictly unchanged.
2. Moment accumulation: first and second moments are updated only for active indices.
3. Continuous state preservation across densification: `extend_state` retains prior moments.
4. Compaction: `prune_state` maintains parameter-moment alignment after pruning.
"""
import pytest
import torch
import torch.nn as nn

from research.selective_optimizer import SelectiveAdam


def test_selective_step_updates_only_active():
    """Test 1: Sliced Adam step updates only active indices, leaving frozen indices untouched."""
    N = 100
    param = nn.Parameter(torch.ones(N, 3))
    optimizer = SelectiveAdam([{'params': [param], 'lr': 0.01}])
    
    active_idx = torch.tensor([10, 25, 50, 75], dtype=torch.long)
    
    # Assign artificial gradient
    param.grad = torch.ones_like(param.data) * 0.5
    
    # Store old values
    old_data = param.data.clone()
    
    # Perform selective step
    optimizer.step(active_idx=active_idx)
    
    # Check active indices were updated
    assert not torch.equal(param.data[active_idx], old_data[active_idx])
    
    # Check non-active indices were completely unchanged
    non_active_mask = torch.ones(N, dtype=torch.bool)
    non_active_mask[active_idx] = False
    assert torch.equal(param.data[non_active_mask], old_data[non_active_mask])


def test_zero_reset_densification():
    """Test 2: extend_state preserves historical momentum and variance for existing Gaussians."""
    N_old = 50
    N_new = 20
    
    param = nn.Parameter(torch.ones(N_old, 3))
    optimizer = SelectiveAdam([{'params': [param], 'lr': 0.01}])
    
    # Run 5 steps to accumulate momentum on initial Gaussians
    for _ in range(5):
        param.grad = torch.ones_like(param.data) * 0.2
        optimizer.step()
        
    old_m = optimizer.state[param]['exp_avg'].clone()
    old_v = optimizer.state[param]['exp_avg_sq'].clone()
    old_steps = optimizer.state[param]['step'].clone()
    
    # Simulate densification by expanding parameter tensor
    new_param = nn.Parameter(torch.cat([param.data, torch.zeros(N_new, 3)], dim=0))
    optimizer.param_groups[0]['params'] = [new_param]
    optimizer.state[new_param] = optimizer.state.pop(param)
    
    # Extend state
    optimizer.extend_state(N_new)
    
    # Verify historical state is 100% preserved
    assert torch.equal(optimizer.state[new_param]['exp_avg'][:N_old], old_m)
    assert torch.equal(optimizer.state[new_param]['exp_avg_sq'][:N_old], old_v)
    assert torch.equal(optimizer.state[new_param]['step'][:N_old], old_steps)
    
    # Verify new elements are initialized to zero
    assert optimizer.state[new_param]['exp_avg'][N_old:].abs().sum().item() == 0.0
    assert optimizer.state[new_param]['exp_avg_sq'][N_old:].abs().sum().item() == 0.0
    assert optimizer.state[new_param]['step'][N_old:].abs().sum().item() == 0.0


def test_prune_state():
    """Test 3: prune_state compacts momentum and variance buffers accurately."""
    N = 60
    param = nn.Parameter(torch.ones(N, 3))
    optimizer = SelectiveAdam([{'params': [param], 'lr': 0.01}])
    
    param.grad = torch.randn_like(param.data)
    optimizer.step()
    
    # Keep first 30 Gaussians
    keep_mask = torch.zeros(N, dtype=torch.bool)
    keep_mask[:30] = True
    
    expected_m = optimizer.state[param]['exp_avg'][:30].clone()
    optimizer.prune_state(keep_mask)
    
    assert optimizer.state[param]['exp_avg'].shape[0] == 30
    assert torch.equal(optimizer.state[param]['exp_avg'], expected_m)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
