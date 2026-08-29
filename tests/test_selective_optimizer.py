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


def test_active_params_change_frozen_unchanged():
    """Task 7: After optimization step, active parameters change and frozen parameters stay unchanged."""
    N = 200
    param = nn.Parameter(torch.randn(N, 3))
    optimizer = SelectiveAdam([{'params': [param], 'lr': 0.01}])
    
    active_idx = torch.tensor([10, 25, 50, 75, 100, 150], dtype=torch.long)
    frozen_mask = torch.ones(N, dtype=torch.bool)
    frozen_mask[active_idx] = False
    
    # Store original values
    original_active = param.data[active_idx].clone()
    original_frozen = param.data[frozen_mask].clone()
    
    # Simulate gradient from loss
    param.grad = torch.randn_like(param.data) * 0.1
    optimizer.step(active_idx=active_idx)
    
    # Active params must have changed
    assert not torch.equal(param.data[active_idx], original_active), "Active params should change after step"
    
    # Frozen params must be exactly unchanged
    assert torch.equal(param.data[frozen_mask], original_frozen), "Frozen params must not change"


def test_optimizer_state_preserved_after_densification():
    """Task 8: After densification (extend_state), old Gaussian m and v are preserved exactly."""
    N_old = 100
    N_new = 30
    
    param = nn.Parameter(torch.randn(N_old, 3))
    optimizer = SelectiveAdam([{'params': [param], 'lr': 0.01}])
    
    # Run multiple steps with varying active sets to build up non-trivial momentum
    for step in range(10):
        param.grad = torch.randn_like(param.data) * (0.1 + 0.05 * step)
        active = torch.randperm(N_old)[:50]
        optimizer.step(active_idx=active)
    
    # Record momentum and variance for ALL old Gaussians
    m_old = optimizer.state[param]['exp_avg'].clone()
    v_old = optimizer.state[param]['exp_avg_sq'].clone()
    steps_old = optimizer.state[param]['step'].clone()
    
    # Simulate densification: expand parameter
    new_param = nn.Parameter(torch.cat([param.data, torch.randn(N_new, 3)], dim=0))
    optimizer.param_groups[0]['params'] = [new_param]
    optimizer.state[new_param] = optimizer.state.pop(param)
    optimizer.extend_state(N_new)
    
    # Verify: old momentum/variance EXACTLY preserved
    assert torch.equal(optimizer.state[new_param]['exp_avg'][:N_old], m_old), \
        "m (exp_avg) for old Gaussians must be exactly preserved after densification"
    assert torch.equal(optimizer.state[new_param]['exp_avg_sq'][:N_old], v_old), \
        "v (exp_avg_sq) for old Gaussians must be exactly preserved after densification"
    assert torch.equal(optimizer.state[new_param]['step'][:N_old], steps_old), \
        "step counts for old Gaussians must be exactly preserved after densification"
    
    # Verify: new entries initialized to zero
    assert optimizer.state[new_param]['exp_avg'][N_old:].abs().max().item() == 0.0
    assert optimizer.state[new_param]['exp_avg_sq'][N_old:].abs().max().item() == 0.0
    assert optimizer.state[new_param]['step'][N_old:].max().item() == 0
    
    # Verify: total state size matches new parameter size
    assert optimizer.state[new_param]['exp_avg'].shape[0] == N_old + N_new
    assert optimizer.state[new_param]['exp_avg_sq'].shape[0] == N_old + N_new
    assert optimizer.state[new_param]['step'].shape[0] == N_old + N_new


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
