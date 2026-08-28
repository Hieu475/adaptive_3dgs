"""Selective Adam Optimizer for Adaptive 3D Gaussian Splatting (R21 / Step 12).

Key Features:
1. Selective Parameter Update:
   - Optimizer arithmetic and memory movement scale with O(M) where M is the number of active Gaussians.
   - Only active slices of parameters, first momentum (m), and second momentum (v) are read and written:
       m_t[idx] = β₁ · m_{t-1}[idx] + (1 - β₁) · g_t[idx]
       v_t[idx] = β₂ · v_{t-1}[idx] + (1 - β₂) · (g_t[idx])²
       θ_t[idx] = θ_{t-1}[idx] - (η / (sqrt(v̂_t[idx]) + ε)) · m̂_t[idx]
2. Continuous State Preservation (Zero-Reset Densification):
   - When new Gaussians are added during densification, `extend_state(n_new)` extends state buffers
     without destroying existing historical momentum and variance.
3. State Compaction:
   - `prune_state(keep_mask)` compacts state tensors when low-value Gaussians are pruned.
"""
import torch
from typing import List, Dict, Union, Optional, Tuple, Any
import math


class SelectiveAdam:
    """Selective Adam optimizer that updates only active parameter slices."""
    
    def __init__(
        self,
        param_groups: List[Dict[str, Any]],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.param_groups = param_groups
        self.betas = betas
        self.eps = eps
        self.state: Dict[torch.nn.Parameter, Dict[str, torch.Tensor]] = {}
        
        # Initialize state for all parameters
        for group in self.param_groups:
            for p in group['params']:
                self._init_param_state(p)

    def _init_param_state(self, p: torch.nn.Parameter):
        """Initialize zero momentum and step counts for parameter p."""
        if p not in self.state:
            N = p.shape[0] if p.numel() > 0 else 0
            dev = p.device
            self.state[p] = {
                'step': torch.zeros(N, dtype=torch.long, device=dev),
                'exp_avg': torch.zeros_like(p.data),
                'exp_avg_sq': torch.zeros_like(p.data),
            }

    def zero_grad(self):
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, mask: Optional[torch.Tensor] = None, active_idx: Optional[torch.Tensor] = None):
        """Perform selective Adam update step on active indices only.
        
        Args:
            mask: (N,) boolean tensor of active Gaussians, OR
            active_idx: (M,) 1D tensor of active Gaussian indices.
        """
        beta1, beta2 = self.betas
        
        for group in self.param_groups:
            lr = group.get('lr', 1e-3)
            
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    continue
                    
                N = p.shape[0]
                if p not in self.state or self.state[p]['exp_avg'].shape[0] != N:
                    self._init_param_state(p)
                    
                state = self.state[p]
                
                # Resolve active indices
                if active_idx is not None:
                    indices = active_idx[active_idx < N]
                elif mask is not None:
                    indices = torch.where(mask[:N])[0]
                else:
                    indices = torch.arange(N, device=p.device)
                    
                if len(indices) == 0:
                    continue
                    
                # Extract active gradient slice (M, ...)
                g = p.grad[indices]
                
                # Extract active momentum and variance slices
                m = state['exp_avg'][indices]
                v = state['exp_avg_sq'][indices]
                steps = state['step'][indices] + 1
                state['step'][indices] = steps
                
                # Sliced Adam momentum update: O(M)
                m = beta1 * m + (1.0 - beta1) * g
                v = beta2 * v + (1.0 - beta2) * (g * g)
                
                state['exp_avg'][indices] = m
                state['exp_avg_sq'][indices] = v
                
                # Bias correction per Gaussian step
                # Step tensor shape broadcast
                step_f = steps.float()
                # Expand dims to match parameter rank
                while step_f.ndim < p.ndim:
                    step_f = step_f.unsqueeze(-1)
                    
                bias_correction1 = 1.0 - beta1 ** step_f
                bias_correction2 = 1.0 - beta2 ** step_f
                
                step_size = lr / bias_correction1
                denom = (torch.sqrt(v) / torch.sqrt(bias_correction2)) + self.eps
                
                # Update parameter data strictly for active slice
                p.data[indices] -= step_size * (m / denom)

    def extend_state(self, n_new: int, device: Optional[torch.device] = None):
        """Extend optimizer momentum and variance state for n_new added Gaussians without reset."""
        if n_new <= 0:
            return
            
        for group in self.param_groups:
            for p in group['params']:
                N_curr = p.shape[0]
                dev = device or p.device
                
                if p not in self.state:
                    self._init_param_state(p)
                    continue
                    
                state = self.state[p]
                old_len = state['step'].shape[0]
                
                if old_len < N_curr:
                    diff = N_curr - old_len
                    # Extend step counter with zeros
                    state['step'] = torch.cat([
                        state['step'],
                        torch.zeros(diff, dtype=torch.long, device=dev)
                    ], dim=0)
                    
                    # Extend exp_avg
                    extra_shape = (diff,) + p.shape[1:]
                    state['exp_avg'] = torch.cat([
                        state['exp_avg'],
                        torch.zeros(extra_shape, dtype=p.dtype, device=dev)
                    ], dim=0)
                    
                    # Extend exp_avg_sq
                    state['exp_avg_sq'] = torch.cat([
                        state['exp_avg_sq'],
                        torch.zeros(extra_shape, dtype=p.dtype, device=dev)
                    ], dim=0)

    def prune_state(self, keep_mask: torch.Tensor):
        """Prune state buffers to match pruned parameter tensors."""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    k_mask = keep_mask[:state['step'].shape[0]]
                    state['step'] = state['step'][k_mask]
                    state['exp_avg'] = state['exp_avg'][k_mask]
                    state['exp_avg_sq'] = state['exp_avg_sq'][k_mask]
