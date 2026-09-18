"""Loss functions for 3D Gaussian Splatting optimization.

Baseline objective: L = w_c·L_color + w_d·L_depth + w_n·L_normal + w_reg·L_reg
Adaptive: L_total = L_rgbd + λ_geo·L_geometry + λ_temp·L_temporal + λ_comp·L_compact
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional


def color_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 photometric loss: L_color = |C_gt - C_render|.
    
    Args:
        pred: (H, W, 3) or (B, 3, H, W) predicted color
        gt: same shape, ground truth color
        mask: optional valid-pixel mask
    
    Returns:
        Scalar loss
    """
    diff = (pred - gt).abs()
    if mask is not None:
        diff = diff[mask]
    return diff.mean()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, device: torch.device) -> torch.Tensor:
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(device)
    return window


def ssim_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Differentiable structural similarity loss: L_ssim = 1 - SSIM(pred, gt).
    
    Args:
        pred: (H, W, 3) predicted image
        gt: (H, W, 3) ground truth image
        window_size: Gaussian window kernel size
    
    Returns:
        Scalar loss = 1 - mean(SSIM)
    """
    device = pred.device
    channel = pred.shape[-1]
    
    # Permute to (1, C, H, W)
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = gt.permute(2, 0, 1).unsqueeze(0)
    
    window = create_window(window_size, channel, device)
    
    mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - ssim_map.mean()


def depth_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 geometric loss: L_depth = |D_gt - D_render|.
    
    Args:
        pred: (H, W) predicted depth
        gt: (H, W) ground truth depth
        valid_mask: (H, W) boolean, True where depth is valid
    
    Returns:
        Scalar loss
    """
    if valid_mask is None:
        valid_mask = gt > 0  # Assume 0 = invalid
    return F.l1_loss(pred[valid_mask], gt[valid_mask])


def normal_consistency_loss(
    pred_normals: torch.Tensor,
    pseudo_normals: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normal consistency loss via cosine similarity.
    
    L_normal = 1 - (n_pred · n_pseudo) averaged over valid pixels.
    
    Args:
        pred_normals: (H, W, 3) or (N, 3) predicted normals
        pseudo_normals: same shape, derived from depth map
        valid_mask: optional mask
    
    Returns:
        Scalar loss
    """
    # Normalize
    pred_n = F.normalize(pred_normals, dim=-1)
    pseudo_n = F.normalize(pseudo_normals, dim=-1)
    
    cos_sim = (pred_n * pseudo_n).sum(dim=-1)  # dot product
    loss = 1.0 - cos_sim
    
    if valid_mask is not None:
        loss = loss[valid_mask]
    return loss.mean()


def robust_loss(
    residual: torch.Tensor,
    loss_type: str = 'charbonnier',
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Robust loss functions that downweight outliers.
    
    Charbonnier: L = sqrt(r² + ε²) - ε
    Huber: L = 0.5*r² if |r|<δ else δ*(|r|-0.5*δ)
    
    Args:
        residual: arbitrary shape tensor of residuals
        loss_type: 'charbonnier' or 'huber'
        epsilon: robustness parameter
    
    Returns:
        Loss tensor of same shape
    """
    if loss_type == 'charbonnier':
        return torch.sqrt(residual ** 2 + epsilon ** 2) - epsilon
    elif loss_type == 'huber':
        return F.huber_loss(residual, torch.zeros_like(residual),
                           reduction='none', delta=epsilon)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compact_loss(opacities: torch.Tensor) -> torch.Tensor:
    """Penalize redundant Gaussians to encourage compactness.
    
    Entropy-based: encourages opacity to be close to 0 or 1.
    L_compact = -Σ [α·log(α) + (1-α)·log(1-α)] / N
    
    Args:
        opacities: (N, 1) opacity values in [0, 1]
    
    Returns:
        Scalar loss
    """
    alpha = opacities.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
    return entropy.mean()


def temporal_loss(
    prev_positions: torch.Tensor,
    curr_positions: torch.Tensor,
) -> torch.Tensor:
    """Encourage scene stability across frames.
    
    L_temporal = ||μ_t - μ_{t-1}||² averaged over Gaussians.
    
    Args:
        prev_positions: (N, 3) previous frame positions
        curr_positions: (N, 3) current frame positions
    
    Returns:
        Scalar loss
    """
    return ((curr_positions - prev_positions) ** 2).sum(dim=-1).mean()


def total_loss(
    pred_color: torch.Tensor,
    gt_color: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    weights: Dict[str, float],
    pred_normals: Optional[torch.Tensor] = None,
    pseudo_normals: Optional[torch.Tensor] = None,
    opacities: Optional[torch.Tensor] = None,
    prev_positions: Optional[torch.Tensor] = None,
    curr_positions: Optional[torch.Tensor] = None,
    depth_valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Combined loss with all components.
    
    L = w_c·L_color + w_d·L_depth + w_n·L_normal + w_reg·L_compact + w_t·L_temporal
    
    Args:
        pred_color, gt_color: color images
        pred_depth, gt_depth: depth maps
        weights: dict with keys 'color', 'depth', 'normal', 'compact', 'temporal'
        pred_normals, pseudo_normals: optional normal maps
        opacities: optional (N, 1) for compact loss
        prev_positions, curr_positions: optional for temporal loss
        depth_valid_mask: optional mask
    
    Returns:
        Dict with 'total' and individual loss components
    """
    losses = {}
    
    l_color = color_loss(pred_color, gt_color)
    l_depth = depth_loss(pred_depth, gt_depth, depth_valid_mask)
    losses['color'] = l_color
    losses['depth'] = l_depth
    
    w_color = weights.get('color', 0.8)
    w_depth = weights.get('depth', 0.5)
    w_ssim = weights.get('ssim', 0.2)
    
    total = w_color * l_color + w_depth * l_depth
    if w_ssim > 0.0 and pred_color.shape[0] >= 11 and pred_color.shape[1] >= 11:
        l_ssim = ssim_loss(pred_color, gt_color)
        losses['ssim'] = l_ssim
        total = total + w_ssim * l_ssim
    
    if pred_normals is not None and pseudo_normals is not None:
        l_normal = normal_consistency_loss(pred_normals, pseudo_normals)
        losses['normal'] = l_normal
        total = total + weights.get('normal', 0.1) * l_normal
    
    if opacities is not None:
        l_compact = compact_loss(opacities)
        losses['compact'] = l_compact
        total = total + weights.get('compact', 0.01) * l_compact
    
    if prev_positions is not None and curr_positions is not None:
        l_temporal = temporal_loss(prev_positions, curr_positions)
        losses['temporal'] = l_temporal
        total = total + weights.get('temporal', 0.01) * l_temporal
    
    losses['total'] = total
    return losses
