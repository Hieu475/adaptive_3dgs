# Oracle Compositing & Gradient Fidelity Verification Report

**Scene:** `tum_fr2_xyz` | **Frames:** 3 | **Active Ratio:** 20.0%

## 1. Summary of Gradient Discrepancy

- **Mean Relative Gradient Error**: `0.3144` (31.44%)
- **Max Relative Gradient Error**: `0.3300` (33.00%)
- **Mean Photometric Loss Discrepancy**: `0.000448`

## 2. Per-Frame Measurements

| Frame | $L_{full}$ | $L_{selective}$ | PSNR Full | PSNR Sel | Relative Grad Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.04823 | 0.04856 | 13.17 dB | 13.14 dB | **0.2952** (29.5%) |
| 2 | 0.04835 | 0.04877 | 13.16 dB | 13.12 dB | **0.3300** (33.0%) |
| 3 | 0.04877 | 0.04936 | 13.12 dB | 13.07 dB | **0.3181** (31.8%) |

## 3. Scientific Finding on Oracle Validity

> [!WARNING]
> **Depth Sorting Violation Confirmed:** Relative gradient error is 31.4% (> 5%). Because active and frozen Gaussians interleave along viewing rays, affine compositing introduces systematic gradient distortion. For authoritative oracle intervention targets ($U_i^\star$), exact full rendering must be used.