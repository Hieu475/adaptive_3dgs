# R30: Comprehensive Selective Optimization Scaling Report

Evaluated with **Real 3DGS Rasterizer + RGB-D Loss** across Gaussian counts and active ratios.

### Systems Break-Even Points ($r^*$ where $\text{Speedup} \approx 1.0\times$)
- **N = 10,000 Gaussians**: $r^* \approx 50.0\%$ (At $r < 50.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)
- **N = 25,000 Gaussians**: $r^* \approx 100.0\%$ (At $r < 100.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)
- **N = 50,000 Gaussians**: $r^* \approx 100.0\%$ (At $r < 100.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)

| N Total | Active Ratio | Active (M) | Masked Bwd (p50) | Selective Bwd (p50) | Bwd Speedup | Total Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | 100% | 10,000 | 103.05 ms | 110.41 ms | **0.93x** | **0.93x** |
| 10,000 | 50% | 5,000 | 108.65 ms | 91.36 ms | **1.19x** | **1.02x** |
| 10,000 | 25% | 2,500 | 113.01 ms | 40.53 ms | **2.79x** | **1.75x** |
| 10,000 | 10% | 1,000 | 113.43 ms | 19.66 ms | **5.77x** | **2.19x** |
| 10,000 | 5% | 500 | 106.76 ms | 12.79 ms | **8.35x** | **2.08x** |
| 10,000 | 2% | 200 | 146.45 ms | 12.57 ms | **11.65x** | **2.20x** |
| 10,000 | 1% | 100 | 170.72 ms | 7.50 ms | **22.75x** | **3.75x** |
| 25,000 | 100% | 25,000 | 178.09 ms | 176.38 ms | **1.01x** | **1.00x** |
| 25,000 | 50% | 12,500 | 176.43 ms | 132.48 ms | **1.33x** | **1.07x** |
| 25,000 | 25% | 6,250 | 178.27 ms | 108.36 ms | **1.65x** | **1.20x** |
| 25,000 | 10% | 2,500 | 180.36 ms | 61.67 ms | **2.92x** | **1.60x** |
| 25,000 | 5% | 1,250 | 182.14 ms | 27.78 ms | **6.56x** | **2.09x** |
| 25,000 | 2% | 500 | 186.48 ms | 13.47 ms | **13.85x** | **2.57x** |
| 25,000 | 1% | 250 | 182.64 ms | 9.83 ms | **18.59x** | **2.46x** |
| 50,000 | 100% | 50,000 | 299.30 ms | 300.87 ms | **0.99x** | **0.96x** |
| 50,000 | 50% | 25,000 | 306.98 ms | 180.19 ms | **1.70x** | **1.21x** |
| 50,000 | 25% | 12,500 | 310.00 ms | 133.78 ms | **2.32x** | **1.49x** |
| 50,000 | 10% | 5,000 | 313.41 ms | 102.38 ms | **3.06x** | **1.64x** |
| 50,000 | 5% | 2,500 | 303.02 ms | 65.72 ms | **4.61x** | **1.86x** |
| 50,000 | 2% | 1,000 | 302.99 ms | 25.21 ms | **12.02x** | **2.38x** |
| 50,000 | 1% | 500 | 300.74 ms | 14.13 ms | **21.29x** | **2.57x** |

