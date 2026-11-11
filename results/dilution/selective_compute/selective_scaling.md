# R30: Comprehensive Selective Optimization Scaling Report

Evaluated with **Real 3DGS Rasterizer + RGB-D Loss** across Gaussian counts and active ratios (Pure Optimization Step Timing).

### Systems Break-Even Points ($r^*$ where $\text{Speedup} \approx 1.0\times$)
- **N = 10,000 Gaussians**: $r^* \approx 100.0\%$ (At $r \le 100.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)
- **N = 25,000 Gaussians**: $r^* \approx 100.0\%$ (At $r \le 100.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)
- **N = 50,000 Gaussians**: $r^* \approx 100.0\%$ (At $r \le 100.0\%$, True Selective Optimization delivers strict speedup over Full/Masked Baseline)

| N Total | Active Ratio | Active (M) | Active Render | Masked Bwd (p50) | Selective Bwd (p50) | Bwd Speedup | Masked Opt | Selective Opt | Opt Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | 100% | 10,000 | 59.99 ms | 120.15 ms | 120.37 ms | **1.00x** | 176.46 ms | 193.86 ms | **0.91x** |
| 10,000 | 50% | 5,000 | 44.21 ms | 123.35 ms | 97.93 ms | **1.26x** | 183.09 ms | 144.83 ms | **1.26x** |
| 10,000 | 25% | 2,500 | 28.69 ms | 115.79 ms | 57.28 ms | **2.02x** | 186.04 ms | 80.45 ms | **2.31x** |
| 10,000 | 10% | 1,000 | 31.38 ms | 109.88 ms | 70.55 ms | **1.56x** | 165.29 ms | 108.86 ms | **1.52x** |
| 10,000 | 5% | 500 | 8.45 ms | 109.25 ms | 14.21 ms | **7.69x** | 166.81 ms | 23.76 ms | **7.02x** |
| 10,000 | 2% | 200 | 7.48 ms | 137.01 ms | 9.75 ms | **14.06x** | 199.74 ms | 18.99 ms | **10.52x** |
| 10,000 | 1% | 100 | 7.79 ms | 164.58 ms | 20.42 ms | **8.06x** | 242.67 ms | 39.52 ms | **6.14x** |
| 25,000 | 100% | 25,000 | 141.02 ms | 247.09 ms | 197.59 ms | **1.25x** | 440.07 ms | 348.09 ms | **1.26x** |
| 25,000 | 50% | 12,500 | 63.24 ms | 182.55 ms | 136.92 ms | **1.33x** | 296.53 ms | 204.96 ms | **1.45x** |
| 25,000 | 25% | 6,250 | 62.54 ms | 176.39 ms | 121.77 ms | **1.45x** | 290.13 ms | 183.23 ms | **1.58x** |
| 25,000 | 10% | 2,500 | 24.47 ms | 183.47 ms | 61.21 ms | **3.00x** | 299.17 ms | 86.87 ms | **3.44x** |
| 25,000 | 5% | 1,250 | 14.67 ms | 187.95 ms | 27.66 ms | **6.79x** | 312.80 ms | 44.25 ms | **7.07x** |
| 25,000 | 2% | 500 | 8.90 ms | 185.50 ms | 14.18 ms | **13.08x** | 306.60 ms | 24.63 ms | **12.45x** |
| 25,000 | 1% | 250 | 7.99 ms | 176.89 ms | 19.52 ms | **9.06x** | 288.64 ms | 28.62 ms | **10.08x** |
| 50,000 | 100% | 50,000 | 202.14 ms | 304.65 ms | 309.72 ms | **0.98x** | 529.93 ms | 521.88 ms | **1.02x** |
| 50,000 | 50% | 25,000 | 113.38 ms | 308.76 ms | 183.37 ms | **1.68x** | 521.40 ms | 302.40 ms | **1.72x** |
| 50,000 | 25% | 12,500 | 75.34 ms | 294.12 ms | 138.31 ms | **2.13x** | 504.12 ms | 217.77 ms | **2.31x** |
| 50,000 | 10% | 5,000 | 53.79 ms | 299.90 ms | 109.43 ms | **2.74x** | 502.57 ms | 166.55 ms | **3.02x** |
| 50,000 | 5% | 2,500 | 26.59 ms | 314.54 ms | 68.04 ms | **4.62x** | 549.99 ms | 98.81 ms | **5.57x** |
| 50,000 | 2% | 1,000 | 13.71 ms | 320.67 ms | 25.20 ms | **12.72x** | 631.86 ms | 39.72 ms | **15.91x** |
| 50,000 | 1% | 500 | 9.61 ms | 308.67 ms | 15.62 ms | **19.76x** | 517.87 ms | 26.79 ms | **19.33x** |

