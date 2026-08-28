# R32 Cost Model Calibration Report

## Calibrated Latency Model
$$T(M) = 127.537 + 0.011258 \times M$$

- **Goodness of Fit ($R^2$)**: **0.9052**
- **Mean Absolute Error (MAE)**: **23.262 ms**

| Active Ratio | Active ($M$) | Actual Measured ($T$) | Predicted ($T$) | Residual Error |
|:---:|:---:|:---:|:---:|:---:|
| 5.0% | 1,000 | 120.62 ms | 138.80 ms | 18.18 ms |
| 10.0% | 2,000 | 180.93 ms | 150.05 ms | 30.87 ms |
| 20.0% | 4,000 | 142.72 ms | 172.57 ms | 29.86 ms |
| 25.0% | 5,000 | 203.84 ms | 183.83 ms | 20.01 ms |
| 50.0% | 10,000 | 221.58 ms | 240.12 ms | 18.55 ms |
| 75.0% | 15,000 | 326.95 ms | 296.41 ms | 30.54 ms |
| 100.0% | 20,000 | 337.87 ms | 352.71 ms | 14.84 ms |

