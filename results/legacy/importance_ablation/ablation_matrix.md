# R35: Systematic Utility Ablation Matrix (V0 → V6)

| Variant | $\rho(U, U_{oracle})$ | Overlap@10% | Coverage@10% | Regret@10% | Jitter (ms) | Switch Rate | PSNR (dB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V0: Error Only** | 0.0923 | 0.0% | 0.0666 | 0.8223 | 55.99 ms | 0.00 | 8.77 dB |
| **V1: Error + Influence** | -0.0279 | 0.0% | 0.0691 | 0.8256 | 9.56 ms | 0.00 | 8.77 dB |
| **V2: + Temporal Dynamics** | 0.0383 | 0.0% | 0.0691 | 0.8256 | 21.37 ms | 0.00 | 8.77 dB |
| **V3: + Uncertainty** | -0.0671 | 0.0% | 0.0691 | 0.8256 | 15.89 ms | 0.00 | 8.77 dB |
| **V4: + Projected Area** | 0.0338 | 0.0% | 0.0738 | 0.8015 | 7.07 ms | 0.00 | 8.77 dB |
| **V5: + Hysteresis** | -0.0104 | 0.0% | 0.0738 | 0.7876 | 8.97 ms | 165.20 | 8.77 dB |
| **V6: Full Utility (+ Prior)** | 0.0354 | 0.0% | 0.0802 | 0.6571 | 5.61 ms | 165.20 | 8.77 dB |

