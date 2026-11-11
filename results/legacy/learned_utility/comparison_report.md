# Step 17: Learned Utility vs Heuristic Utility Benchmark

| Method | $\rho(U, U_{oracle})$ | Overlap@10% | Overlap@20% | Gain Ratio@20% | Regret@10% |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Error-Only** | 0.1037 | 0.0% | 0.0% | 0.4851 | 0.5149 |
| **2. Error × Influence** | 0.0995 | 0.0% | 0.0% | 0.4851 | 0.5149 |
| **3. Heuristic (Ours V6)** | 0.1037 | 0.0% | 0.0% | 0.4851 | 0.5149 |
| **4. Learned MLP (64→32→1)** | 0.0545 | 20.0% | 40.0% | 0.5698 | 0.4302 |
| **5. Oracle (Upper Bound)** | 1.0000 | 100.0% | 100.0% | 1.0000 | 0.0000 |

