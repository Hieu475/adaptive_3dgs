# Table 2: Utility Prediction & Ranking Fidelity (Claim B)

Evaluated against Ground-Truth Oracle Utility ($U_i^{oracle} = \Delta Q_i / \Delta T_i$).

| Method | Spearman $\rho$ ↑ | Overlap@10% ↑ | Overlap@20% ↑ | Gain Ratio@20% ↑ | Regret@20% ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | -0.0372 | 16.7% | 25.0% | 0.4765 | 0.5235 |
| Error-Only | -0.2691 | 0.0% | 0.0% | 0.1485 | 0.8515 |
| Error × Influence | -0.0367 | 8.3% | 16.7% | 0.4663 | 0.5337 |
| Binary (RTG-SLAM) | -0.0434 | 16.7% | 16.7% | 0.3861 | 0.6139 |
| **Heuristic Utility (Ours)** | **-0.0304** | 33.3% | 25.0% | 0.5145 | 0.4855 |
| **Oracle (Upper Bound)** | **+1.0000** | 100.0% | 100.0% | 1.0000 | 0.0000 |
