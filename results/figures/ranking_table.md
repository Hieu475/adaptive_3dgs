# Phase 3: Heuristic Utility Validation Benchmark

Evaluated against Ground-Truth Oracle Marginal Utility ($U_i^\star = \Delta Q_i / \Delta T_i$).

| Method | $\rho(U^\star_{joint})$ ↑ | $\rho(U^\star_{rgb})$ | $\rho(U^\star_{depth})$ | NDCG@20% ↑ | Overlap@20% ↑ | OSE@20% ↑ | Regret@20% ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | -0.0537 | -0.1455 | +0.0768 | 0.8987 | 18.8% | 0.3497 | +1.403313 |
| Color-Error Alone | +0.1529 | +0.0602 | -0.0028 | 0.9152 | 31.2% | 0.5034 | +1.071582 |
| Depth-Error Alone | +0.0355 | +0.0349 | +0.3546 | 0.8882 | 12.5% | 0.1890 | +1.750055 |
| Error-Only (RGB + Depth) | +0.0893 | +0.0495 | +0.2330 | 0.8919 | 15.6% | 0.2170 | +1.689632 |
| Error × Influence | +0.0011 | +0.0506 | +0.0809 | 0.8881 | 3.1% | 0.2140 | +1.696111 |
| Temporal Drift Alone | NaN | NaN | NaN | 0.9005 | 25.0% | 0.3507 | +1.401180 |
| Binary (RTG-SLAM) | -0.0552 | -0.1290 | +0.0593 | 0.8579 | 18.8% | -0.0240 | +2.209872 |
| **Heuristic Utility (Ours)** | **+0.1136** | +0.0318 | +0.1576 | 0.8996 | 18.8% | **0.3230** | +1.461066 |
| **Oracle (Upper Bound)** | **+1.0000** | +0.6725 | +0.4035 | 1.0000 | 100.0% | **1.0000** | +0.000000 |
