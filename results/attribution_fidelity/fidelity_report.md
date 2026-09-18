# Scientific Validation: Exact vs. Fast Approximate Attribution

Evaluation of fidelity and speedup when substituting Exact Pixel-Weighted Attribution ($\sum_u w_{u,i} e(u) / \sum_u w_{u,i}$) with Fast Approximate Attribution (Center-Sampled Projection):

- **Spearman Rank Correlation $\rho$**: **0.9079** (High ranking preservation)
- **Pearson Linear Correlation $r$**: **0.9028**
- **Mean Latency**: Exact = **523.90 ms** vs. Fast = **0.85 ms**
- **Speedup**: **614.5x**

### Per-Frame Metrics

| Frame | Gaussians | Visible | Spearman $\rho$ | Pearson $r$ | Exact Latency | Fast Latency | Speedup | Top-100 Overlap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 16008 | 15805 | 0.9084 | 0.9169 | 585.1 ms | 0.98 ms | 596.6x | 37.0% |
| 2 | 16008 | 15638 | 0.9152 | 0.9170 | 523.0 ms | 0.79 ms | 662.5x | 37.0% |
| 3 | 16008 | 15219 | 0.9085 | 0.8960 | 492.1 ms | 0.87 ms | 563.9x | 39.0% |
| 4 | 16008 | 14923 | 0.8993 | 0.8813 | 495.4 ms | 0.77 ms | 639.7x | 35.0% |
