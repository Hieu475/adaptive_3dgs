# Online RGB-D 3D Gaussian Splatting with Marginal Utility Estimation under Compute Budget

[![Tests](https://img.shields.io/badge/tests-417%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Custom%20C%2B%2B%2FCUDA-green.svg)](csrc/)
[![Status](https://img.shields.io/badge/Phase%206-FROZEN%20(Case%20B)-blueviolet.svg)](results/phase6_context_utility/manifest.json)

---

## 1. Problem Formulation

Dense online 3D reconstruction from streaming RGB-D sensors requires maintaining photometric and geometric fidelity under rigid per-frame execution deadlines (e.g., $15\text{–}33\text{ ms}$). While 3D Gaussian Splatting (3DGS) provides interactive differentiable rendering, existing SLAM and mapping systems optimize Gaussians indiscriminately or rely on heuristic residual thresholds (e.g., *"high rendering error $\Rightarrow$ optimize"*).

This work establishes that **high residual error does not imply high marginal optimization utility**: on geometric silhouettes, planar surfaces, and saturated regions, naive gradient updates frequently degrade local geometry ($U_i^\star < 0$). We formally re-frame online 3DGS scheduling as **budget-constrained marginal utility optimization**:

$$\max_{S_t \subseteq G_t} \Delta Q(S_t) \quad \text{subject to} \quad C(S_t) \le B_t$$

where the counterfactual (intervention-based) marginal utility of candidate Gaussian $g_i$ under our experimental protocol is defined as:

$$U_i^\star = \frac{\Delta Q_i^{\text{intervention}}}{C_i} \in \mathbb{R}$$

- $\Delta Q_i^{\text{intervention}} = Q(G_t \cup \{\Delta \theta_i\}) - Q(G_t)$ is the realized change in global reconstruction quality under isolated trial intervention (combining photometric PSNR and geometric depth fidelity).
- $C_i$ is the empirical execution time cost (ms).
- Under this intervention protocol, $U_i^\star$ quantifies the counterfactual intervention effect of allocating gradient updates to primitive $g_i$.
- When optimization causes depth tearing or appearance degradation, $U_i^\star < 0$, providing an explicit penalty signal without artificial zero-clamping.

---

## 2. Research Questions (RQs)

| Research Question | Core Hypothesis | Status & Empirical Answer |
| :--- | :--- | :--- |
| **RQ1: State Predictability** | Can observable Gaussian state variables $s_i$ predict marginal reconstruction gain $U_i^\star$? | **Answered Affirmatively (Moderate Signal)**: TwoHeadMLP achieves cross-scene rank correlation $\rho = 0.2035 \pm 0.172$ and $\text{NDCG@20} = 0.4566$ on unseen test scene `tum_fr2_xyz`, significantly exceeding random and linear baselines. |
| **RQ2: Budgeted Selection** | Does utility prediction improve Gaussian selection efficiency under equal compute budgets ($\hat{U}_i \to S_B$)? | **Answered Affirmatively under Tight Budgets**: At $B \le 20\%$, TwoHeadMLP achieves $\text{OSE} = 0.497 \pm 0.102$ vs Error $\text{OSE} = 0.239$ (**$+108.0\%$ relative gain**). Converges toward heuristic baseline at high budgets ($B \ge 60\%$). |
| **RQ3: Contextual Re-Ranking** | Does conditioning on selected context $S_t$ modify candidate ranking and improve online quality? | **Resolved as Case B (Negative Result for Re-Ranking)**: Context modulates utility magnitude ($U^*(i \mid S) \neq U^*(i \mid \emptyset)$), but within-frame candidate priority rank is substantially stable ($\bar{\rho}_{\text{rank}} = 0.8916$, Overlap@5 = $80.0\%$). Adaptive greedy re-ranking adds computational overhead without realized quality gains over static selection. |

---

## 3. Method Overview

```
RGB-D Frame (I_t, D_t)
         │
         ▼
3D Gaussian State (μ, Σ, α, c)
         │
         ▼
Observable Features (s_i ∈ R^11) ──► [Pre-fusion Normalizer]
         │
         ▼
Utility Predictor (TwoHeadMLP) ──► Decoupled ΔQ_hat and C_hat
         │
         ▼
Budget-Constrained Selection (S_B) ──► Knapsack greedy under Σ C_i ≤ B
         │
         ▼
Selective Optimization (SelectiveAdam) ──► Update only S_B; Background Cache
         │
         ▼
Optimized Scene State G_{t+1}
```

1. **State Observation**: Extracts 11 canonical features per Gaussian: photometric residual, depth residual, gradient norm, screen-space visibility, attribution mass, positional drift, residual EMA drift, temporal drift, uncertainty, projected area, and update age.
2. **Predictor Architecture**: Two-Head decoupled MLP predicting marginal quality gain $\widehat{\Delta Q}_i$ and execution cost $\widehat{C}_i$ with softplus cost enforcement, trained with difference-weighted pairwise ranking loss.
3. **Budgeted Selection**: Greedy selection maximizing collective utility within scheduled budget deadline $B_t$, rejecting negative predicted utilities ($\hat{U}_i \le 0$).
4. **Selective Optimization**: `SelectiveAdam` executes gradient descent exclusively on selected subset $S_B$, while `FrozenBackgroundCache` preserves unselected map regions.

---

## 4. Scientific Findings & Empirical Gates

| Phase / Gate | Core Question | Empirical Finding & Authoritative Metric |
| :--- | :--- | :--- |
| **Oracle (Gate 1)** | Is marginal utility measurable and empirically non-negative? | **Yes (Measurable & Frequently Negative)**: Evaluated on TUM RGB-D (`freiburg1_desk`); positive headroom $H = +0.000149 > 0$. Crucially, **$20.5\%$** of interventions yield negative utility ($U_i^\star < 0$), refuting non-negativity assumptions. Concurrent optimization is severely sub-additive ($R_{add} = 0.2249$ at $|S|=4$, $R_{add} = 0.0048$ at $|S|=16$). |
| **Phase 4 (Gate 2)** | Can observable state predict marginal utility? | **Limited but Non-Trivial**: TwoHeadMLP achieves $\rho = +0.2035 \pm 0.172$, $\text{NDCG@20} = 0.4566$, $\text{OSE@20} = 0.497 \pm 0.102$, outperforming error-only heuristics on zero-shot cross-scene transfer (`tum_fr2_xyz`). |
| **Phase 5 (Gate 3)** | Does utility prediction improve budgeted selection? | **Substantial Gain at Tight Budgets**: At $10\%\text{–}20\%$ budget, TwoHeadMLP delivers nearly double the selection efficiency of error ranking ($+\text{92.6}\%\text{–}+108.0\%$). In multi-frame online SLAM, reduces optimization latency by $33.6\%$ vs heuristic knapsack and $47.7\%$ vs error-only while matching reconstruction PSNR. |
| **Phase 6 (Gate 6A-6E)** | Does context alter marginal utility and candidate ranking? | **Magnitude Shifts, Candidate Priority Substantially Stable**: Screen-space co-visibility modulates utility sub-additively ($\rho(\text{IoU}, \|I\|) = 0.5357, p = 0.0048$). However, candidate rank remains substantially stable: $\bar{\rho}_{\text{rank}} = \mathbf{0.8916} \pm \mathbf{0.1104}$, Kendall $\bar{\tau} = \mathbf{0.8043}$, and Top-5 candidate overlap reaches $\mathbf{80.0\%}$ across 27 full-coverage groups. |
| **Phase 6 (Case B)** | Does adaptive contextual re-ranking improve selection? | **No Statistically Significant Gain**: Oracle Context Advantage $\equiv +0.00 \times 10^{-5}$ ($Q_{\text{OracleCond}} \equiv Q_{\text{OracleStatic}}$); 5-seed online selection gain $Q(P_6) - Q(P_4)$ spans zero across all budgets (Wilcoxon $p \ge 0.7035$). Pointwise ranking already captures most of the observed candidate priority structure. |

---

## 5. Core Scientific Narrative (Case B: No-Spin Honest Science)

The empirical trajectory across Phases 1 through 6 establishes a coherent, non-trivial scientific insight:

$$\begin{aligned}
\text{Co-visibility \& alpha-compositing interaction} &\implies U^*(i \mid S) \neq U^*(i \mid \emptyset) \quad (\text{Interaction Exists}) \\
&\implies \text{The observed pattern is consistent with alpha-compositing attenuation} \\
&\implies \operatorname{rank}(U^*(i \mid S)) \approx \operatorname{rank}(U^*(i \mid \emptyset)) \quad (\bar{\rho}_{\text{rank}} = 0.8916, \text{Overlap@5} = 80.0\%) \\
&\implies \text{Pointwise utility } U^*(i \mid \emptyset) \text{ already captures most of the observed candidate priority structure} \\
&\implies \text{Adaptive contextual selection increases selection-stage latency by } 424.2\times \text{ with no quality benefit}
\end{aligned}$$

Rather than claiming that contextual modeling enhances selection, our study **empirically characterizes contextual marginal utility and demonstrates that, under the evaluated online RGB-D setting, contextual interactions modulate utility magnitude while preserving substantial candidate rank stability, rendering adaptive greedy re-ranking unnecessary under the tested budget regime.**

---

## 6. Limitations

To maintain scientific integrity, four fundamental limitations are explicitly acknowledged:

1. **Short-Horizon Oracle**: Counterfactual interventions evaluate short trial horizons ($M=5$ gradient steps). While computationally tractable for collecting thousands of ground-truth points, $U_i^\star$ reflects immediate local improvement rather than long-horizon trajectory dynamics over distant keyframes.
2. **Non-Additivity & Submodularity Limits**: Direct measurements confirm severe sub-additivity ($\Delta Q(S) \neq \sum_i \Delta Q_i$, with $R_{add} \to 0.0048$ at $|S|=16$). Pointwise greedy selection relies on a linear additive proxy; rigorous combinatorial submodular guarantees require conditions that do not strictly hold in dense alpha-blended rendering.
3. **Dataset & Scene Coverage**: Confirmatory experiments are evaluated on real indoor sequences from the TUM RGB-D benchmark (`tum_fr1_desk` and `tum_fr2_xyz`). Extrapolation to wide-baseline outdoor environments, dynamic scenes, or extreme viewpoint shifts remains to be characterized.
4. **Systems Overhead vs Model Inference**: Profiling reveals an essential AI Systems insight: **neural model inference itself is negligible ($T_{\text{MLP}} \approx 1.12\text{ ms}$, $0.1\%$ of stage time), but state construction ($225.79\text{ ms}$), context extraction ($66.84\text{ ms}$), and adaptive selection ($76.36\text{ ms}$) dominate the pipeline.** State management and orchestration, rather than deep learning compute, are the primary bottlenecks in real-time execution.

---

## 7. AI Systems & Runtime Breakdown

Under a target budget $B_{\text{sched}} = 15.0\text{ ms}$, the per-stage execution times for Phase 4 (Pointwise) vs Phase 6 (Context-Aware) are:

$$T_{\text{stage}} = T_{\text{feature}} + T_{\text{context}} + T_{\text{MLP}} + T_{\text{selection}} + T_{\text{optimization}}$$

| Pipeline Stage | Symbol | Phase 4 (Pointwise) | Phase 6 (Context-Aware) | Breakdown (%) | Computational Nature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Feature Extraction** | $T_{\text{feat}}$ | ~12.5 ms | **225.79 ms** | 24.4% | Attribution rendering & error mass aggregation |
| **Context Construction** | $T_{\text{ctx}}$ | 0.0 ms | **66.84 ms** | 7.2% | Spatial KNN search & screen-space IoU projection |
| **Prediction Inference** | $T_{\text{MLP}}$ | ~0.85 ms | **1.12 ms** | 0.1% | 2-Head Residual Context MLP forward pass |
| **Subset Selection** | $T_{\text{sel}}$ | **0.18 ms** | **76.36 ms** | 8.3% | Iterative adaptive greedy sort across $|S_B|$ steps |
| **Gaussian Optimization** | $T_{\text{opt}}$ | ~550 ms | **554.48 ms** | 60.0% | Selective Adam backward passes & parameter updates |
| **Total Pipeline Stage** | $T_{\text{total}}$ | **~563.5 ms** | **924.59 ms** | 100.0% | $1.64\times$ total stage runtime penalty |

Specifically, adaptive contextual selection increases the selection-stage latency by $424.2\times$ ($76.36\text{ ms}$ vs $0.18\text{ ms}$), with a $+361.06\text{ ms}$ ($+64.1\%$) total pipeline stage latency overhead ($924.59\text{ ms}$ vs $563.53\text{ ms}$). Coupled with Case B rank stability ($\bar{\rho}_{\text{rank}} = 0.8916$, Overlap@5 = $80.0\%$), paying this runtime penalty yields zero statistically significant quality gain in online reconstruction.

---

## 8. Repository Structure & Single Source of Truth

All experimental numbers, checkpoints, and reports in this repository adhere to a **Single Source of Truth** rooted in frozen artifacts:

- **Manifest Single Source of Truth**: [`results/phase6_context_utility/manifest.json`](results/phase6_context_utility/manifest.json)
- **Rank Stability Analysis**: [`results/phase6_context_utility/rank_stability_analysis.json`](results/phase6_context_utility/rank_stability_analysis.json)
- **Ablation Ladder Summary**: [`results/phase6_context_utility/ablation/ablation_summary.json`](results/phase6_context_utility/ablation/ablation_summary.json)
- **Oracle Gap & Regret**: [`results/phase6_context_utility/oracle_gap_and_regret.json`](results/phase6_context_utility/oracle_gap_and_regret.json)
- **Runtime Breakdown**: [`results/phase6_context_utility/runtime_breakdown.json`](results/phase6_context_utility/runtime_breakdown.json)

```
adaptive_3dgs/
├── configs/
│   ├── default_config.yaml         # Base SLAM and rendering configuration
│   └── protocol_v1.yaml            # Frozen scientific confirmatory protocol
├── csrc/                           # Custom CUDA rasterizer & backward kernels
├── datasets/                       # Dataset loaders (TUM RGB-D, Replica)
├── experiments/                    # Scientific evaluation & reproduction scripts
│   ├── run_gate1_headroom.py       # Gate 1 & Headroom verification
│   ├── train_utility_model.py      # Phase 4 Two-Head utility training (5 seeds)
│   ├── eval_utility_model.py       # Phase 4 RQ1 prediction fidelity
│   ├── eval_selection.py           # Phase 4 RQ2 budget selection sweep
│   ├── run_phase5_budget_benchmark.py  # Phase 5 Stage A controlled budget benchmark
│   ├── run_phase5_online_trajectory.py # Phase 5 Stage B online trajectory
│   ├── run_phase6_rank_stability.py# Phase 6 Candidate coverage & rank stability audit
│   ├── run_phase6_ablation.py      # Phase 6 8-variant architecture ablation ladder
│   ├── run_phase6_oracle_gap.py    # Phase 6 5-policy oracle decomposition
│   ├── run_phase6_selection.py     # Phase 6 Multi-seed budget & safety sweep
│   └── run_phase6_runtime_profile.py   # Phase 6 Runtime profiling & latency breakdown
├── research/                       # Core algorithms & scientific modules
│   ├── gaussian_model.py           # 3D Gaussian representation & state
│   ├── oracle_utility.py           # Counterfactual intervention engine
│   ├── utility_models.py           # TwoHeadMLP and baseline architectures
│   ├── utility_features.py         # 11 canonical observable features
│   ├── phase5_selection.py         # Budget-constrained knapsack selection
│   ├── phase6_context.py           # Context representation (KNN, screen IoU, S_t)
│   ├── phase6_model.py             # ResidualContextModel with frozen P4 backbone
│   ├── phase6_dataset.py           # GroupedBatchSampler & train-only normalizer
│   ├── phase6_oracle.py            # Conditional oracle U*(i|S_t) engine
│   └── phase6_selection.py         # Adaptive greedy selection dispatch
├── results/
│   ├── learned_utility/            # Phase 4 evaluation tables & checkpoints
│   ├── phase5_budget_selection/    # Phase 5 budget sweep & trajectory data
│   └── phase6_context_utility/     # Phase 6 FROZEN authoritative artifacts
│       ├── manifest.json           # Phase 6 Single Source of Truth
│       ├── rank_stability_analysis.json # 27 exact groups, 100% pool coverage
│       ├── oracle_gap_and_regret.json   # 5-policy oracle decomposition
│       ├── runtime_breakdown.json       # T_P6 pipeline latency breakdown
│       ├── ablation/               # 8-variant ablation ladder & checkpoints
│       └── datasets/               # Context-centric conditional oracle dataset
└── tests/                          # Comprehensive test suite (417/417 PASS)
```

---

## 9. Quickstart & Reproduction

### Test Suite Execution
```bash
pytest -q
# Expected: 417 passed in ~8s (100% PASS)
```

### Reproducing Authoritative Phase 6 Artifacts
```bash
# 1. Candidate pool coverage audit & rank stability (Case B proof)
python3 experiments/run_phase6_rank_stability.py

# 2. 8-Variant architecture ablation ladder
python3 experiments/run_phase6_ablation.py --seed 42

# 3. 5-Policy oracle gap & regret decomposition
python3 experiments/run_phase6_oracle_gap.py

# 4. Multi-seed budget sweep & statistical tests (5 seeds)
python3 experiments/run_phase6_selection.py --seeds 42 43 44 45 46

# 5. Pipeline runtime profiling & decision latency breakdown
python3 experiments/run_phase6_runtime_profile.py
```

---

## 10. Research Provenance & Frozen Integrity

Every scientific number in this repository can be reverse-traced to exact source files:

$$\text{Reported Metric} \longrightarrow \text{Authoritative Artifact} \longrightarrow \text{Evaluation Script} \longrightarrow \text{Dataset Hash} \longrightarrow \text{Frozen Checkpoint} \longrightarrow \text{Git Commit}$$

- **Authoritative Provenance**: `results/phase6_context_utility/manifest.json`
- **Backbone Model Invariance**: Phase 4 checkpoint `two_head_mlp_seed_42.pt` SHA-256 hash verified bitwise immutable during all Phase 6 operations.
- **Data Integrity**: Zero synthetic baseline defaulting; all rank stability metrics evaluate strictly on measured candidate vectors.
