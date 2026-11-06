# Adaptive 3DGS — Research Upgrade Plan
## From Pointwise Gaussian Utility Prediction to Conditional, Interaction-Aware Budget Allocation

## 0. Mục tiêu tài liệu

Tài liệu này trình bày hướng phát triển tiếp theo cho `adaptive_3dgs` sau Phase 12, với mục tiêu không chỉ “làm paper được” mà tìm một formulation có cơ sở khoa học sâu hơn và có khả năng tạo ra cải thiện thực sự.

Trạng thái hiện tại đã cho thấy:

\[
s_i \rightarrow \hat U_i
\]

chưa đủ mạnh. Controlled M0–M4 cho thấy GradNorm vẫn có pointwise ranking tốt hơn learned model, trong khi thêm Positive Utility Head chỉ giúp cải thiện so với M2 và Global Context không giúp thêm. Vì vậy, vấn đề tiếp theo cần được nghiên cứu không nên là “tăng capacity của MLP”, mà là:

\[
\boxed{
\text{pointwise utility có thực sự là sufficient statistic cho bài toán subset selection hay không?}
}
\]

---

# 1. Kết luận khoa học quan trọng từ Phase 12

## 1.1 Kết quả M0–M4

Các formulation hiện tại:

- **M0:** Error Heuristic
- **M1:** GradNorm
- **M2:** Current TwoHeadMLP
- **M3:** M2 + Positive Utility Head
- **M4:** M3 + Global Frame Context

Kết quả chính:

| Model | Spearman \(\rho\) | AUROC | NDCG@20 | OSE | Realized \(\Delta Q\) |
|---|---:|---:|---:|---:|---:|
| M0 Error | 0.3552 | 0.6277 | 0.2434 | 0.2273 | \(3.26\times10^{-4}\) |
| M1 GradNorm | **0.5143** | **0.6995** | **0.4068** | **0.4857** | **\(6.96\times10^{-4}\)** |
| M2 Current MLP | 0.2500 | 0.5879 | 0.3768 | 0.2697 | \(3.86\times10^{-4}\) |
| M3 + Positive Head | 0.2153 | 0.6502 | 0.3656 | 0.3652 | \(5.23\times10^{-4}\) |
| M4 + Global Context | 0.0666 | 0.5877 | 0.3167 | 0.2768 | \(3.97\times10^{-4}\) |

Điều này cho thấy:

\[
M_1 > M_2,M_3,M_4
\]

trên nhiều pointwise metrics.

Nhưng đây chưa phải lý do để bỏ research question.

Ngược lại, nó cho thấy một vấn đề sâu hơn:

\[
\boxed{
\text{pointwise prediction quality}
\neq
\text{set-selection quality}
}
\]

---

# 2. Evidence thứ hai: Gaussian interventions có tương tác

Các phase trước đã đo:

\[
R_{add}(S)
=
\frac{\Delta Q(S)}
{\sum_{i\in S}\Delta Q_i}.
\]

Đã quan sát:

\[
R_{add}(|S|=4)\approx0.2249
\]

và:

\[
R_{add}(|S|=16)\approx0.0048.
\]

Do đó:

\[
\boxed{
\Delta Q(S)\neq
\sum_{i\in S}\Delta Q_i.
}
\]

Nói cách khác, hai Gaussian có thể đều có utility cao khi xét riêng nhưng giá trị của chúng có thể giảm mạnh khi update đồng thời.

Với:

\[
\Delta Q(i\mid S)
=
Q(G\oplus S\oplus i)-Q(G\oplus S),
\]

không nhất thiết:

\[
\Delta Q(i\mid S)
=
\Delta Q(i\mid\emptyset).
\]

Đây chính là lý do cần chuyển từ pointwise ranking sang conditional set decision.

---

# 3. Research hypothesis mới

## H0 — formulation hiện tại

\[
U_i^\star
=
U_i^\star(\emptyset)
\]

đủ để lựa chọn các Gaussian.

## H1 — formulation mới

\[
\boxed{
U_i^\star
=
U_i^\star(S)
}
\]

phụ thuộc vào subset \(S\) đã được chọn.

Câu hỏi nghiên cứu:

> **Does conditioning Gaussian utility on the already-selected set improve budgeted subset allocation relative to pointwise ranking?**

Đây là hypothesis cần test trước khi xây model.

---

# 4. Formulation mới

Cho active Gaussian map:

\[
G_t=\{g_1,\ldots,g_N\}
\]

và candidate pool:

\[
P_t\subseteq G_t.
\]

Mục tiêu:

\[
\boxed{
\max_{S_t\subseteq P_t}\Delta Q(S_t)
}
\]

subject to:

\[
\boxed{
C(S_t)\le B_t.
}
\]

## 4.1 Conditional marginal quality

Với \(i\notin S\):

\[
\boxed{
\Delta Q(i\mid S)
=
Q(G_t\oplus S\oplus i)
-
Q(G_t\oplus S).
}
\]

## 4.2 Conditional cost

\[
\boxed{
\Delta C(i\mid S)
=
C(S\cup\{i\})-C(S).
}
\]

## 4.3 Conditional marginal utility

\[
\boxed{
U^\star(i\mid S)
=
\frac{\Delta Q(i\mid S)}
{\Delta C(i\mid S)+\epsilon}.
}
\]

Điểm quan trọng: \(S=\emptyset\) chính là utility formulation hiện tại.

Do đó:

\[
U^\star(i\mid\emptyset)=U_i^\star.
\]

---

# 5. Vì sao cần kiểm tra cost non-additivity

Current scheduler gần như giả định:

\[
C(S)
\approx
\sum_{i\in S}C_i.
\]

Nhưng trên GPU có thể có:

\[
\boxed{
C(S)\neq\sum_i C_i
}
\]

do:

- kernel launch overhead;
- shared optimizer state;
- memory access;
- parameter grouping;
- GPU occupancy;
- rasterization/tile interactions.

Do đó research problem đầy đủ hơn là:

\[
\boxed{
\text{joint quality–cost conditional allocation}.
}
\]

---

# 6. Phase 12-I — Interaction Audit
## Đây là việc phải làm đầu tiên

**Không build model mới trước khi audit.**

Mục tiêu:

1. đo magnitude của interaction;
2. xác định interaction có ảnh hưởng đến decision hay không;
3. tìm descriptor nào giải thích interaction;
4. quyết định GO / NO-GO.

---

## 6.1 Thiết kế sampling

Mỗi frame:

1. xây candidate pool;
2. chọn candidate \(i\);
3. sinh selected context \(S\);
4. chạy intervention;
5. restore state chính xác;
6. ghi lại conditional utility.

Các kích thước:

\[
|S|\in\{0,1,2,4\}.
\]

Không exhaustive enumeration.

---

## 6.2 Sampling theo overlap

Chia:

### Low overlap

\[
IoU(i,S)<0.1
\]

### Medium overlap

\[
0.1\le IoU(i,S)<0.5
\]

### High overlap

\[
IoU(i,S)\ge0.5.
\]

---

## 6.3 Sampling theo depth

Chia thêm:

- gần depth;
- trung bình;
- khác depth mạnh.

Mục đích là tách:

\[
\text{screen-space conflict}
\]

khỏi:

\[
\text{depth conflict}.
\]

---

# 7. Interaction metrics

## 7.1 Pair interaction

Với \(i,j\):

\[
\boxed{
I_{ij}
=
\Delta Q(\{i,j\})
-
\Delta Q_i
-
\Delta Q_j.
}
\]

Diễn giải:

- \(I_{ij}\approx0\): additive;
- \(I_{ij}<0\): redundancy / destructive interference;
- \(I_{ij}>0\): complementarity.

---

## 7.2 Pair interaction ratio

\[
\boxed{
R_{pair}
=
\frac{\Delta Q(\{i,j\})}
{\Delta Q_i+\Delta Q_j+\epsilon}.
}
\]

Plot:

\[
IoU(i,j)
\rightarrow
R_{pair}.
\]

Không được giả định trước dấu của correlation.

---

## 7.3 Conditional utility deviation

\[
\boxed{
D_Q(i,S)
=
\Delta Q(i\mid S)
-
\Delta Q(i\mid\emptyset).
}
\]

Normalized:

\[
\boxed{
R_Q(i,S)
=
\frac{
\Delta Q(i\mid S)
}{
\Delta Q(i\mid\emptyset)+\epsilon
}.
}
\]

---

# 8. Metric mạnh nhất: sign-flip rate

Đây là metric có thể quyết định toàn bộ hướng research.

Tìm:

\[
U^\star(i\mid\emptyset)>0
\]

nhưng:

\[
U^\star(i\mid S)<0.
\]

Định nghĩa:

\[
\boxed{
R_{flip}
=
P\left[
\operatorname{sign}U^\star(i\mid S)
\neq
\operatorname{sign}U^\star(i\mid\emptyset)
\right].
}
\]

Nếu \(R_{flip}\) lớn:

\[
\boxed{
\text{pointwise utility is fundamentally insufficient}.
}
\]

Nếu \(R_{flip}\) gần 0:

\[
\boxed{
\text{interaction probably does not justify a complex conditional model}.
}
\]

---

# 9. Rank instability

So sánh:

\[
rank\left(U^\star(i\mid\emptyset)\right)
\]

với:

\[
rank\left(U^\star(i\mid S)\right).
\]

Metrics:

- Spearman;
- Kendall \(\tau\);
- NDCG;
- Top-K overlap.

Điểm cần phân biệt:

\[
\text{utility magnitude changes}
\]

không nhất thiết kéo theo:

\[
\text{candidate order changes}.
\]

Phase 6 đã cho thấy rank có thể khá ổn định trong một protocol. Vì vậy Phase 12-I phải tìm **decision-relevant rank changes**, không chỉ magnitude differences.

---

# 10. Interaction audit output

Đề xuất:

```text
results/phase12_paper_evidence/interaction_audit/

├── interaction_pairs.csv
├── interaction_contexts.csv
├── interaction_groups.csv
├── interaction_summary.md
├── interaction_matrix.png
├── interaction_vs_iou.png
├── conditional_vs_isolated.png
├── sign_flip_by_overlap.png
├── rank_instability.png
└── manifest.json
```

Báo cáo bắt buộc:

\[
R_{pair}
\]

\[
D_Q
\]

\[
R_Q
\]

\[
R_{flip}
\]

\[
\rho,\tau,\text{NDCG},\text{Top-K}.
\]

---

# 11. GO / NO-GO checkpoint

## GO

Đi tiếp nếu tìm thấy ít nhất một pattern rõ ràng:

\[
R_{flip}>0
\]

ở mức đủ lớn để ảnh hưởng selection,

hoặc:

\[
R_Q
\]

thay đổi mạnh theo overlap/depth conflict,

hoặc:

\[
I_{ij}
\]

có magnitude đáng kể và predict được bằng interaction descriptors.

---

## NO-GO

Không xây interaction model nếu:

\[
U(i\mid S)
\approx
U(i\mid\emptyset)
\]

và:

\[
rank(i\mid S)
\approx
rank(i\mid\emptyset)
\]

trên phần lớn candidate.

Khi đó chuyển research direction khác.

---

# 12. Phase 12-J — Conditional Counterfactual Dataset

Nếu 12-I = GO:

mỗi record:

\[
\boxed{
(scene,t,S,i,s_i,h_S,\Delta Q,\Delta C,U)
}
\]

Trong đó:

- \(s_i\): local state của candidate;
- \(S\): selected set;
- \(h_S\): representation của selected set;
- \(\Delta Q\): conditional quality gain;
- \(\Delta C\): conditional cost;
- \(U\): conditional utility.

---

# 13. Context representation

Không quay lại full Phase 6 cross-attention.

Dùng permutation-invariant pooling:

\[
\boxed{
h_S
=
\rho\left(
\sum_{j\in S}\phi(s_j)
\right).
}
\]

Điều này có ba lợi ích:

1. không phụ thuộc thứ tự Gaussian;
2. chi phí nhỏ;
3. phù hợp bản chất set selection.

Ngoài embedding, nên đưa thêm các statistics vật lý:

\[
IoU,\ depth\ conflict,\ alpha\ competition,\ attribution redundancy.
\]

---

# 14. Interaction-aware model

Model mới:

\[
\boxed{
V_\theta(i\mid S,B_{rem})
}
\]

đầu ra:

\[
p_i=P(U^\star(i\mid S)>0)
\]

\[
\widehat{\Delta Q}(i\mid S)
\]

\[
\widehat{\Delta C}(i\mid S).
\]

Utility:

\[
\boxed{
\hat U_i(S)
=
p_i
\frac{
\widehat{\Delta Q}(i\mid S)
}{
\widehat{\Delta C}(i\mid S)+\epsilon
}.
}
\]

---

# 15. Architecture được đề xuất

```text
Candidate local state s_i
          │
          ▼
     Local Encoder
          │
          ├──────────────┐
          │              │
          ▼              ▼
    Base embedding   Interaction descriptors
                           │
                           ▼
                     Interaction Encoder
                           ▲
                           │
             Selected set S
                           │
                 permutation-invariant
                       pooling
                           │
                           ▼
                 Conditional Fusion
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
                 p_i      ΔQ_hat     C_hat
                  │
                  └───────┬─────────┘
                          ▼
                     U_hat(i|S)
                          │
                          ▼
                 Sequential Selector
```

Không cần:

- Transformer toàn map;
- GNN hàng nghìn node;
- full pairwise attention;
- model > vài chục nghìn parameters.

---

# 16. Two-stage candidate screening

Candidate pool có thể có hàng nghìn Gaussian.

Do đó:

## Stage 1

Cheap ranking:

\[
N\rightarrow K'
\]

bằng:

- Error;
- GradNorm;
- current utility.

Ví dụ:

\[
K'=32.
\]

## Stage 2

Conditional model:

\[
K'\rightarrow K.
\]

Complexity:

\[
O(N)+O(K'K)
\]

thay vì:

\[
O(N^2).
\]

Điều này giúp contribution có ý nghĩa ở AI Systems.

---

# 17. Sequential adaptive greedy scheduler

Khác scheduler hiện tại:

```text
score once
sort once
pack
```

Scheduler mới:

```text
S = {}
B_rem = B

while feasible candidates remain:

    evaluate V(i | S, B_rem)

    remove unsafe candidates

    select best feasible i

    S ← S ∪ {i}

    B_rem ← B_rem - ΔC(i | S)
```

Công thức:

\[
\boxed{
i^\star
=
\arg\max_{i\notin S}
V_\theta(i\mid S,B_{rem})
}
\]

subject to:

\[
\Delta C(i\mid S)\le B_{rem}.
\]

Lặp lại đến khi không còn candidate feasible.

---

# 18. Risk-aware extension

Sau khi conditional model hoạt động, mới thêm uncertainty.

Model dự đoán:

\[
\mu_i(S)
\]

và:

\[
\sigma_i(S).
\]

Định nghĩa:

\[
\boxed{
LCB_i(S)
=
\mu_i(S)-\kappa\sigma_i(S).
}
\]

Scheduler dùng:

\[
\boxed{
V_i^{safe}
=
\frac{LCB_i(S)}
{\widehat{\Delta C}(i\mid S)+\epsilon}.
}
\]

Nếu:

\[
LCB_i(S)<0,
\]

thì abstain.

Điều này đặc biệt phù hợp vì Phase 1–3 đã phát hiện negative utility không hiếm.

---

# 19. Joint cost model

Nếu audit cho thấy cost interactions đáng kể:

\[
\boxed{
\Delta C(i\mid S)
}
\]

phải được model riêng.

Ví dụ:

\[
C(S\cup i)-C(S).
\]

Không tiếp tục giả định:

\[
C(S)=\sum_iC_i.
\]

Khi đó scheduler tối ưu:

\[
\max_S \Delta Q(S)
\]

subject to:

\[
C(S)\le B.
\]

---

# 20. Các baseline bắt buộc cho experiment cuối

Cùng candidate pool, cùng state, cùng budget:

### B0

Error-only

### B1

GradNorm

### B2

Current TwoHeadMLP

### B3

Positive-head MLP

### B4

Conditional interaction-aware model

### B5

Conditional + uncertainty

### Oracle

Counterfactual oracle selection.

---

# 21. Metrics cuối

Không chỉ dùng PSNR.

## Prediction

\[
\rho,\tau
\]

\[
NDCG@K
\]

\[
AUROC(U^\star>0)
\]

\[
\text{Calibration}
\]

## Selection

\[
OSE
\]

\[
\text{Oracle Recovery}
\]

\[
\text{negative selected rate}
\]

## Compute

\[
\boxed{
E_Q=
\frac{\Delta Q}{T_{opt}}
}
\]

## Reconstruction

- PSNR;
- SSIM;
- depth L1;
- tracking metric nếu pipeline dùng SLAM.

## System

- scheduler latency;
- optimization latency;
- end-to-end latency;
- Gaussian count;
- VRAM.

---

# 22. Experiment matrix

## Stage I — Scientific diagnosis

\[
12\text{-I Interaction Audit}
\]

Chỉ audit, chưa học.

## Stage II — Learning

\[
12\text{-J Conditional Dataset}
\]

\[
12\text{-K Conditional Model}
\]

## Stage III — Selection

\[
12\text{-L Interaction-Aware Scheduler}
\]

## Stage IV — Risk

\[
12\text{-M Uncertainty / Abstention}
\]

## Stage V — Final benchmark

- 4 unseen TUM sequences;
- 5 seeds;
- budgets \(5,10,15,20,30\) ms;
- population shift;
- all baselines;
- oracle.

---

# 23. Expected scientific outcomes

Không được giả định trước result.

Có ba khả năng.

## Outcome A — Strong positive

Tìm thấy:

\[
U(i\mid S)
\]

khác đáng kể với:

\[
U(i\mid\emptyset),
\]

và conditional model cải thiện:

\[
NDCG,\ OSE,\ \Delta Q/T.
\]

Khi đó contribution mạnh:

\[
\boxed{
\text{Conditional Counterfactual Utility for Sequential Gaussian Allocation}
}
\]

---

## Outcome B — Interaction exists but is difficult to predict

Ví dụ:

\[
R_{flip}>0
\]

nhưng model prediction yếu.

Khi đó contribution vẫn có thể là:

\[
\boxed{
\text{isolated utility is insufficient; interactions are the prediction bottleneck}.
}
\]

Đây là một scientific finding mạnh, và có thể làm paper theo hướng limitation + new empirical analysis.

---

## Outcome C — Interaction negligible

Khi:

\[
U(i\mid S)\approx U(i\mid\emptyset)
\]

thì interaction không phải root cause.

Không nên build model phức tạp.

Chuyển sang một hypothesis khác.

---

# 24. Các hướng KHÔNG nên làm lúc này

Không nên:

- tăng MLP từ 64 lên 256 chỉ để tối ưu metric;
- thêm Transformer;
- thêm GNN toàn map;
- thêm RL;
- thêm diffusion;
- thêm hàng chục handcrafted features;
- tune theo test sequence;
- thay đổi frozen Phase 10 benchmark chỉ để có số đẹp;
- tiếp tục thêm benchmark nếu chưa hiểu failure mechanism.

Mục tiêu là giải thích:

\[
\boxed{
\text{why current pointwise utility does not produce the desired subset selection}.
}
\]

---

# 25. Sửa terminology trong paper

Tránh:

> “external baselines”

nếu đó chỉ là heuristic tự cài đặt.

Tránh:

> “real-time”

nếu \(T_{frame}\) vẫn ở mức giây.

Nên dùng:

> compute-constrained

> optimization-budgeted

> online

> latency-aware

> closed-loop

> zero-shot

Tránh khẳng định:

\[
\text{utility model is optimal}
\]

khi GradNorm có pointwise correlation cao hơn.

---

# 26. Cấu trúc paper mạnh hơn sau khi nâng cấp

## 1. Introduction

Problem:

\[
\text{too many Gaussian updates}
\]

Existing proxy:

\[
\text{residual / gradient}
\]

Observation:

\[
\text{isolated utility}
\]

Failure:

\[
\text{interaction}.
\]

---

## 2. Related Work

- online Gaussian SLAM;
- adaptive Gaussian optimization;
- pruning/sensitivity;
- budget-aware allocation;
- set selection / submodular optimization;
- conditional decision modeling.

---

## 3. Problem

\[
\max_S\Delta Q(S)
\]

subject to:

\[
C(S)\le B.
\]

---

## 4. Empirical Diagnosis

\[
\text{Error}\neq U^\star
\]

and:

\[
\Delta Q(S)\neq\sum_i\Delta Q_i.
\]

---

## 5. Method

\[
U(i\mid S)
\]

conditional utility model.

---

## 6. Scheduler

Sequential conditional greedy allocation.

---

## 7. Experiments

- utility prediction;
- interaction;
- selection;
- zero-shot;
- budget;
- systems.

---

## 8. Limitations

- physical end-to-end latency;
- remaining prediction uncertainty;
- renderer bottleneck.

---

# 27. The key conceptual shift

Current project:

\[
\boxed{
\text{Gaussian} \rightarrow \text{score}
}
\]

New project:

\[
\boxed{
\text{Gaussian + selected set + remaining budget}
\rightarrow
\text{conditional marginal decision}
}
\]

This changes the problem from:

\[
\text{pointwise ranking}
\]

to:

\[
\boxed{
\text{sequential budgeted set optimization}.
}
\]

Đây là thay đổi về **problem formulation**, không chỉ architecture.

---

# 28. Final roadmap

```text
Phase 11
   │
   ▼
Phase 12: Evidence + M0-M4
   │
   ▼
┌───────────────────────────────┐
│ 12-I Interaction Audit        │
│ no model yet                  │
└───────────────┬───────────────┘
                │
        GO      │      NO-GO
        ▼       │       ▼
12-J Conditional│   alternate
Dataset         │   hypothesis
        │
        ▼
12-K Conditional Model
        │
        ▼
12-L Interaction-Aware Scheduler
        │
        ▼
12-M Uncertainty / Risk
        │
        ▼
12-N Final Multi-Sequence Benchmark
        │
        ▼
Final Results Freeze
        │
        ▼
Paper Rewrite
        │
        ▼
Submission Package
```

---

# 29. Việc cần làm ngay bây giờ

Không tạo model mới.

Không rerun toàn bộ benchmark.

Chỉ thực hiện:

\[
\boxed{\textbf{12-I Interaction Audit}}
\]

và trả lời đúng 5 câu hỏi:

1. Bao nhiêu phần trăm intervention thay đổi utility khi \(S\) thay đổi?
2. Bao nhiêu candidate đổi dấu utility?
3. Interaction có phụ thuộc mạnh vào screen-space overlap không?
4. Có phụ thuộc vào depth conflict / attribution không?
5. Interaction có đủ mạnh để làm thay đổi top-K selection không?

Nếu câu trả lời là **có**, khi đó chúng ta mới xây:

\[
\boxed{
\text{Conditional Counterfactual Utility Model}
}
\]

và đó sẽ là thay đổi mang tính nền tảng nhất của Adaptive 3DGS hiện tại.

