# Checkpoint Registry & Model Architecture

This document records the cryptographic registry, architectural specifications, and loading protocols for the frozen utility model checkpoints utilized in Phase 10.

---

## 1. Cryptographic Checkpoint Registry

All checkpoints were trained in Phase 9B / 9C and frozen unconditionally for Phase 10 end-to-end evaluation. No retraining, fine-tuning, or parameter updates are permitted.

| Seed | Checkpoint File | Checkpoint SHA-256 Hash | Normalizer File | Normalizer SHA-256 Hash |
| :---: | :--- | :--- | :--- | :--- |
| **42** | `A1_online_adaptive_seed_42.pt` | `3e9ce12dac70ccfe37d687ba3cb67957b8cba29bb9fa3874ff7ff5bd6977003a` | `A1_online_adaptive_seed_42.json` | `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458` |
| **43** | `A1_online_adaptive_seed_43.pt` | `759169c5cf68149891b432f1a56f2fec1e1cc963bb43ccbcbd1899b0316ab219` | `A1_online_adaptive_seed_43.json` | `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458` |
| **44** | `A1_online_adaptive_seed_44.pt` | `c564f6b1f83810e460e7ceb3c9c7d67828395446634c4eda8df931a60826b6cc` | `A1_online_adaptive_seed_44.json` | `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458` |
| **45** | `A1_online_adaptive_seed_45.pt` | `94155ec2f8d0e895c58ba985c39d43122c40da80ad94be3dac02c24dc3dc47d9` | `A1_online_adaptive_seed_45.json` | `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458` |
| **46** | `A1_online_adaptive_seed_46.pt` | `5d87caa388c6827d860f1afdd50046e65bdc07ca4a96b1d63e98ee5ef7108d7d` | `A1_online_adaptive_seed_46.json` | `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458` |

Deliverables copy location: `results/phase10_e2e/checkpoints/`  
Source location: `results/phase9b_robust_normalization/checkpoints/`

---

## 2. Model Architecture: TwoHeadMLP

The utility predictor employs a decoupled two-head architecture designed to estimate both the marginal quality delta $\widehat{\Delta Q}_i$ and optimization cost $\widehat{C}_i$:

$$\hat{U}_i = \frac{\widehat{\Delta Q}_i}{\widehat{C}_i}$$

```
Input: Observable Features Z_norm ∈ R^11
               │
               ▼
        Shared Trunk (Linear 11 → 64 + LeakyReLU)
               ├─────────────────────────┐
               ▼                         ▼
         Quality Head                Cost Head
    (Linear 64 → 32 + LeakyReLU) (Linear 64 → 32 + LeakyReLU)
               │                         │
               ▼                         ▼
         Linear 32 → 1             Linear 32 → 1 + Softplus
               │                         │
               ▼                         ▼
         Predicted ΔQ_hat          Predicted C_hat (> 0)
```

### Architectural Parameters:
- Input Dimension: $D = 11$ (canonical observable features)
- Shared Trunk: `Linear(11, 64)`, `LeakyReLU(negative_slope=0.2)`
- Quality Head: `Linear(64, 32)`, `LeakyReLU(0.2)`, `Linear(32, 1)`
- Cost Head: `Linear(64, 32)`, `LeakyReLU(0.2)`, `Linear(32, 1)`, `Softplus(beta=1.0)`
- Total Parameters: 4,386 parameters ($\approx 17.5\text{ KB}$)

---

## 3. Pre-Processing Pipeline: A1 + B2

Features undergo a two-step stabilization pipeline before feeding into the TwoHeadMLP:

1. **A1 Representation (`geometry_relative`)**:
   - Removes absolute camera distance and coordinate scale variance.
   - Transforms spatial coordinates and bounding box metrics relative to Gaussian covariance scale $\sigma_i$ and screen projection.

2. **B2 Online Normalization (`OnlineEMANormalizer`, $\beta = 0.90$)**:
   - Adapts feature moments online across streaming video frames:
     $$\mu_t = \beta \mu_{t-1} + (1 - \beta) \bar{x}_t$$
     $$\sigma_t^2 = \beta \sigma_{t-1}^2 + (1 - \beta) s_t^2$$
   - Prevents domain-shift breakdown when transitioning zero-shot between indoor rooms.

---

## 4. Model Loading & Immutability Verification

```python
from research.phase10_runtime import Phase10ModelBundle

# Load frozen bundle
bundle = Phase10ModelBundle(seed=42, device="cuda")

# 1. Verify evaluation mode and parameter freeze
assert not bundle.model.training
assert all(not p.requires_grad for p in bundle.model.parameters())

# 2. Snapshot baseline weights and hash
w0 = bundle.snapshot_weights()
h0 = bundle.compute_weights_hash()

# ... execute trajectory reconstruction ...

# 3. Mathematically assert zero parameter mutation
is_immutable, max_diff = bundle.verify_immutability(w0)
assert is_immutable and max_diff == 0.0
assert bundle.compute_weights_hash() == h0
```
