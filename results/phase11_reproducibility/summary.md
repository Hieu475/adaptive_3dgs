# Phase 11: Scientific Reproducibility Audit & Integrity Report

## Executive Summary

Phase 11 formalizes the reproducibility infrastructure, artifact provenance, and cryptographic verification for the **Adaptive 3D Gaussian Splatting (Adaptive 3DGS)** research project. All empirical results established in Phase 10 have been audited across code, environment, dataset, checkpoints, and deliverable artifacts.

$$\boxed{\text{Pinned Environment + Fixed Protocol + Fixed Seeds} \implies \text{Reproducible Evaluation}}$$

---

## 1. Formal Gate Audit Matrix (Gates 11A – 11F)

| Gate | Focus | Evaluation Criteria | Result | Evidence |
| :--- | :--- | :--- | :---: | :--- |
| **Gate 11A** | Code Integrity | Clean git state, tag `phase10-frozen` verified, active branch `phase11-reproducibility` | :white_check_mark: **PASS** | Commit `f1222ef4d3`, working tree clean |
| **Gate 11B** | Environment | Python, PyTorch, CUDA, GCC, CMake, dependencies documented with determinism disclosures | :white_check_mark: **PASS** | [`environment.json`](environment.json), [`docs/environment.md`](../../docs/environment.md) |
| **Gate 11C** | Dataset Provenance | TUM FR1/FR2 splits, camera models, depth scaling, SE(3) causal pose sync, 5 seeds | :white_check_mark: **PASS** | [`dataset.json`](dataset.json), [`docs/dataset.md`](../../docs/dataset.md) |
| **Gate 11D** | Reproduction | 3-tier testing (456 unit tests, 8 runtime tests, fast smoke test, master reproduction script) | :white_check_mark: **PASS** | [`smoke_test.json`](smoke_test.json), [`reproduction_log.txt`](reproduction_log.txt) |
| **Gate 11E** | Artifact Integrity | Cryptographic SHA-256 matching for 14 Phase 10 artifacts and 5 frozen model checkpoints | :white_check_mark: **PASS** | [`manifest.json`](manifest.json), 14/14 bit-for-bit match |
| **Gate 11F** | Documentation | Standardized 11-section README, primary results table, paired statistics, latency disclosures | :white_check_mark: **PASS** | [`README.md`](../../README.md), [`docs/checkpoints.md`](../../docs/checkpoints.md) |

---

## 2. 3-Tier Regression Results

1. **Tier 1 (Unit Tests)**: `pytest tests/ -q` $\implies$ **456 passed in 8.8s** (100% pass rate).
2. **Tier 2 (Phase 10 Tests)**: `pytest tests/test_phase10_runtime.py -v` $\implies$ **8 passed in 1.7s**.
3. **Tier 3 (Runtime Smoke Test)**: `python experiments/run_phase10_smoke.py` $\implies$ **Passed in 22.99s**:
   - Initial primitives: $N = 3502$ $\to$ Final primitives: $N = 3786$
   - Model parameter freeze verified: `requires_grad=False`
   - Strict weight immutability: $\|\theta_T - \theta_0\|_\infty = 0.0$ (SHA-256: `ab30387c2af8c4a2...`)
   - Zero-oracle verified: `True`
   - Zero-future leakage verified: `True`

---

## 3. Cryptographic Checkpoint Registry

$$\boxed{\text{Paper Result} \longrightarrow \text{Model Checkpoint} \longrightarrow \text{Cryptographic SHA-256}}$$

| Seed | Architecture | Checkpoint File | Checkpoint SHA-256 Digest |
| :---: | :---: | :--- | :--- |
| **42** | TwoHeadMLP | `A1_online_adaptive_seed_42.pt` | `3e9ce12dac70ccfe37d687ba3cb67957b8cba29bb9fa3874ff7ff5bd6977003a` |
| **43** | TwoHeadMLP | `A1_online_adaptive_seed_43.pt` | `759169c5cf68149891b432f1a56f2fec1e1cc963bb43ccbcbd1899b0316ab219` |
| **44** | TwoHeadMLP | `A1_online_adaptive_seed_44.pt` | `c564f6b1f83810e460e7ceb3c9c7d67828395446634c4eda8df931a60826b6cc` |
| **45** | TwoHeadMLP | `A1_online_adaptive_seed_45.pt` | `94155ec2f8d0e895c58ba985c39d43122c40da80ad94be3dac02c24dc3dc47d9` |
| **46** | TwoHeadMLP | `A1_online_adaptive_seed_46.pt` | `5d87caa388c6827d860f1afdd50046e65bdc07ca4a96b1d63e98ee5ef7108d7d` |

- Preprocessing representation: A1 (`geometry_relative`)
- Streaming normalizer: B2 (`OnlineEMANormalizer`, $\beta = 0.90$, SHA-256: `7893010e29b5a2694160dbdf26cca08148cbafc0583fa2f322526120761a8458`)

---

## 4. Authoritative Phase 10 Confirmatory Benchmark

Primary comparison on zero-shot unseen sequence `tum_fr2_xyz` ($N = 145$ paired frames, budget $B = 15.0\text{ ms}$):

| Policy | Mean PSNR | Final PSNR | Mean Opt | Mean Frame |
| :--- | :---: | :---: | :---: | :---: |
| **NO_OP** | 12.34 dB | 12.15 dB | 0 ms | 6505.6 ms |
| **ERROR_ONLY** | 12.34 dB | 12.14 dB | 10.36 ms | 6805.5 ms |
| **OURS** | **12.34 dB** | **12.15 dB** | **9.25 ms** | **7005.3 ms** |
| **FULL** | 12.36 dB | 12.18 dB | 467.6 ms | 7057.8 ms |

$$\Delta Q_{\text{OURS-ERROR}} = +0.0047\text{ dB}, \quad 95\%\text{ CI} = [+0.0024, +0.0070]\text{ dB}, \quad p = 2.3245 \times 10^{-4}, \quad d = 0.337$$

---

## 5. Methodological & Systems Disclosures

1. **Reproducibility vs. Determinism**:
   Evaluation protocols enforce reproducible evaluation via persistent random seeds and pinned software environments. `torch.use_deterministic_algorithms(False)` is deliberately maintained to avoid massive CUDA throughput degradation and unsupported kernel exceptions.
2. **Physical Latency Gap**:
   $$T_{\text{scheduler}} \approx 4.32\text{ ms} \quad \ll \quad T_{\text{opt}} \approx 9.25\text{ ms} \quad \ll \quad T_{\text{frame}} \approx 7005\text{ ms}$$
   *The scheduler strictly satisfies the modeled 15 ms budget, but the current Python/PyTorch implementation is not a real-time 15 ms end-to-end system.* Full real-time deployment requires fusing attribution tracing directly into the CUDA rasterizer (Phase 13).
