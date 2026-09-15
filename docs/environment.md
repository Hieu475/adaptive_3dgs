# Environment Specification & Reproducibility Guide

This document specifies the exact software and hardware environment used to develop, train, and validate the **Adaptive 3D Gaussian Splatting (Adaptive 3DGS)** research pipeline.

$$\boxed{\text{Pinned Environment + Fixed Protocol + Fixed Seeds} \implies \text{Reproducible Evaluation}}$$

---

## 1. Hardware & System Configuration

All authoritative Phase 10 end-to-end benchmarks and empirical validations were executed on the following system configuration:

| Component | Specification | Details / Notes |
| :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu 24.04.1 LTS) | Kernel `6.8.0-94-generic` x86_64 |
| **CPU Architecture** | x86_64 | Multi-core Intel/AMD host processor |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU | 6 GB GDDR6 VRAM |
| **Compute Capability** | SM 8.9 | Ada Lovelace Architecture |
| **C/C++ Compiler** | GCC / G++ 13.3.0 | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| **Build Tools** | CMake 4.0.1, Ninja 1.11.1 | Standard Unix make / ninja build support |
| **CUDA Driver & Toolkit** | CUDA 12.8 / Driver 570+ | `nvcc` compiler driver 12.8 |

---

## 2. Core Python & Deep Learning Stack

The core Python environment relies on PyTorch with CUDA 12.8 acceleration:

| Package | Pinned / Validated Version | Role in Pipeline |
| :--- | :---: | :--- |
| **Python** | `3.12.13` (conda-forge) | Runtime execution environment |
| **PyTorch (`torch`)** | `2.11.0+cu128` | Tensor computation, backprop & CUDA operations |
| **TorchVision (`torchvision`)** | `0.16.0+` | Image preprocessing and tensor transformations |
| **NumPy (`numpy`)** | `2.2.6` | Numerical linear algebra, array serialization |
| **SciPy (`scipy`)** | `1.13.1` | Statistical hypothesis testing (Wilcoxon, bootstrap) |
| **Pandas (`pandas`)** | `3.0.5` | Metrics aggregation, CSV persistence, audit logs |
| **Matplotlib (`matplotlib`)** | `3.11.1` | Scientific publication figure generation |
| **PyYAML (`pyyaml`)** | `6.0+` | Configuration parsing |
| **tqdm** | `4.65.0+` | Progress tracking across long benchmark trajectories |
| **pytest** | `7.0.0+` | Continuous integration and regression testing |

---

## 3. Installation & Setup Instructions

### Option A: Conda Environment Setup (Recommended)

Using the provided [`environment.yml`](../environment.yml):

```bash
# 1. Clone repository
git clone https://github.com/Hieu475/adaptive_3dgs.git
cd adaptive_3dgs

# 2. Create conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate adaptive_3dgs

# 4. Verify PyTorch CUDA acceleration
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Option B: Pip Virtual Environment Setup

Using the provided [`requirements.txt`](../requirements.txt):

```bash
# 1. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# 3. Install pinned PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install project requirements
pip install -r requirements.txt
```

---

## 4. Verification & Diagnostics

Run the integrated verification suite to validate environment compliance:

```bash
# Verify environment and libraries
python -c "
import torch, numpy, scipy, pandas, matplotlib
print('✓ PyTorch:   ', torch.__version__)
print('✓ CUDA:      ', torch.cuda.is_available())
print('✓ NumPy:     ', numpy.__version__)
print('✓ SciPy:     ', scipy.__version__)
print('✓ Pandas:    ', pandas.__version__)
print('✓ Matplotlib:', matplotlib.__version__)
"

# Run the runtime unit test suite
pytest tests/test_phase10_runtime.py -v

# Run the 30-second closed-loop smoke test
python experiments/run_phase10_smoke.py
```

---

## 5. Reproducibility vs. Numerical Determinism

All experiments fix random seeds across PyTorch, NumPy, and Python standard random libraries:
- Default multi-seed confirmatory evaluation suite: `seeds = [42, 43, 44, 45, 46]`
- Model checkpoints and evaluation protocols set `torch.use_deterministic_algorithms(False)` for standard CuDNN speed while ensuring stateful experimental reproducibility via persistent seeds and frozen model weights.

> [!NOTE]
> **Reproducibility $\neq$ Absolute Bit-for-Bit Determinism**:
> While static deliverable artifacts (CSVs, figures, summary reports, and frozen model weights) are verified bit-for-bit via cryptographic SHA-256 digests against the manifest, active CUDA/cuDNN neural execution inherently exhibits minor floating-point non-associativity across parallel warp reduction orders and GPU architectures. We guarantee rigorous experimental reproducibility (consistent rankings, effect sizes, statistical significance, and invariant adherence) through pinned dependencies, fixed data splits, and explicit evaluation protocols.
