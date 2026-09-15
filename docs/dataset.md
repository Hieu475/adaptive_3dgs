# Dataset Provenance & Split Invariants

This document formalizes the dataset provenance, camera models, and split partitions utilized across the Adaptive 3D Gaussian Splatting project.

---

## 1. Overview

Online 3D reconstruction benchmarks evaluate photometric and geometric accuracy on real streaming RGB-D sequences from the **TUM RGB-D Benchmark** (Sturm et al., IROS 2012).

$$\boxed{ \text{Training Sequence} \quad\neq\quad \text{Validation Sequence} \quad\neq\quad \text{Zero-Shot Test Sequence} }$$

| Partition | Sequence Name | Camera | Primary Purpose | Frame Count |
| :--- | :--- | :---: | :--- | :---: |
| **Training** | `rgbd_dataset_freiburg1_desk` (`tum_fr1_desk`) | FR1 | Oracle utility generation & TwoHeadMLP training | 150 frames |
| **Validation** | `rgbd_dataset_freiburg1_desk` (`tum_fr1_desk`) | FR1 | Hyperparameter validation (A1 representation & B2 $\beta$) | 50 held-out frames |
| **Zero-Shot Test** | `rgbd_dataset_freiburg2_xyz` (`tum_fr2_xyz`) | FR2 | Closed-loop end-to-end confirmatory benchmark | 30 continuous frames |

> [!IMPORTANT]
> **Zero-Shot Separation Guarantee**:
> The primary test scene `tum_fr2_xyz` is completely unseen during utility model training. It features a different room, distinct surface textures, different camera intrinsics (FR2 sensor), and distinct motion dynamics. No test-set tuning, checkpoint cherry-picking, or feature adaptation was performed on `tum_fr2_xyz`.

---

## 2. Sensor & Camera Parameters

Images are evaluated at $320 \times 240$ resolution (downsampled $2\times$ from native $640 \times 480$ for low-latency streaming reconstruction).

### Freiburg 1 (FR1) Sensor
- Native Focal Length: $f_x = 517.3, f_y = 516.5$
- Native Principal Point: $c_x = 318.6, c_y = 255.3$
- Scaled ($320 \times 240$):
  $$K_{\text{FR1}} = \begin{bmatrix} 258.65 & 0 & 159.30 \\ 0 & 258.25 & 127.65 \\ 0 & 0 & 1 \end{bmatrix}$$

### Freiburg 2 (FR2) Sensor
- Native Focal Length: $f_x = 520.9, f_y = 521.0$
- Native Principal Point: $c_x = 325.1, c_y = 249.7$
- Scaled ($320 \times 240$):
  $$K_{\text{FR2}} = \begin{bmatrix} 260.45 & 0 & 162.55 \\ 0 & 260.50 & 124.85 \\ 0 & 0 & 1 \end{bmatrix}$$

### Depth Calibration
- Depth scale factor: $5000.0$ (16-bit PNG format: $1\text{ meter} = 5000\text{ units}$).
- Valid depth range: $z \in [0.1\text{ m}, 5.0\text{ m}]$. Invalid or missing depth values ($z \le 0$ or $z > 5.0$) are masked out during geometric error computation.

---

## 3. Ground Truth Trajectory Alignment

Camera poses are recorded via high-precision external motion capture (Vicon optical tracker at 100 Hz).
- Pose format: $T_{WC} \in \mathrm{SE}(3)$ represented as a $4 \times 4$ rigid body transformation matrix.
- Timestamp synchronization: RGB frames, depth frames, and ground-truth poses are causally synchronized using nearest-neighbor timestamp matching within a $\pm 10\text{ ms}$ temporal window.

---

## 4. Sequence Acquisition & Local Storage

Sequences are organized within the repository workspace:
```
data/
├── tum_fr1_desk/
│   ├── rgb/
│   ├── depth/
│   └── groundtruth.txt
└── tum_fr2_xyz/
    ├── rgb/
    ├── depth/
    └── groundtruth.txt
```

Automated download and preparation script:
```bash
bash scripts/download_tum.sh
```

---

## 5. Multi-Seed Protocol & Trajectory Configuration

To ensure statistical rigor and eliminate stochastic cherry-picking, evaluations across all policies are repeated over 5 predefined seeds:

- **Seeds**: `42`, `43`, `44`, `45`, `46`
- **Resolution**: $320 \times 240$ (downsampled by $2\times$)
- **Frame Range (Phase 10 E2E)**: Frames `[0, 29]` (30 continuous streaming steps on `tum_fr2_xyz`)
- **Full Sequence Availability**: `tum_fr2_xyz` contains 464 total frames available for extended long-horizon tests
- **Budget**: $B = 15.0\text{ ms}$ per frame with safety factor $\alpha = 1.1$

