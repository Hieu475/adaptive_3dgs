"""Noise-adaptive selection + throttling (ICCV novelty candidate).

Diagnosis (Phase-14): fixed kappa=0.90 and the temporal x influence heuristic
are overfit to noisy real-sensor depth — they win on TUM but lose -5.5 dB on
clean synthetic Replica. Kappa ablation recovers only ~0.8 dB; the remaining
~4.7 dB is selection.

Method (training-free, <1 ms/frame, CPU NumPy):
- Per-frame depth-noise score n in [0,1] from already-available signals:
  invalid/hole ratio, edge density (Sobel on filled depth), and near-range
  saturation. Clean synthetic -> n ~ 0; noisy Kinect -> n ~ 1.
- Controller routes per frame:
    noise -> 1 : policy 'error_influence_temporal' (proven on TUM) + kappa 0.90
    noise -> 0 : policy 'error_only' (proven on Replica) + kappa 0.98 (~off)
  Linear blend in between; hysteresis optional (disabled by default).

Interface is pipeline-level (no scheduler signature change): the harness calls
``route(noise)`` then dispatches to the existing policies / kappa overrides.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


def frame_noise_score_v2(raw_invalid: float, raw_edge: float) -> Dict[str, float]:
    """Sensor-noise score v2 from RAW (pre-hygiene) depth stats. SUPERSEDED by v3.

    Failure: invalid conflates far-but-valid geometry (>4 m, e.g. 18% of
    replica_room0) with sensor holes. Kept for provenance.
    """
    score = float(np.clip(2.2 * raw_invalid + 1.5 * raw_edge, 0.0, 1.0))
    return {"noise": score, "raw_invalid": float(raw_invalid), "raw_edge": float(raw_edge)}


def frame_noise_score_v3(raw_holes: float, raw_edge: float) -> Dict[str, float]:
    """Sensor-domain cue v3 from RAW missing-measurement rate.

    Raw missing-depth rate is strongly associated with the evaluated
    real-vs-synthetic datasets (TUM Kinect holes ~0.18-0.31 vs Replica
    ~0.000, 300-500x gap); out-of-range far geometry is content and
    excluded. NOT claimed independent of scene content in general: hole
    rate also varies with reflective/transparent surfaces, incidence
    angle, range, occlusion, and preprocessing. Validated scope: 5/5
    development scenes + 7/7 held-out scenes (thresholds frozen;
    see results/phase14_corrected/router_heldout.json).
    Edge density is secondary (TUM ~0.06 vs Replica ~0.01-0.02).
    score = clip(2.5 * holes + 1.0 * edge, 0, 1):
      Replica -> ~0.02, TUM -> ~0.55-0.85.
    """
    score = float(np.clip(2.5 * raw_holes + 1.0 * raw_edge, 0.0, 1.0))
    return {"noise": score, "raw_holes": float(raw_holes), "raw_edge": float(raw_edge)}


@dataclass
class NoiseAdaptiveConfig:
    kappa_noisy: float = 0.90
    kappa_clean: float = 0.98
    # Calibrated on v3 RAW-hole scores: Replica ~0.02, TUM ~0.55+.
    blend_lo: float = 0.10   # below: pure error_only
    blend_hi: float = 0.30   # above: pure temporal x influence
    invalid_far: float = 4.0
    invalid_near: float = 0.4


def frame_noise_score(depth: np.ndarray, cfg: NoiseAdaptiveConfig | None = None) -> Dict[str, float]:
    """Depth-noise score in [0,1] (1 = noisy real-sensor-like). v0, depth-only.

    Kept for backward compatibility. Known failure: invalid_ratio is
    scene-content dependent (replica_room0 scores like TUM). Prefer v1 below.
    """
    cfg = cfg or NoiseAdaptiveConfig()
    d = np.asarray(depth, dtype=np.float32)
    valid = (d >= cfg.invalid_near) & (d <= cfg.invalid_far)
    invalid_ratio = float(1.0 - valid.mean()) if d.size else 1.0
    try:
        import cv2
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        edge_density = float((grad > 0.20).mean()) if grad.size else 0.0
    except Exception:
        edge_density = 0.0
    score = float(np.clip(0.55 * invalid_ratio + 0.45 * edge_density, 0.0, 1.0))
    return {"noise": score, "invalid_ratio": invalid_ratio, "edge_density": edge_density}


def _patch_median_laplacian_noise(gray: np.ndarray, patch: int = 16) -> float:
    """Immerkaer Laplacian noise estimate, median over patches (texture-robust).

    sigma ≈ mean(|Laplacian * N|) / 0.6745 with the [1 -2 1; -2 4 -2; 1 -2 1]
    kernel (noise variance var = sigma^2/36 for this kernel). Median across
    non-overlapping patches rejects textured patches; clean renders -> ~0.
    Returns sigma in grayscale intensity units [0, 1].
    """
    import cv2
    g = np.ascontiguousarray(np.asarray(gray, dtype=np.float32))
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=1)  # ksize=1 == 3x3 kernel above
    h, w = lap.shape
    vals = []
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            p = np.abs(lap[y:y + patch, x:x + patch]).ravel()
            mad = float(np.median(p))
            vals.append(mad / 0.6745 / 6.0)  # /6: kernel L1 gain for white noise
    return float(np.median(vals)) if vals else 0.0


def frame_noise_score_v1(rgb: np.ndarray, depth: np.ndarray,
                         cfg: NoiseAdaptiveConfig | None = None) -> Dict[str, float]:
    """Sensor-noise score v1 in [0,1] (1 = noisy real-sensor-like).

    Content-robust by design (v0's invalid_ratio failed: replica_room0 looked
    like TUM because far/empty content inflates invalid pixels):
    - rgb_noise: patch-median Laplacian sigma. Real Kinect frames carry sensor
      + compression noise (sigma >> 0); clean synthetic renders -> ~0.
    - step_ratio: among depth-edge pixels (grad > 0.05), fraction with grad >
      0.50 (ideal 1-pixel steps). Synthetic depth is stepwise (-> 1); real
      depth ramps over several pixels + noise (-> low).
    - score = clip(w_n * norm(rgb_noise) + w_s * (1 - step_ratio)).
    """
    cfg = cfg or NoiseAdaptiveConfig()
    img = np.asarray(rgb)
    if img.ndim == 3:
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    else:
        gray = np.asarray(img, dtype=np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    rgb_noise = _patch_median_laplacian_noise(np.clip(gray, 0.0, 1.0))
    d = np.asarray(depth, dtype=np.float32)
    try:
        import cv2
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        edge = grad > 0.05
        step_ratio = float((grad[edge] > 0.50).mean()) if edge.any() else 0.0
    except Exception:
        step_ratio = 0.0
    # Calibration (fit on 5 labeled scenes, see calibration note in tests):
    # rgb_noise: replica ~0.001-0.003, TUM ~0.008-0.020 -> norm at 0.010.
    n_norm = float(np.clip(rgb_noise / 0.010, 0.0, 1.0))
    score = float(np.clip(0.60 * n_norm + 0.40 * (1.0 - step_ratio), 0.0, 1.0))
    return {"noise": score, "rgb_noise": rgb_noise, "step_ratio": step_ratio}


class NoiseAdaptiveController:
    """Routes (policy, kappa) from a noise score. Stateless + hysteresis-free."""

    def __init__(self, cfg: NoiseAdaptiveConfig | None = None):
        self.cfg = cfg or NoiseAdaptiveConfig()

    def route(self, noise: float) -> Tuple[str, float]:
        """Return (selection_policy, kappa)."""
        n = float(np.clip(noise, 0.0, 1.0))
        if n >= self.cfg.blend_hi:
            return "error_influence_temporal", self.cfg.kappa_noisy
        if n <= self.cfg.blend_lo:
            return "error_only", self.cfg.kappa_clean
        t = (n - self.cfg.blend_lo) / max(self.cfg.blend_hi - self.cfg.blend_lo, 1e-6)
        kappa = self.cfg.kappa_clean + t * (self.cfg.kappa_noisy - self.cfg.kappa_clean)
        policy = "error_influence_temporal" if t >= 0.5 else "error_only"
        return policy, float(kappa)
