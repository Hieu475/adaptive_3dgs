"""Phase 9B: Robust Normalization Architectures.

Implements three controlled normalization strategies operating on fixed
A1 scale-invariant geometric feature representations:

    1. StandardNormalizer (B0):
       Standard z-score standardization: (x - mu_train) / (sigma_train + eps).
       Baseline reference. Fit strictly on train split.

    2. RobustMADNormalizer (B1):
       Outlier-resistant median / MAD normalization:
           tilde_z = (z - median_train) / (1.4826 * MAD_train + eps)
       Fit strictly on train split. Test data never enters fitting.
       Includes guaranteed non-zero scale fallback for sparse features.

    3. OnlineEMANormalizer (B2):
       Test-time unsupervised covariate adaptation.
       Initialized from train reference statistics: mu_0 = mu_train, sigma_0 = sigma_train.
       Updated online frame-by-frame via exponential moving average:
           mu_t    = beta * mu_{t-1}    + (1 - beta) * mu_t^{frame}
           sigma_t = beta * sigma_{t-1} + (1 - beta) * sigma_t^{frame}
       Features normalized with updated online statistics:
           tilde_z = (z - mu_t) / (sigma_t + eps)
       Model weights remain strictly frozen at inference.
       Tracks normalization drift metrics D_t^{norm} and runtime latency.

Invariants & Constraints:
    - Never uses oracle utility U*, delta_q, delta_t, or test labels.
    - Never accesses future frames.
    - B0 and B1 fit strictly on train split.
    - Strictly deterministic, finite (no NaN, no Inf).
"""
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

from research.phase9b_protocol import (
    CANONICAL_FEATURE_SCHEMA,
    B2_EMA_BETA,
    EPS,
    NORMALIZATION_VARIANTS,
    VARIANT_ALIASES_9B,
)


class StandardNormalizer:
    """Standard train-domain z-score normalizer (B0 baseline)."""

    def __init__(self, eps: float = EPS):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: List[str] = list(CANONICAL_FEATURE_SCHEMA)
        self.n_samples_fit: int = 0

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "StandardNormalizer":
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.mean = np.mean(X, axis=0).astype(np.float32)
        self.std = (np.std(X, axis=0) + self.eps).astype(np.float32)
        self.n_samples_fit = len(X)
        return self

    def reset(self) -> "StandardNormalizer":
        """No-op for static normalizer to maintain uniform interface."""
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardNormalizer must be fit before calling transform.")
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            device = X.device
            mean_t = torch.tensor(self.mean, device=device, dtype=X.dtype)
            std_t = torch.tensor(self.std, device=device, dtype=X.dtype)
            res = (X - mean_t) / std_t
            if not torch.all(torch.isfinite(res)):
                raise FloatingPointError("NaN/Inf produced by StandardNormalizer.transform")
            return res

        res = ((X - self.mean) / self.std).astype(np.float32)
        if not np.all(np.isfinite(res)):
            raise FloatingPointError("NaN/Inf produced by StandardNormalizer.transform")
        return res

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalizer_type": "StandardNormalizer",
            "n_samples_fit": self.n_samples_fit,
            "eps": self.eps,
            "features": {
                name: {
                    "mean": float(self.mean[i]) if self.mean is not None else 0.0,
                    "std": float(self.std[i]) if self.std is not None else 1.0,
                }
                for i, name in enumerate(self.feature_names)
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardNormalizer":
        normalizer = cls(eps=data.get("eps", EPS))
        feats = data.get("features", {})
        names = list(CANONICAL_FEATURE_SCHEMA)
        means = [feats.get(n, {}).get("mean", 0.0) for n in names]
        stds = [feats.get(n, {}).get("std", 1.0) for n in names]
        normalizer.mean = np.array(means, dtype=np.float32)
        normalizer.std = np.array(stds, dtype=np.float32)
        normalizer.n_samples_fit = data.get("n_samples_fit", 0)
        return normalizer

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "StandardNormalizer":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RobustMADNormalizer:
    """Outlier-resistant Median and Median Absolute Deviation (MAD) Normalizer (B1).
    
    Center: median(X)
    Scale:  1.4826 * MAD(X), with safe fallbacks if MAD=0 (e.g. sparse features).
    Fit strictly on train split; test data never enters fitting.
    """

    # Normal consistency factor: 1 / norm.ppf(0.75) approx 1.4826022185
    NORMAL_CONSISTENCY_CONSTANT: float = 1.482602218505602

    def __init__(self, eps: float = EPS):
        self.eps = eps
        self.median: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.raw_mad: Optional[np.ndarray] = None
        self.feature_names: List[str] = list(CANONICAL_FEATURE_SCHEMA)
        self.n_samples_fit: int = 0

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "RobustMADNormalizer":
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X_clean = np.asarray(X, dtype=np.float32)

        n, d = X_clean.shape
        self.median = np.median(X_clean, axis=0).astype(np.float32)
        abs_dev = np.abs(X_clean - self.median)
        self.raw_mad = np.median(abs_dev, axis=0).astype(np.float32)

        # Compute consistent scale with robust fallbacks
        scales = np.zeros(d, dtype=np.float32)
        for j in range(d):
            mad_val = self.raw_mad[j]
            scaled_mad = self.NORMAL_CONSISTENCY_CONSTANT * mad_val
            if scaled_mad > self.eps:
                scales[j] = scaled_mad
            else:
                # MAD is zero (more than 50% identical values, e.g. sparse drift)
                # Fallback 1: Mean Absolute Deviation
                mean_ad = float(np.mean(abs_dev[:, j]))
                if mean_ad > self.eps:
                    scales[j] = mean_ad
                else:
                    # Fallback 2: Standard deviation
                    std_val = float(np.std(X_clean[:, j]))
                    if std_val > self.eps:
                        scales[j] = std_val
                    else:
                        # Fallback 3: constant feature -> unit scale
                        scales[j] = 1.0

        self.scale = np.maximum(scales, self.eps).astype(np.float32)
        self.n_samples_fit = n
        return self

    def reset(self) -> "RobustMADNormalizer":
        """No-op for static normalizer to maintain uniform interface."""
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.median is None or self.scale is None:
            raise RuntimeError("RobustMADNormalizer must be fit before calling transform.")
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            device = X.device
            med_t = torch.tensor(self.median, device=device, dtype=X.dtype)
            scale_t = torch.tensor(self.scale, device=device, dtype=X.dtype)
            res = (X - med_t) / scale_t
            if not torch.all(torch.isfinite(res)):
                raise FloatingPointError("NaN/Inf produced by RobustMADNormalizer.transform")
            return res

        res = ((X - self.median) / self.scale).astype(np.float32)
        if not np.all(np.isfinite(res)):
            raise FloatingPointError("NaN/Inf produced by RobustMADNormalizer.transform")
        return res

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalizer_type": "RobustMADNormalizer",
            "n_samples_fit": self.n_samples_fit,
            "eps": self.eps,
            "features": {
                name: {
                    "median": float(self.median[i]) if self.median is not None else 0.0,
                    "scale": float(self.scale[i]) if self.scale is not None else 1.0,
                    "raw_mad": float(self.raw_mad[i]) if self.raw_mad is not None else 0.0,
                }
                for i, name in enumerate(self.feature_names)
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustMADNormalizer":
        normalizer = cls(eps=data.get("eps", EPS))
        feats = data.get("features", {})
        names = list(CANONICAL_FEATURE_SCHEMA)
        meds = [feats.get(n, {}).get("median", 0.0) for n in names]
        scales = [feats.get(n, {}).get("scale", 1.0) for n in names]
        mads = [feats.get(n, {}).get("raw_mad", 0.0) for n in names]
        normalizer.median = np.array(meds, dtype=np.float32)
        normalizer.scale = np.array(scales, dtype=np.float32)
        normalizer.raw_mad = np.array(mads, dtype=np.float32)
        normalizer.n_samples_fit = data.get("n_samples_fit", 0)
        return normalizer

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "RobustMADNormalizer":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class OnlineEMANormalizer:
    """Test-time Online Adaptive Normalizer with Exponential Moving Average (B2).
    
    Initialized from training statistics:
        mu_0 = mu_train, sigma_0 = sigma_train
    
    Online update at frame t:
        mu_t    = beta * mu_{t-1}    + (1 - beta) * mu_t^{frame}
        sigma_t = beta * sigma_{t-1} + (1 - beta) * sigma_t^{frame}
        tilde_z = (z - mu_t) / (sigma_t + eps)
    
    Guarantees:
        - Completely unlabeled test-time adaptation.
        - Model weights remain frozen.
        - Zero future frame access.
        - Tracks latency and stability metrics (D_t^{norm}).
    """

    def __init__(self, beta: float = B2_EMA_BETA, eps: float = EPS):
        self.beta = beta
        self.eps = eps
        self.mu_init: Optional[np.ndarray] = None
        self.sigma_init: Optional[np.ndarray] = None
        self.mu_current: Optional[np.ndarray] = None
        self.sigma_current: Optional[np.ndarray] = None
        self.feature_names: List[str] = list(CANONICAL_FEATURE_SCHEMA)
        self.n_samples_fit: int = 0
        
        # Temporal tracking history
        self.history: List[Dict[str, Any]] = []

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "OnlineEMANormalizer":
        """Fit initial reference statistics from the train split."""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        X_clean = np.asarray(X, dtype=np.float32)
        self.mu_init = np.mean(X_clean, axis=0).astype(np.float32)
        self.sigma_init = (np.std(X_clean, axis=0) + self.eps).astype(np.float32)
        self.n_samples_fit = len(X_clean)
        self.reset()
        return self

    def reset(self) -> "OnlineEMANormalizer":
        """Reset online dynamic statistics back to initial train reference."""
        if self.mu_init is not None and self.sigma_init is not None:
            self.mu_current = self.mu_init.copy()
            self.sigma_current = self.sigma_init.copy()
        self.history = []
        return self

    def transform_static(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Static transform using current (or initial) state without online update."""
        if self.mu_current is None or self.sigma_current is None:
            raise RuntimeError("OnlineEMANormalizer must be fit before calling transform.")
        is_torch = isinstance(X, torch.Tensor)
        if is_torch:
            device = X.device
            mu_t = torch.tensor(self.mu_current, device=device, dtype=X.dtype)
            sig_t = torch.tensor(self.sigma_current, device=device, dtype=X.dtype)
            return (X - mu_t) / sig_t
        return ((X - self.mu_current) / self.sigma_current).astype(np.float32)

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Standard transform interface (delegates to transform_static)."""
        return self.transform_static(X)

    def update_and_transform_frame(
        self,
        X_frame: Union[np.ndarray, torch.Tensor],
        frame_id: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, float]]:
        """Online step: compute frame stats, update EMA, normalize, record stability.
        
        Args:
            X_frame: [N_frame, 11] features for current frame
            frame_id: optional integer identifier for logging
            
        Returns:
            Z_norm: [N_frame, 11] normalized features
            metrics: dict containing latency and stability indicators
        """
        if self.mu_current is None or self.sigma_current is None:
            raise RuntimeError("OnlineEMANormalizer must be fit before calling update_and_transform_frame.")

        t0 = time.perf_counter()

        is_torch = isinstance(X_frame, torch.Tensor)
        if is_torch:
            X_np = X_frame.detach().cpu().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X_frame, dtype=np.float32)

        prev_mu = self.mu_current.copy()
        prev_sigma = self.sigma_current.copy()

        if len(X_np) > 0:
            frame_mu = np.mean(X_np, axis=0).astype(np.float32)
            frame_std = np.std(X_np, axis=0).astype(np.float32)
            frame_sigma = np.maximum(frame_std, self.eps).astype(np.float32)

            # EMA Update
            new_mu = self.beta * prev_mu + (1.0 - self.beta) * frame_mu
            new_sigma = self.beta * prev_sigma + (1.0 - self.beta) * frame_sigma
            self.mu_current = new_mu.astype(np.float32)
            self.sigma_current = np.maximum(new_sigma, self.eps).astype(np.float32)
        else:
            # Empty frame: keep previous state
            pass

        # Normalize with newly adapted statistics
        if is_torch:
            device = X_frame.device
            mu_t = torch.tensor(self.mu_current, device=device, dtype=X_frame.dtype)
            sig_t = torch.tensor(self.sigma_current, device=device, dtype=X_frame.dtype)
            Z_norm = (X_frame - mu_t) / sig_t
        else:
            Z_norm = ((X_np - self.mu_current) / self.sigma_current).astype(np.float32)

        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Compute stability metrics
        d_norm = float(np.mean(np.abs(self.mu_current - prev_mu)))
        mu_l2 = float(np.linalg.norm(self.mu_current - prev_mu))
        sigma_l2 = float(np.linalg.norm(self.sigma_current - prev_sigma))

        step_metrics = {
            "frame_id": frame_id if frame_id is not None else len(self.history),
            "latency_ms": t_elapsed_ms,
            "d_norm": d_norm,
            "mu_l2_shift": mu_l2,
            "sigma_l2_shift": sigma_l2,
            "mean_mu": float(np.mean(self.mu_current)),
            "mean_sigma": float(np.mean(self.sigma_current)),
        }
        self.history.append(step_metrics)

        return Z_norm, step_metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalizer_type": "OnlineEMANormalizer",
            "beta": self.beta,
            "eps": self.eps,
            "n_samples_fit": self.n_samples_fit,
            "features_init": {
                name: {
                    "mean": float(self.mu_init[i]) if self.mu_init is not None else 0.0,
                    "std": float(self.sigma_init[i]) if self.sigma_init is not None else 1.0,
                }
                for i, name in enumerate(self.feature_names)
            },
            "features_current": {
                name: {
                    "mean": float(self.mu_current[i]) if self.mu_current is not None else 0.0,
                    "std": float(self.sigma_current[i]) if self.sigma_current is not None else 1.0,
                }
                for i, name in enumerate(self.feature_names)
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineEMANormalizer":
        normalizer = cls(beta=data.get("beta", B2_EMA_BETA), eps=data.get("eps", EPS))
        names = list(CANONICAL_FEATURE_SCHEMA)
        
        f_init = data.get("features_init", {})
        means_init = [f_init.get(n, {}).get("mean", 0.0) for n in names]
        stds_init = [f_init.get(n, {}).get("std", 1.0) for n in names]
        normalizer.mu_init = np.array(means_init, dtype=np.float32)
        normalizer.sigma_init = np.array(stds_init, dtype=np.float32)

        f_curr = data.get("features_current", f_init)
        means_curr = [f_curr.get(n, {}).get("mean", 0.0) for n in names]
        stds_curr = [f_curr.get(n, {}).get("std", 1.0) for n in names]
        normalizer.mu_current = np.array(means_curr, dtype=np.float32)
        normalizer.sigma_current = np.array(stds_curr, dtype=np.float32)
        normalizer.n_samples_fit = data.get("n_samples_fit", 0)
        return normalizer

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "OnlineEMANormalizer":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_normalizer(
    variant: str,
    eps: float = EPS,
    beta: float = B2_EMA_BETA,
) -> Union[StandardNormalizer, RobustMADNormalizer, OnlineEMANormalizer]:
    """Factory creating normalizer instance for variant B0, B1, or B2."""
    var_norm = VARIANT_ALIASES_9B.get(variant, variant)
    if var_norm == "A1_standard":
        return StandardNormalizer(eps=eps)
    elif var_norm == "A1_robust_static":
        return RobustMADNormalizer(eps=eps)
    elif var_norm == "A1_online_adaptive":
        return OnlineEMANormalizer(beta=beta, eps=eps)
    else:
        raise ValueError(f"Unknown Phase 9B variant '{variant}'. Expected one of {NORMALIZATION_VARIANTS}")
