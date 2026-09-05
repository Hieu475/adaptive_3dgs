"""Phase 6: Dataset Loading and Normalization for Context-Aware Utility.

Extends Phase 4 FeatureNormalizer to handle 32-dim Phase 6 feature vectors.
Provides PyTorch Dataset for training loops.

Invariants:
    - Normalization fitted strictly on train split only.
    - Phase 6 features: self(11) + neighbor(8) + overlap(5) + selected(8) = 32.
    - Targets: delta_q_conditional, delta_t_conditional_ms, utility_conditional.
    - No leakage: train/val/test splits identical to Phase 4 protocol.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from .phase6_context import PHASE6_FEATURE_NAMES, PHASE6_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 Feature Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class Phase6FeatureNormalizer:
    """Standardizes 32-dim Phase 6 features using train-only statistics.

    Extends the Phase 4 FeatureNormalizer pattern to the full 32-dim vector:
    [self(11), neighbor(8), overlap(5), selected(8)].

    Usage:
        normalizer = Phase6FeatureNormalizer()
        normalizer.fit(X_train)           # Fit on train split ONLY
        X_train_norm = normalizer.transform(X_train)
        X_val_norm = normalizer.transform(X_val)    # Use train stats
        X_test_norm = normalizer.transform(X_test)   # Use train stats
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None  # Shape (32,)
        self.std: Optional[np.ndarray] = None   # Shape (32,)
        self.feature_names: List[str] = list(PHASE6_FEATURE_NAMES)
        self.n_samples_fit: int = 0

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "Phase6FeatureNormalizer":
        """Fit normalization parameters on training data.

        Args:
            X: (N, 32) feature matrix.

        Returns:
            Self for chaining.
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.mean = np.mean(X, axis=0).astype(np.float32)
        self.std = (np.std(X, axis=0) + self.eps).astype(np.float32)
        self.n_samples_fit = len(X)
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Transform features using fitted statistics.

        Args:
            X: (N, 32) or (32,) feature matrix/vector.

        Returns:
            Normalized features in same type as input.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Phase6FeatureNormalizer must be fit before transform.")
        if isinstance(X, torch.Tensor):
            device = X.device
            mean_t = torch.tensor(self.mean, device=device, dtype=X.dtype)
            std_t = torch.tensor(self.std, device=device, dtype=X.dtype)
            return (X - mean_t) / std_t
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": "phase6",
            "n_features": len(self.feature_names),
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
    def from_dict(cls, data: Dict[str, Any]) -> "Phase6FeatureNormalizer":
        normalizer = cls(eps=data.get("eps", 1e-6))
        feats = data.get("features", {})
        names = list(PHASE6_FEATURE_NAMES)
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
    def load_json(cls, path: str) -> "Phase6FeatureNormalizer":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Phase6UtilityDataset(Dataset):
    """PyTorch Dataset for Phase 6 conditional utility training.

    Each sample contains:
        - features: (32,) normalized Phase 6 feature vector
        - delta_q: scalar conditional quality gain ΔQ(i|S)
        - delta_t: scalar conditional cost ΔT(i|S) in ms
        - utility: scalar conditional utility U*(i|S)
        - context_size: int, size of context set |S|
    """

    def __init__(
        self,
        features: np.ndarray,
        delta_q: np.ndarray,
        delta_t: np.ndarray,
        utility: np.ndarray,
        context_sizes: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        """Initialize dataset.

        Args:
            features: (N, 32) feature matrix.
            delta_q: (N,) conditional quality gains.
            delta_t: (N,) conditional costs in ms.
            utility: (N,) conditional utilities.
            context_sizes: (N,) context set sizes.
            metadata: Optional list of sample metadata dicts.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.delta_q = torch.tensor(delta_q, dtype=torch.float32)
        self.delta_t = torch.tensor(delta_t, dtype=torch.float32)
        self.utility = torch.tensor(utility, dtype=torch.float32)
        self.context_sizes = context_sizes
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.features[idx],
            'delta_q': self.delta_q[idx],
            'delta_t': self.delta_t[idx],
            'utility': self.utility[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_phase6_dataset(
    dataset_path: str,
    split: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Load Phase 6 conditional oracle dataset from JSON.

    Args:
        dataset_path: Path to conditional_oracle_seed_*.json.
        split: Optional split filter ("train", "validation", "cross_scene_test").

    Returns:
        Tuple of (features, delta_q, delta_t, utility, raw_samples).
    """
    with open(dataset_path, 'r') as f:
        samples = json.load(f)

    if split is not None:
        samples = [s for s in samples if s.get("split") == split]

    N = len(samples)
    if N == 0:
        return (
            np.zeros((0, PHASE6_FEATURE_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )

    features = np.array([s["full_feature_vector"] for s in samples], dtype=np.float32)
    delta_q = np.array([s["delta_q_conditional"] for s in samples], dtype=np.float32)
    delta_t = np.array([s["delta_t_conditional_ms"] for s in samples], dtype=np.float32)
    utility = np.array([s["utility_conditional"] for s in samples], dtype=np.float32)

    assert features.shape == (N, PHASE6_FEATURE_DIM), (
        f"Expected ({N}, {PHASE6_FEATURE_DIM}), got {features.shape}"
    )

    return features, delta_q, delta_t, utility, samples


def prepare_phase6_splits(
    dataset_paths: List[str],
    normalizer_save_path: Optional[str] = None,
    variant: str = "V11",
) -> Tuple[Phase6UtilityDataset, Phase6UtilityDataset, Phase6UtilityDataset, Phase6FeatureNormalizer]:
    """Load, normalize, and split Phase 6 dataset.

    Normalization is fitted strictly on train split only.

    Args:
        dataset_paths: List of paths to conditional oracle JSON files
            (multiple seeds can be merged).
        normalizer_save_path: Optional path to save normalizer JSON.
        variant: Ablation variant for feature subsetting.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, normalizer).
    """
    all_train_feats, all_train_dq, all_train_dt, all_train_u = [], [], [], []
    all_val_feats, all_val_dq, all_val_dt, all_val_u = [], [], [], []
    all_test_feats, all_test_dq, all_test_dt, all_test_u = [], [], [], []

    for path in dataset_paths:
        with open(path, 'r') as f:
            samples = json.load(f)

        for s in samples:
            feat = np.array(s["full_feature_vector"], dtype=np.float32)
            dq = float(s["delta_q_conditional"])
            dt = float(s["delta_t_conditional_ms"])
            u = float(s["utility_conditional"])

            split = s.get("split", "cross_scene_test")
            if split == "train":
                all_train_feats.append(feat)
                all_train_dq.append(dq)
                all_train_dt.append(dt)
                all_train_u.append(u)
            elif split == "validation":
                all_val_feats.append(feat)
                all_val_dq.append(dq)
                all_val_dt.append(dt)
                all_val_u.append(u)
            else:
                all_test_feats.append(feat)
                all_test_dq.append(dq)
                all_test_dt.append(dt)
                all_test_u.append(u)

    def _to_arrays(feats, dq, dt, u):
        if not feats:
            return (
                np.zeros((0, PHASE6_FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.stack(feats),
            np.array(dq, dtype=np.float32),
            np.array(dt, dtype=np.float32),
            np.array(u, dtype=np.float32),
        )

    X_train, dq_train, dt_train, u_train = _to_arrays(
        all_train_feats, all_train_dq, all_train_dt, all_train_u
    )
    X_val, dq_val, dt_val, u_val = _to_arrays(
        all_val_feats, all_val_dq, all_val_dt, all_val_u
    )
    X_test, dq_test, dt_test, u_test = _to_arrays(
        all_test_feats, all_test_dq, all_test_dt, all_test_u
    )

    # Fit normalizer on train split ONLY
    normalizer = Phase6FeatureNormalizer()
    if len(X_train) > 0:
        normalizer.fit(X_train)
        X_train = normalizer.transform(X_train)
        if len(X_val) > 0:
            X_val = normalizer.transform(X_val)
        if len(X_test) > 0:
            X_test = normalizer.transform(X_test)

    if normalizer_save_path:
        normalizer.save_json(normalizer_save_path)

    # Apply variant feature subsetting
    feature_mask = _get_variant_mask(variant)

    train_ds = Phase6UtilityDataset(
        X_train[:, feature_mask] if len(X_train) > 0 else X_train,
        dq_train, dt_train, u_train,
    )
    val_ds = Phase6UtilityDataset(
        X_val[:, feature_mask] if len(X_val) > 0 else X_val,
        dq_val, dt_val, u_val,
    )
    test_ds = Phase6UtilityDataset(
        X_test[:, feature_mask] if len(X_test) > 0 else X_test,
        dq_test, dt_test, u_test,
    )

    return train_ds, val_ds, test_ds, normalizer


def _get_variant_mask(variant: str) -> np.ndarray:
    """Get boolean feature mask for ablation variant.

    Args:
        variant: One of 'V8', 'V9', 'V10', 'V11'.

    Returns:
        Boolean array of length 32.
    """
    mask = np.zeros(PHASE6_FEATURE_DIM, dtype=bool)

    # Self features always included
    mask[:11] = True

    if variant in ('V9', 'V10', 'V11'):
        mask[11:19] = True   # Neighbor

    if variant in ('V10', 'V11'):
        mask[19:24] = True   # Overlap

    if variant == 'V11':
        mask[24:32] = True   # Selected

    return mask
