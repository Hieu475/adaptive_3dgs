"""Canonical Dataset Parser and Split Management for Learned Utility Estimation.

This module provides the single source of truth for parsing oracle_dataset.json,
extracting the 11 canonical features, targets (Delta Q, Delta T, U*), sample metadata,
and strictly enforcing train-only feature normalization.

Guarantees:
  - No ad-hoc json.load(...) or manual dict indexing in experiment scripts.
  - Frozen feature schema matching research.utility_features.
  - Target labels strictly use global metrics: delta_quality, delta_time_ms, oracle_utility_joint.
  - Normalization parameters (mean, std) strictly fit on the train split only.
  - Native PyTorch Dataset compatibility for training loops.
"""
from dataclasses import dataclass, field
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from research.utility_features import (
    CANONICAL_FEATURE_NAMES,
    extract_feature_vector,
)
from research.protocol import (
    load_protocol,
    get_splits,
    get_repo_root,
)


@dataclass
class SampleMetadata:
    """Provenance and context metadata for an individual Gaussian intervention sample."""
    sample_idx: int
    seed: int
    scene: str
    frame: int
    split: str
    gaussian_id: int
    persistent_id: Any
    geometry_stratum: str
    predicted_importance: float
    predicted_utility: float
    influence_mass: float
    projected_area: float
    n_influence_pixels: int


class FeatureNormalizer:
    """Standardizes features using mean and standard deviation fit strictly on the train split."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: List[str] = list(CANONICAL_FEATURE_NAMES)
        self.n_samples_fit: int = 0

    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> "FeatureNormalizer":
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.mean = np.mean(X, axis=0).astype(np.float32)
        self.std = (np.std(X, axis=0) + self.eps).astype(np.float32)
        self.n_samples_fit = len(X)
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer must be fit before calling transform.")
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
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureNormalizer":
        normalizer = cls(eps=data.get("eps", 1e-6))
        feats = data.get("features", {})
        names = list(CANONICAL_FEATURE_NAMES)
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
    def load_json(cls, path: str) -> "FeatureNormalizer":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class UtilityDataset(Dataset):
    """Canonical dataset for Learned Utility Estimation in 3D-GS.
    
    Holds input feature matrix X, targets y_delta_q, y_delta_t, y_utility,
    and metadata for each sample.
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        delta_q: Union[np.ndarray, torch.Tensor],
        delta_t: Union[np.ndarray, torch.Tensor],
        utility: Union[np.ndarray, torch.Tensor],
        metadata: List[SampleMetadata],
        feature_names: Optional[List[str]] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ):
        if isinstance(X, np.ndarray):
            self.X_tensor = torch.from_numpy(X).float()
        else:
            self.X_tensor = X.float()

        if isinstance(delta_q, np.ndarray):
            self.delta_q_tensor = torch.from_numpy(delta_q).float()
        else:
            self.delta_q_tensor = delta_q.float()

        if isinstance(delta_t, np.ndarray):
            self.delta_t_tensor = torch.from_numpy(delta_t).float()
        else:
            self.delta_t_tensor = delta_t.float()

        if isinstance(utility, np.ndarray):
            self.utility_tensor = torch.from_numpy(utility).float()
        else:
            self.utility_tensor = utility.float()

        self._metadata = list(metadata)
        self._feature_names = list(feature_names or CANONICAL_FEATURE_NAMES)
        self.normalizer = normalizer

        assert len(self.X_tensor) == len(self.delta_q_tensor) == len(self.delta_t_tensor) == len(self.utility_tensor) == len(self._metadata), (
            f"Mismatched dataset lengths: X={len(self.X_tensor)}, dq={len(self.delta_q_tensor)}, "
            f"dt={len(self.delta_t_tensor)}, u={len(self.utility_tensor)}, meta={len(self._metadata)}"
        )

    def __len__(self) -> int:
        return len(self.X_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_tensor[idx],
            self.delta_q_tensor[idx],
            self.delta_t_tensor[idx],
            self.utility_tensor[idx],
        )

    @property
    def features(self) -> torch.Tensor:
        """Alias for X_tensor."""
        return self.X_tensor

    @property
    def X(self) -> torch.Tensor:
        return self.X_tensor

    @property
    def delta_q(self) -> torch.Tensor:
        return self.delta_q_tensor

    @property
    def delta_t(self) -> torch.Tensor:
        return self.delta_t_tensor

    @property
    def utility(self) -> torch.Tensor:
        return self.utility_tensor

    @property
    def metadata(self) -> List[SampleMetadata]:
        return self._metadata

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def seeds(self) -> List[int]:
        return [m.seed for m in self._metadata]

    @property
    def frames(self) -> List[int]:
        return [m.frame for m in self._metadata]

    @property
    def scenes(self) -> List[str]:
        return [m.scene for m in self._metadata]

    @property
    def geometry_strata(self) -> List[str]:
        return [m.geometry_stratum for m in self._metadata]

    @property
    def splits(self) -> List[str]:
        return [m.split for m in self._metadata]

    # Numpy array accessors
    @property
    def X_np(self) -> np.ndarray:
        return self.X_tensor.detach().cpu().numpy()

    @property
    def delta_q_np(self) -> np.ndarray:
        return self.delta_q_tensor.detach().cpu().numpy()

    @property
    def delta_t_np(self) -> np.ndarray:
        return self.delta_t_tensor.detach().cpu().numpy()

    @property
    def utility_np(self) -> np.ndarray:
        return self.utility_tensor.detach().cpu().numpy()

    def get_indices(self, indices: Union[List[int], np.ndarray, torch.Tensor]) -> "UtilityDataset":
        """Subset dataset by integer indices."""
        if isinstance(indices, torch.Tensor):
            idx_list = indices.cpu().tolist()
        elif isinstance(indices, np.ndarray):
            idx_list = indices.tolist()
        else:
            idx_list = list(indices)

        sub_X = self.X_tensor[idx_list]
        sub_dq = self.delta_q_tensor[idx_list]
        sub_dt = self.delta_t_tensor[idx_list]
        sub_u = self.utility_tensor[idx_list]
        sub_meta = [self._metadata[i] for i in idx_list]

        return UtilityDataset(
            X=sub_X,
            delta_q=sub_dq,
            delta_t=sub_dt,
            utility=sub_u,
            metadata=sub_meta,
            feature_names=self._feature_names,
            normalizer=self.normalizer,
        )

    def get_split(self, split_name: str) -> "UtilityDataset":
        """Filter dataset by split name ('train', 'validation', 'cross_scene_test', or aliases)."""
        alias_map = {
            "test": "cross_scene_test",
            "val": "validation",
            "train": "train",
        }
        target_split = alias_map.get(split_name, split_name)
        matched_indices = [
            i for i, m in enumerate(self._metadata)
            if m.split == target_split
        ]
        return self.get_indices(matched_indices)

    def filter_by(
        self,
        seed: Optional[int] = None,
        scene: Optional[str] = None,
        frame: Optional[int] = None,
        stratum: Optional[str] = None,
    ) -> "UtilityDataset":
        """Filter dataset by metadata attributes."""
        matched = []
        for i, m in enumerate(self._metadata):
            if seed is not None and m.seed != seed:
                continue
            if scene is not None and m.scene != scene:
                continue
            if frame is not None and m.frame != frame:
                continue
            if stratum is not None and m.geometry_stratum != stratum:
                continue
            matched.append(i)
        return self.get_indices(matched)

    def select_features(self, feature_names_or_indices: Union[List[str], List[int]]) -> "UtilityDataset":
        """Subset features (for ablation experiments V0 to V7)."""
        if len(feature_names_or_indices) == 0:
            raise ValueError("Must select at least one feature.")
            
        if isinstance(feature_names_or_indices[0], str):
            feat_idx = [self._feature_names.index(name) for name in feature_names_or_indices]
            selected_names = list(feature_names_or_indices)
        else:
            feat_idx = list(feature_names_or_indices)
            selected_names = [self._feature_names[i] for i in feat_idx]

        sub_X = self.X_tensor[:, feat_idx]
        return UtilityDataset(
            X=sub_X,
            delta_q=self.delta_q_tensor,
            delta_t=self.delta_t_tensor,
            utility=self.utility_tensor,
            metadata=self._metadata,
            feature_names=selected_names,
            normalizer=None,
        )

    def normalized_with(self, normalizer: FeatureNormalizer) -> "UtilityDataset":
        """Return a new UtilityDataset where features X are transformed by the given normalizer."""
        X_norm = normalizer.transform(self.X_tensor)
        return UtilityDataset(
            X=X_norm,
            delta_q=self.delta_q_tensor,
            delta_t=self.delta_t_tensor,
            utility=self.utility_tensor,
            metadata=self._metadata,
            feature_names=self._feature_names,
            normalizer=normalizer,
        )


def load_canonical_oracle_dataset(
    dataset_path: Optional[str] = None,
    filter_invisible: bool = True,
) -> UtilityDataset:
    """Load and canonically parse the Phase 3 Oracle Dataset.
    
    Args:
        dataset_path: Path to oracle_dataset.json. If None, resolves from repo root.
        filter_invisible: If True, retains only Gaussians with visible=True and n_influence_pixels > 0.
        
    Returns:
        Canonical UtilityDataset instance.
    """
    if dataset_path is None:
        repo_root = get_repo_root()
        dataset_path = os.path.join(repo_root, "results", "oracle_dataset", "oracle_dataset.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Canonical oracle dataset not found at: {dataset_path}")

    with open(dataset_path, "r") as f:
        raw_rows = json.load(f)

    X_list = []
    dq_list = []
    dt_list = []
    u_list = []
    meta_list = []

    for idx, row in enumerate(raw_rows):
        is_visible = row.get("visible", True)
        n_pixels = int(row.get("n_influence_pixels", row.get("features", {}).get("visibility_count", 1)))
        
        if filter_invisible and (not is_visible or n_pixels <= 0):
            continue

        # Extract 11 canonical features
        x_vec = extract_feature_vector(row)

        # Canonical global targets (MUST-FIX #4 from Protocol v1)
        dq = float(row.get("delta_quality", row.get("delta_quality_global", 0.0)))
        dt = float(row.get("delta_time_ms", row.get("measured_trial_cost_ms", 1.0)))
        u = float(row.get("oracle_utility_joint", row.get("oracle_utility_joint_global", row.get("oracle_utility", 0.0))))

        seed = int(row.get("dataset_seed", row.get("seed", 42)))
        scene = str(row.get("scene", "tum_fr1_desk"))
        frame = int(row.get("frame", 0))
        split = str(row.get("split", "train"))
        gid = int(row.get("gaussian_id", 0))
        pid = row.get("persistent_id", gid)
        stratum = str(row.get("geometry_stratum", "unknown"))
        pred_imp = float(row.get("predicted_importance", 0.0))
        pred_u = float(row.get("predicted_utility", 0.0))
        inf_mass = float(row.get("influence_mass", 1.0))
        proj_area = float(row.get("projected_area", 1.0))

        metadata = SampleMetadata(
            sample_idx=idx,
            seed=seed,
            scene=scene,
            frame=frame,
            split=split,
            gaussian_id=gid,
            persistent_id=pid,
            geometry_stratum=stratum,
            predicted_importance=pred_imp,
            predicted_utility=pred_u,
            influence_mass=inf_mass,
            projected_area=proj_area,
            n_influence_pixels=n_pixels,
        )

        X_list.append(x_vec)
        dq_list.append(dq)
        dt_list.append(dt)
        u_list.append(u)
        meta_list.append(metadata)

    X_arr = np.stack(X_list, axis=0) if len(X_list) > 0 else np.zeros((0, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32)
    dq_arr = np.array(dq_list, dtype=np.float32)
    dt_arr = np.array(dt_list, dtype=np.float32)
    u_arr = np.array(u_list, dtype=np.float32)

    return UtilityDataset(
        X=X_arr,
        delta_q=dq_arr,
        delta_t=dt_arr,
        utility=u_arr,
        metadata=meta_list,
        feature_names=CANONICAL_FEATURE_NAMES,
    )


def prepare_normalized_splits(
    dataset: Optional[UtilityDataset] = None,
    save_stats_path: Optional[str] = None,
) -> Tuple[UtilityDataset, UtilityDataset, UtilityDataset, FeatureNormalizer]:
    """Prepares train, validation, and test splits with normalization strictly fit on train.
    
    Returns:
        (train_norm, val_norm, test_norm, normalizer)
    """
    if dataset is None:
        dataset = load_canonical_oracle_dataset()

    train_raw = dataset.get_split("train")
    val_raw = dataset.get_split("validation")
    test_raw = dataset.get_split("cross_scene_test")

    # Strictly fit normalization on train split only
    normalizer = FeatureNormalizer()
    normalizer.fit(train_raw.X_tensor)

    train_norm = train_raw.normalized_with(normalizer)
    val_norm = val_raw.normalized_with(normalizer)
    test_norm = test_raw.normalized_with(normalizer)

    if save_stats_path is not None:
        normalizer.save_json(save_stats_path)

    return train_norm, val_norm, test_norm, normalizer
