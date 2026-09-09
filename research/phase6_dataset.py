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

    def fit(self, X: Union[np.ndarray, torch.Tensor], anchor_p4: bool = True) -> "Phase6FeatureNormalizer":
        """Fit normalization parameters on training data.

        Args:
            X: (N, 32) feature matrix.
            anchor_p4: If True, anchors the 11 self-features to the frozen Phase 4
                       normalizer to preserve exact numerical consistency.

        Returns:
            Self for chaining.
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self.mean = np.mean(X, axis=0).astype(np.float32)
        raw_std = np.std(X, axis=0).astype(np.float32)
        # For constant/near-zero variance features, set std to 1.0 to avoid 1e-6 scaling explosion
        raw_std[raw_std < 1e-4] = 1.0
        self.std = raw_std
        self.n_samples_fit = len(X)

        if anchor_p4:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            p4_norm_path = os.path.join(repo_root, "results", "learned_utility", "normalization.json")
            if os.path.exists(p4_norm_path):
                import json
                with open(p4_norm_path, "r") as f:
                    p4_data = json.load(f)
                p4_feats = p4_data.get("features", {})
                for idx, name in enumerate(self.feature_names[:11]):
                    if name in p4_feats:
                        self.mean[idx] = float(p4_feats[name]["mean"])
                        self.std[idx] = float(p4_feats[name]["std"])

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
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Phase6UtilityDataset(Dataset):
    """PyTorch Dataset for Phase 6 conditional utility training.

    Each sample contains:
        - features: (D,) normalized Phase 6 feature vector (D <= 32 based on variant)
        - delta_q: scalar conditional quality gain ΔQ(i|S)
        - delta_t: scalar conditional cost ΔT(i|S) in ms
        - utility: scalar conditional utility U*(i|S)
        - target_r: scalar residual target r* = U*(i|S) - U*(i|∅) (P0.4)
        - residual_utility: identical to target_r for explicit semantic naming
        - is_empty: float scalar (1.0 if S=∅ else 0.0) for empty context penalty (P0.4)
        - group_id: int scalar identifying context group (P0.1)
        - context_size: int, size of context set |S|
    """

    def __init__(
        self,
        features: np.ndarray,
        delta_q: np.ndarray,
        delta_t: np.ndarray,
        utility: np.ndarray,
        target_r: Optional[np.ndarray] = None,
        is_empty: Optional[np.ndarray] = None,
        group_ids: Optional[np.ndarray] = None,
        context_sizes: Optional[np.ndarray] = None,
        scene_ids: Optional[List[str]] = None,
        frame_ids: Optional[np.ndarray] = None,
        context_identities: Optional[List[Tuple[int, ...]]] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.delta_q = torch.tensor(delta_q, dtype=torch.float32)
        self.delta_t = torch.tensor(delta_t, dtype=torch.float32)
        self.utility = torch.tensor(utility, dtype=torch.float32)

        N = len(features)
        if target_r is not None:
            self.target_r = torch.tensor(target_r, dtype=torch.float32)
        else:
            self.target_r = torch.zeros(N, dtype=torch.float32)

        if is_empty is not None:
            self.is_empty = torch.tensor(is_empty, dtype=torch.float32)
        else:
            self.is_empty = torch.zeros(N, dtype=torch.float32)

        if group_ids is not None:
            self.group_ids = torch.tensor(group_ids, dtype=torch.long)
        else:
            self.group_ids = torch.zeros(N, dtype=torch.long)

        self.context_sizes = context_sizes
        self.scene_ids = scene_ids
        self.frame_ids = frame_ids
        self.context_identities = context_identities
        self.metadata = metadata

        # Explicit target fields (P0.4)
        self.utility_conditional = self.utility
        self.utility_residual = self.target_r
        self.residual_utility = self.target_r
        self.utility_empty = self.utility - self.target_r

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.features[idx],
            'delta_q': self.delta_q[idx],
            'delta_t': self.delta_t[idx],
            'utility': self.utility[idx],
            'target_r': self.target_r[idx],
            'residual_utility': self.target_r[idx],
            'is_empty': self.is_empty[idx],
            'group_id': self.group_ids[idx],
            'utility_conditional': self.utility[idx],
            'utility_residual': self.target_r[idx],
            'utility_empty': self.utility[idx] - self.target_r[idx],
        }


class GroupedBatchSampler:
    """Yields batches where all samples belong to intact context groups (P0.1, Sửa số 3).

    Guarantees that a conditional candidate group g = (scene, frame, S_t) is
    NEVER split across batches. Can pack multiple intact groups up to max_batch_size,
    or yield 1 group per batch if max_batch_size is None or <= 1.
    """

    def __init__(
        self,
        group_ids: Union[np.ndarray, torch.Tensor],
        max_batch_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if isinstance(group_ids, torch.Tensor):
            group_ids = group_ids.cpu().numpy()
        self.group_ids = np.asarray(group_ids)
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by group_id
        self.group_to_indices: Dict[int, List[int]] = {}
        for idx, gid in enumerate(self.group_ids):
            gid = int(gid)
            if gid not in self.group_to_indices:
                self.group_to_indices[gid] = []
            self.group_to_indices[gid].append(idx)

        self.groups = sorted(list(self.group_to_indices.keys()))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups = list(self.groups)
        if self.shuffle:
            rng.shuffle(groups)

        if self.max_batch_size is None or self.max_batch_size <= 1:
            for g in groups:
                yield self.group_to_indices[g]
        else:
            current_batch: List[int] = []
            for g in groups:
                g_indices = self.group_to_indices[g]
                # If adding this group exceeds max_batch_size and current_batch is non-empty, yield current
                if len(current_batch) > 0 and (len(current_batch) + len(g_indices) > self.max_batch_size):
                    yield current_batch
                    current_batch = []
                current_batch.extend(g_indices)
            if len(current_batch) > 0:
                yield current_batch

    def __len__(self):
        if self.max_batch_size is None or self.max_batch_size <= 1:
            return len(self.groups)
        count = 0
        cur_len = 0
        for g in self.groups:
            g_len = len(self.group_to_indices[g])
            if cur_len > 0 and (cur_len + g_len > self.max_batch_size):
                count += 1
                cur_len = 0
            cur_len += g_len
        if cur_len > 0:
            count += 1
        return count

    def set_epoch(self, epoch: int):
        self.epoch = epoch


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
    """Load, normalize, and split Phase 6 dataset with group-awareness (P0.1) and residual targets (P0.4).

    Normalization is fitted strictly on train split only.
    Computes ground truth residual targets r* = U*(i|S) - U*(i|∅) and empty context indicator.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, normalizer).
    """
    all_train_feats, all_train_dq, all_train_dt, all_train_u = [], [], [], []
    all_val_feats, all_val_dq, all_val_dt, all_val_u = [], [], [], []
    all_test_feats, all_test_dq, all_test_dt, all_test_u = [], [], [], []

    train_r, train_empty, train_gids = [], [], []
    val_r, val_empty, val_gids = [], [], []
    test_r, test_empty, test_gids = [], [], []

    train_scenes, val_scenes, test_scenes = [], [], []
    train_frames, val_frames, test_frames = [], [], []
    train_ctx_ids, val_ctx_ids, test_ctx_ids = [], [], []

    # 1. Pre-pass across all loaded files: map (scene, frame, candidate_id) -> utility(empty)
    empty_utils: Dict[Tuple[str, int, int], float] = {}
    for path in dataset_paths:
        with open(path, 'r') as f:
            samples = json.load(f)
        for s in samples:
            if s.get("context_size", 0) == 0 or s.get("context_type") == "empty":
                k = (str(s["scene"]), int(s["frame"]), int(s["candidate_id"]))
                empty_utils[k] = float(s["utility_conditional"])

    group_key_to_id: Dict[Tuple[str, int, Tuple[int, ...]], int] = {}
    group_to_split: Dict[Tuple[str, int, Tuple[int, ...]], str] = {}

    for path in dataset_paths:
        with open(path, 'r') as f:
            samples = json.load(f)

        for s in samples:
            feat = np.array(s["full_feature_vector"], dtype=np.float32)
            dq = float(s["delta_q_conditional"])
            dt = float(s["delta_t_conditional_ms"])
            u = float(s["utility_conditional"])

            # Compute residual target r* = U*(i|S) - U*(i|∅) (P0.4)
            cand_key = (str(s["scene"]), int(s["frame"]), int(s["candidate_id"]))
            u_empty = empty_utils.get(cand_key, u)
            r_target = u - u_empty
            assert abs(r_target - (u - u_empty)) < 1e-8, "Residual target definition must satisfy r* = U*(i|S) - U*(i|∅)"
            is_empty_val = 1.0 if (s.get("context_size", 0) == 0 or s.get("context_type") == "empty") else 0.0

            # Canonical group key representing exact context identity (Sửa số 2 & 3)
            # g = (scene_id, frame_id, tuple(sorted(selected_gaussian_ids)))
            selected_ids = s.get("context_ids", []) or []
            context_ident = tuple(sorted([int(x) for x in selected_ids]))
            g_key = (str(s["scene"]), int(s["frame"]), context_ident)
            gid = group_key_to_id.setdefault(g_key, len(group_key_to_id))

            scene_str = str(s["scene"])
            frame_int = int(s["frame"])

            split = s.get("split", "cross_scene_test")
            # Invariant: No exact group may be split across train/val/test splits
            if g_key in group_to_split and group_to_split[g_key] != split:
                raise ValueError(
                    f"Split-integrity violation: exact group {g_key} assigned to multiple splits: "
                    f"'{group_to_split[g_key]}' vs '{split}'. All candidates in an exact context group must share the same split."
                )
            group_to_split[g_key] = split
            if split == "train":
                all_train_feats.append(feat)
                all_train_dq.append(dq)
                all_train_dt.append(dt)
                all_train_u.append(u)
                train_r.append(r_target)
                train_empty.append(is_empty_val)
                train_gids.append(gid)
                train_scenes.append(scene_str)
                train_frames.append(frame_int)
                train_ctx_ids.append(context_ident)
            elif split == "validation":
                all_val_feats.append(feat)
                all_val_dq.append(dq)
                all_val_dt.append(dt)
                all_val_u.append(u)
                val_r.append(r_target)
                val_empty.append(is_empty_val)
                val_gids.append(gid)
                val_scenes.append(scene_str)
                val_frames.append(frame_int)
                val_ctx_ids.append(context_ident)
            else:
                all_test_feats.append(feat)
                all_test_dq.append(dq)
                all_test_dt.append(dt)
                all_test_u.append(u)
                test_r.append(r_target)
                test_empty.append(is_empty_val)
                test_gids.append(gid)
                test_scenes.append(scene_str)
                test_frames.append(frame_int)
                test_ctx_ids.append(context_ident)

    def _to_arrays(feats, dq, dt, u, r, empty, gids):
        if not feats:
            return (
                np.zeros((0, PHASE6_FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return (
            np.stack(feats),
            np.array(dq, dtype=np.float32),
            np.array(dt, dtype=np.float32),
            np.array(u, dtype=np.float32),
            np.array(r, dtype=np.float32),
            np.array(empty, dtype=np.float32),
            np.array(gids, dtype=np.int64),
        )

    X_train, dq_train, dt_train, u_train, r_train, empty_train, gids_train = _to_arrays(
        all_train_feats, all_train_dq, all_train_dt, all_train_u, train_r, train_empty, train_gids
    )
    X_val, dq_val, dt_val, u_val, r_val, empty_val, gids_val = _to_arrays(
        all_val_feats, all_val_dq, all_val_dt, all_val_u, val_r, val_empty, val_gids
    )
    X_test, dq_test, dt_test, u_test, r_test, empty_test, gids_test = _to_arrays(
        all_test_feats, all_test_dq, all_test_dt, all_test_u, test_r, test_empty, test_gids
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
        target_r=r_train,
        is_empty=empty_train,
        group_ids=gids_train,
        scene_ids=train_scenes,
        frame_ids=np.array(train_frames, dtype=np.int64),
        context_identities=train_ctx_ids,
    )
    val_ds = Phase6UtilityDataset(
        X_val[:, feature_mask] if len(X_val) > 0 else X_val,
        dq_val, dt_val, u_val,
        target_r=r_val,
        is_empty=empty_val,
        group_ids=gids_val,
        scene_ids=val_scenes,
        frame_ids=np.array(val_frames, dtype=np.int64),
        context_identities=val_ctx_ids,
    )
    test_ds = Phase6UtilityDataset(
        X_test[:, feature_mask] if len(X_test) > 0 else X_test,
        dq_test, dt_test, u_test,
        target_r=r_test,
        is_empty=empty_test,
        group_ids=gids_test,
        scene_ids=test_scenes,
        frame_ids=np.array(test_frames, dtype=np.int64),
        context_identities=test_ctx_ids,
    )

    return train_ds, val_ds, test_ds, normalizer


def _get_variant_mask(variant: str) -> np.ndarray:
    """Get boolean feature mask for ablation variant.

    Supports both legacy ladder (V8-V11) and full 2-way combinatorial ablation:
      - self_only (V8)
      - self_neighbor (V9)
      - self_overlap
      - self_selected
      - self_neighbor_overlap (V10)
      - self_neighbor_selected
      - self_overlap_selected
      - all_features (V11)

    Returns:
        Boolean array of length 32.
    """
    mask = np.zeros(PHASE6_FEATURE_DIM, dtype=bool)

    # Self features always included (0:11)
    mask[:11] = True

    v = variant.lower()
    if v in ('v9', 'v10', 'v11', 'self_neighbor', 'self_neighbor_overlap', 'self_neighbor_selected', 'all_features'):
        mask[11:19] = True   # Neighbor (11:19)

    if v in ('v10', 'v11', 'self_overlap', 'self_neighbor_overlap', 'self_overlap_selected', 'all_features'):
        mask[19:24] = True   # Overlap (19:24)

    if v in ('v11', 'self_selected', 'self_neighbor_selected', 'self_overlap_selected', 'all_features'):
        mask[24:32] = True   # Selected (24:32)

    return mask
