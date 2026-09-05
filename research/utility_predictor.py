"""Frozen Utility Predictor for Phase 5.

Strictly encapsulates the frozen Phase 4 Two-Head MLP model and normalizer.
Enforces:
  - Evaluation mode (model.eval())
  - Zero gradients (torch.no_grad() and requires_grad=False)
  - Standardized feature normalization via FeatureNormalizer
  - Execution time profiling (T_pred)
  - Decoupled Quality Gain (\\hat{\\Delta Q}) and Cost (\\hat{\\Delta T})
  - Derived Marginal Utility (\\hat U = \\hat{\\Delta Q} / \\hat{\\Delta T})
"""
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

from research.utility_features import (
    CANONICAL_FEATURE_NAMES,
    UTILITY_FEATURES,
    extract_feature_vector,
)
from research.utility_dataset import FeatureNormalizer
from research.utility_models import TwoHeadMLP


class FrozenUtilityPredictor:
    """Frozen Phase 4 model wrapper for Phase 5 scheduling and selection."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        normalizer_path: Optional[str] = None,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        """Initializes the frozen predictor.

        Args:
            checkpoint_path: Path to PyTorch model checkpoint (.pt).
            normalizer_path: Path to normalization JSON file.
            seed: Seed of model checkpoint if path not explicitly given.
            device: Device to run inference on ('cpu', 'cuda', etc.).
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                repo_root, "results", "learned_utility", "models", f"seed_{seed}.pt"
            )
        if normalizer_path is None:
            normalizer_path = os.path.join(
                repo_root, "results", "learned_utility", "normalization.json"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Frozen checkpoint not found at: {checkpoint_path}")
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Feature normalizer not found at: {normalizer_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Feature Normalizer
        self.normalizer = FeatureNormalizer.load_json(normalizer_path)
        self.feature_names = list(CANONICAL_FEATURE_NAMES)

        # 2. Load Checkpoint and Instantiate TwoHeadMLP
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.in_features = int(ckpt.get("in_features", len(self.feature_names)))
        self.metadata = ckpt.get("metadata", {})
        self.seed = int(self.metadata.get("seed", seed))

        self.model = TwoHeadMLP(in_features=self.in_features).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        
        # 3. Strictly Freeze Model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def predict_features(
        self,
        features: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Predicts Delta Q, Delta T, and Utility for a 2D feature matrix [N, 11].

        Args:
            features: 2D array or tensor of shape [N, 11] with canonical features.

        Returns:
            Dictionary containing:
              - predicted_delta_q: np.ndarray [N]
              - predicted_delta_t: np.ndarray [N]
              - predicted_utility: np.ndarray [N]
              - pred_time_ms: float (total inference latency in ms)
        """
        if isinstance(features, torch.Tensor):
            feat_np = features.detach().cpu().numpy()
        else:
            feat_np = np.asarray(features, dtype=np.float32)

        if feat_np.ndim == 1:
            feat_np = feat_np.reshape(1, -1)

        N = feat_np.shape[0]
        if N == 0:
            return {
                "predicted_delta_q": np.zeros(0, dtype=np.float32),
                "predicted_delta_t": np.zeros(0, dtype=np.float32),
                "predicted_utility": np.zeros(0, dtype=np.float32),
                "pred_time_ms": 0.0,
            }

        # 1. Normalize features strictly using frozen normalizer
        norm_feat = self.normalizer.transform(feat_np)
        input_tensor = torch.tensor(norm_feat, dtype=torch.float32, device=self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            delta_q, delta_t, utility = self.model(input_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        pred_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_delta_q": delta_q.detach().cpu().numpy().astype(np.float32),
            "predicted_delta_t": delta_t.detach().cpu().numpy().astype(np.float32),
            "predicted_utility": utility.detach().cpu().numpy().astype(np.float32),
            "pred_time_ms": float(pred_time_ms),
        }

    def predict_candidates(
        self,
        candidates: List[Dict[str, Any]],
        strict: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Extracts features, runs predictions, and annotates candidates in-place.

        Args:
            candidates: List of candidate dictionaries with 'features' key.
            strict: Whether to enforce strict schema check on all 11 features.

        Returns:
            Tuple of:
              - annotated_candidates: List of candidate dicts with predicted keys added.
              - feat_time_ms: Time spent extracting features in ms.
              - pred_time_ms: Time spent in model inference in ms.
        """
        if not candidates:
            return candidates, 0.0, 0.0

        t_feat_0 = time.perf_counter()
        feat_vectors = []
        for cand in candidates:
            vec = extract_feature_vector(cand, strict=strict)
            feat_vectors.append(vec)
        X = np.stack(feat_vectors, axis=0)
        feat_time_ms = (time.perf_counter() - t_feat_0) * 1000.0

        preds = self.predict_features(X)

        delta_q = preds["predicted_delta_q"]
        delta_t = preds["predicted_delta_t"]
        utility = preds["predicted_utility"]

        for i, cand in enumerate(candidates):
            cand["predicted_delta_q"] = float(delta_q[i])
            cand["predicted_delta_t"] = float(delta_t[i])
            cand["predicted_utility"] = float(utility[i])

        return candidates, float(feat_time_ms), float(preds["pred_time_ms"])
