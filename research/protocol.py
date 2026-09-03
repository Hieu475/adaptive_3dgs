"""Protocol v1 Specification and Unified Experiment Configuration Loader.

Guarantees zero researcher degrees of freedom by centralizing:
  - Resolution (320x240)
  - Seeds [42, 43, 44, 45, 46]
  - Dataset paths & camera configurations
  - Explicit Splits (Train: fr1/desk 0-40, Val: fr1/desk 41-60, Test: fr2/xyz)
  - Oracle specification and budget constraints
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml


def get_repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parent.parent


def load_protocol(path: str = "configs/protocol_v1.yaml") -> Dict[str, Any]:
    """Load frozen protocol configuration from yaml file."""
    p = Path(path)
    if not p.is_absolute():
        p = get_repo_root() / path
    if not p.exists():
        raise FileNotFoundError(f"Protocol file not found at: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_seeds(protocol: Optional[Dict[str, Any]] = None) -> List[int]:
    """Return frozen multi-seed list (n=5)."""
    if protocol is None:
        protocol = load_protocol()
    return list(protocol["reproducibility"]["seeds"])


def get_dataset_config(dataset_name: str = "tum_fr1_desk", protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return dataset config with resolution and paths."""
    if protocol is None:
        protocol = load_protocol()
    cfg = dict(protocol["datasets"][dataset_name])
    raw_path = cfg["path"]
    if not os.path.isabs(raw_path):
        cfg["full_path"] = str(get_repo_root() / raw_path)
    else:
        cfg["full_path"] = raw_path
    return cfg


def get_resolution(dataset_name: str = "tum_fr1_desk", protocol: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """Return (H, W) for the specified dataset (e.g. 240, 320)."""
    cfg = get_dataset_config(dataset_name, protocol)
    return int(cfg["image_height"]), int(cfg["image_width"])


def get_splits(protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return canonical split specifications.
    
    Train: fr1_desk frames 0-40
    Val:   fr1_desk frames 41-60
    Test:  fr2_xyz
    """
    if protocol is None:
        protocol = load_protocol()
    return {
        "train_frames": list(range(0, 41)),
        "val_frames": list(range(41, 61)),
        "test_scene": "tum_fr2_xyz",
        "raw_splits": protocol.get("splits", {})
    }


def get_oracle_config(protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return oracle specification dictionary."""
    if protocol is None:
        protocol = load_protocol()
    return dict(protocol["oracle_specification"])


def get_budget_config(protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return budget levels configuration dictionary."""
    if protocol is None:
        protocol = load_protocol()
    return dict(protocol["budget_levels"])


def get_statistics_config(protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return statistical testing configuration dictionary."""
    if protocol is None:
        protocol = load_protocol()
    return dict(protocol["statistical_testing"])


def get_state_factor_config(protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return state factor feature configuration dictionary."""
    if protocol is None:
        protocol = load_protocol()
    return dict(protocol["state_factors"])

