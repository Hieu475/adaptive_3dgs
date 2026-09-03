"""
Reproducibility, Provenance, and Statistical Confidence Utilities.

Guarantees full scientific reproducibility by logging hardware state,
git revisions, random seeds, and computing bootstrap confidence intervals.
"""
import os
import sys
import subprocess
import platform
import random
import time
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

from research.schema import ExperimentResult


def get_git_commit() -> str:
    """Retrieve current git commit hash, with dirty flag if uncommitted changes exist."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        if status:
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    """Retrieve active git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Capture detailed hardware, OS, and PyTorch runtime specifications."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if cuda_available else 0.0
    )
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_name": device_name,
        "vram_gb": round(vram_gb, 2),
        "cpu_count": os.cpu_count(),
    }


def set_seed(seed: int = 42) -> None:
    """Strictly set deterministic random seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for a given metric.
    
    Args:
        data: sample observations
        stat_fn: statistical estimator (mean, median, etc.)
        n_boot: number of bootstrap resamples
        ci: confidence level (default: 0.95 for 95% CI)
        seed: RNG seed for bootstrap
        
    Returns:
        (ci_lower, ci_upper): bounds of the empirical confidence interval
    """
    arr = np.asarray(data)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return (val, val)
        
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_stats = np.empty(n_boot)
    
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)
        
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100.0 * alpha))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return (round(lower, 4), round(upper, 4))


def save_experiment_bundle(
    result: ExperimentResult,
    run_dir: str,
    markdown_report: str = "",
) -> None:
    """Save immutable-ish experiment bundle (Section XXXII).
    
    Creates:
      - result.json
      - config.yaml
      - git_commit.txt
      - environment.txt
      - report.md
    """
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. result.json
    result_path = os.path.join(run_dir, "result.json")
    result.to_json(result_path)
    
    # 2. config.yaml
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(result.experiment.config, f, default_flow_style=False)
        
    # 3. git_commit.txt
    commit_path = os.path.join(run_dir, "git_commit.txt")
    with open(commit_path, "w") as f:
        f.write(f"Commit: {result.experiment.git_commit}\n")
        f.write(f"Branch: {get_git_branch()}\n")
        f.write(f"Timestamp: {result.experiment.timestamp}\n")
        
    # 4. environment.txt
    env_path = os.path.join(run_dir, "environment.txt")
    with open(env_path, "w") as f:
        for k, v in result.experiment.hardware.items():
            f.write(f"{k}: {v}\n")
            
    # 5. report.md
    if markdown_report:
        report_path = os.path.join(run_dir, "report.md")
        with open(report_path, "w") as f:
            f.write(markdown_report)
            
    print(f"[Bundle] Saved complete reproducibility bundle to: {run_dir}")


class DatasetManifest:
    """Explicit dataset provenance manifest (Section XXXIII)."""
    def __init__(
        self,
        dataset: str,
        sequence: str,
        frames: List[int],
        rgb_resolution: Tuple[int, int],
        depth_scale: float = 5000.0,
        pose_source: str = "ground_truth",
    ):
        self.dataset = dataset
        self.sequence = sequence
        self.frames = frames
        self.rgb_resolution = f"{rgb_resolution[0]}x{rgb_resolution[1]}"
        self.depth_scale = depth_scale
        self.pose_source = pose_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "frames": self.frames,
            "rgb_resolution": self.rgb_resolution,
            "depth_scale": self.depth_scale,
            "pose_source": self.pose_source,
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
