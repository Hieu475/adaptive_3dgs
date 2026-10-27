"""Provenance and statistical-confidence utilities.

These helpers record run identity and seed Python, NumPy, and PyTorch RNGs.
Seeding alone does not guarantee bitwise reproducibility for CUDA kernels,
third-party GPU libraries, or algorithms with unstable tie handling.
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
    """Seed Python, NumPy, and PyTorch RNGs and set cuDNN repeatability hints.

    This does not enable ``torch.use_deterministic_algorithms`` and does not
    guarantee identical CUDA execution or results across repeated runs.
    """
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


def compute_config_hash(config: Union[Dict[str, Any], str]) -> str:
    """Compute SHA-256 hash of configuration dict or YAML file."""
    import hashlib
    if isinstance(config, str):
        if os.path.exists(config):
            with open(config, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        return hashlib.sha256(config.encode("utf-8")).hexdigest()
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def classify_experiment_manifest(
    manifest: Dict[str, Any], artifact_path: str = ""
) -> Dict[str, Any]:
    """Conservatively classify whether an experiment bundle is paper-usable.

    This validator never upgrades an artifact based on its reported metrics.
    It checks provenance fields and verifies any declared artifact hashes.
    """
    path_lower = str(artifact_path).replace("\\", "/").lower()
    issues: List[str] = []
    warnings: List[str] = []

    if "final_confirmation" in path_lower and "phase14_corrected" not in path_lower:
        return {
            "status": "legacy",
            "issues": ["Known pre-pose-fix final_confirmation artifact."],
            "warnings": ["Do not use its absolute metrics in primary tables."],
        }
    if manifest.get("protocol_version") == "phase13-authoritative-v1":
        return {
            "status": "legacy",
            "issues": ["Superseded Phase 13 v1 protocol lacks current pose/config provenance."],
            "warnings": ["Re-run with the v3 protocol before using in primary tables."],
        }

    git_info = manifest.get("git", {}) if isinstance(manifest.get("git"), dict) else {}
    source_snapshots = manifest.get("source_snapshots", [])
    snapshot_commits = [str(s.get("commit", "")) for s in source_snapshots
                        if isinstance(s, dict)] if isinstance(source_snapshots, list) else []
    commit = str(manifest.get("git_sha") or git_info.get("sha") or
                 manifest.get("source_snapshot_commit") or
                 (snapshot_commits[0] if snapshot_commits else ""))
    if not commit or commit == "unknown":
        warnings.append("Missing commit SHA.")
    if "dirty" in commit.lower():
        issues.append("Commit SHA records a dirty worktree.")

    config_hash = str(manifest.get("config_sha256") or "")
    artifacts = manifest.get("artifacts", {})
    if config_hash in ("", "none", "unknown"):
        if isinstance(artifacts, dict) and artifacts.get("config_snapshot.yaml"):
            config_hash = str(artifacts["config_snapshot.yaml"])
        elif manifest.get("source_runner_sha256"):
            config_hash = str(manifest["source_runner_sha256"])
            warnings.append("Using runner source hash; no effective runtime config hash is recorded.")
        else:
            warnings.append("Missing effective configuration hash or config snapshot hash.")

    scope = manifest.get("scope", {})
    dataset = manifest.get("dataset_name") or manifest.get("dataset") or manifest.get("scene")
    if not dataset:
        protocol = manifest.get("protocol", {})
        dataset = protocol.get("scene") if isinstance(protocol, dict) else None
    if not dataset and isinstance(scope, dict):
        dataset = scope.get("scenes") or scope.get("scene")
    if not dataset:
        warnings.append("Missing dataset/scene identity.")

    protocol = manifest.get("protocol", {})
    pose = manifest.get("pose_convention") or (
        protocol.get("pose_convention") if isinstance(protocol, dict) else None
    )
    if not pose:
        warnings.append("Missing camera pose convention.")

    seeds = (manifest.get("seeds") or manifest.get("seed") or
             (protocol.get("seeds") if isinstance(protocol, dict) else None) or
             (scope.get("seeds") if isinstance(scope, dict) else None))
    if not seeds and manifest.get("seed") is None:
        warnings.append("Missing seed/replicate identity.")

    backend = manifest.get("renderer_backend") or manifest.get("backend")
    if not backend and isinstance(protocol, dict):
        backend = protocol.get("backend")
    if not backend and isinstance(scope, dict):
        backend = scope.get("backend")
    if not backend:
        warnings.append("Missing renderer backend.")

    declared_status = str(manifest.get("status", "")).lower()
    tree_clean = git_info.get("experiment_relevant_tree_clean")
    if tree_clean is False:
        issues.append("Experiment-relevant worktree was not clean.")
    if declared_status != "frozen":
        warnings.append("Manifest does not declare a frozen protocol status.")
    if source_snapshots and any(
        isinstance(s, dict) and s.get("worktree_status") != "clean"
        for s in source_snapshots
    ):
        issues.append("At least one recorded source snapshot was not clean.")

    # Verify file hashes when an artifact directory and a hash inventory exist.
    if artifact_path and isinstance(artifacts, dict):
        import hashlib
        for relpath, expected in artifacts.items():
            expected_text = str(expected)
            if len(expected_text) == 64 and all(c in "0123456789abcdefABCDEF" for c in expected_text):
                candidate = os.path.join(artifact_path, relpath)
                if not os.path.isfile(candidate):
                    issues.append(f"Hashed artifact is missing: {relpath}.")
                    continue
                digest = hashlib.sha256()
                with open(candidate, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_text.lower():
                    issues.append(f"Artifact hash mismatch: {relpath}.")
            else:
                import glob
                pattern = expected_text.replace("<scene>", "*")
                candidate = os.path.normpath(os.path.join(artifact_path, pattern))
                if not glob.glob(candidate):
                    issues.append(f"Declared artifact path is missing: {expected_text}.")

    if issues:
        status = "invalid"
    elif declared_status == "frozen" and not warnings:
        status = "paper_ready"
    else:
        status = "exploratory"
    return {"status": status, "issues": issues, "warnings": warnings}


def create_provenance_manifest(
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "tum_fr2_xyz",
    dataset_path: Optional[str] = None,
    renderer_backend: Optional[str] = None,
    init_stride: Optional[int] = None,
    scale_pixel_multiplier: Optional[float] = None,
    initial_opacity: Optional[float] = None,
    n_micro_steps: Optional[int] = None,
    psnr_mask: str = "valid_depth",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create comprehensive provenance manifest for scientific reproducibility.
    
    Contains exact execution environment, commit hash, configuration SHA-256,
    backend provenance, and reconstruction hyperparameters.
    """
    hw = get_hardware_info()
    git_sha = get_git_commit()
    
    cuda_ext_status = "unavailable"
    try:
        import adaptive_3dgs._C
        cuda_ext_status = "available (prototype)"
    except Exception as e:
        cuda_ext_status = f"unavailable: {e}"

    if renderer_backend is None:
        from research.rasterizer import get_renderer_backend
        renderer_backend = get_renderer_backend(device_type="cuda" if hw["cuda_available"] else "cpu")

    config_hash = compute_config_hash(config) if config is not None else "none"

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha,
        "git_branch": get_git_branch(),
        "config_sha256": config_hash,
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "renderer_backend": renderer_backend,
        "cuda_extension": cuda_ext_status,
        "torch_version": hw["pytorch"],
        "cuda_version": hw["cuda_version"],
        "gpu": hw["device_name"],
        "vram_gb": hw["vram_gb"],
        "init_stride": init_stride,
        "scale_pixel_multiplier": scale_pixel_multiplier,
        "initial_opacity": initial_opacity,
        "n_micro_steps": n_micro_steps,
        "psnr_mask": psnr_mask,
    }
    if extra_metadata:
        manifest.update(extra_metadata)
    return manifest

