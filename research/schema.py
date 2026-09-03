"""
Unified Scientific Experiment Schema for Adaptive 3DGS.

Enforces reproducibility across all research trials:
Every result record encapsulates exact code version, configuration, dataset,
scene, random seed, hardware specifications, and structured metrics.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import yaml
import os


@dataclass
class ExperimentMetadata:
    """Rigorous provenance and environment metadata."""
    name: str
    version: str = "1.0.0"
    git_commit: str = "unknown"
    timestamp: str = ""
    seed: int = 42
    dataset: str = "TUM"
    scene: str = "freiburg1_desk"
    frame_range: List[int] = field(default_factory=lambda: [0, 10])
    hardware: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Reconstruction fidelity metrics with distribution statistics."""
    psnr_mean: float = 0.0
    psnr_std: float = 0.0
    psnr_ci95: Tuple[float, float] = (0.0, 0.0)
    ssim_mean: float = 0.0
    ssim_std: float = 0.0
    ssim_ci95: Tuple[float, float] = (0.0, 0.0)
    depth_l1_mean: float = 0.0
    depth_l1_std: float = 0.0
    depth_l1_ci95: Tuple[float, float] = (0.0, 0.0)
    delta_psnr_vs_random: float = 0.0
    delta_psnr_vs_full: float = 0.0
    per_frame_psnr: List[float] = field(default_factory=list)
    per_frame_depth: List[float] = field(default_factory=list)


@dataclass
class LatencyMetrics:
    """Execution timing and budget adherence statistics."""
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    jitter_ms: float = 0.0
    target_budget_ms: float = 0.0
    utilization_rate: float = 0.0
    violation_rate: float = 0.0
    recovery_time_frames: float = 0.0
    breakdown_ms: Dict[str, float] = field(default_factory=dict)
    per_frame_opt_ms: List[float] = field(default_factory=list)


@dataclass
class MemoryMetrics:
    """GPU memory footprint and Gaussian population dynamics."""
    active_gaussians_mean: float = 0.0
    active_ratio_mean: float = 0.0
    total_gaussians_final: int = 0
    gpu_vram_peak_mb: float = 0.0
    allocated_mb: float = 0.0


@dataclass
class SelectionMetrics:
    """Ranking fidelity, overlap, and scheduler efficiency."""
    spearman_rho: float = 0.0
    p_value: float = 1.0
    overlap_10pct: float = 0.0
    overlap_20pct: float = 0.0
    gain_efficiency: float = 0.0
    gain_ratio_20pct: float = 0.0
    regret_20pct: float = 0.0
    switches_per_frame: float = 0.0
    interaction_error: float = 0.0


@dataclass
class ExperimentMetrics:
    """Aggregated metrics container."""
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    selection: SelectionMetrics = field(default_factory=SelectionMetrics)


@dataclass
class ExperimentResult:
    """Top-level container uniting provenance metadata with multidimensional metrics."""
    experiment: ExperimentMetadata
    metrics: ExperimentMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire experiment result to nested JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Export result to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_yaml(self, filepath: str) -> None:
        """Export result to YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Construct ExperimentResult from dictionary."""
        exp_meta = ExperimentMetadata(**data.get('experiment', {}))
        m = data.get('metrics', {})
        q_m = QualityMetrics(**m.get('quality', {}))
        l_m = LatencyMetrics(**m.get('latency', {}))
        mem_m = MemoryMetrics(**m.get('memory', {}))
        s_m = SelectionMetrics(**m.get('selection', {}))
        metrics = ExperimentMetrics(quality=q_m, latency=l_m, memory=mem_m, selection=s_m)
        return cls(experiment=exp_meta, metrics=metrics)
