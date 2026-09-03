"""
Base Experiment Engine for Adaptive 3DGS Research.

Provides standardized setup, execution timing, GPU memory tracking,
and provenance bundle emission across all research experiments.
"""
import os
import sys
import time
import abc
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any

from research.schema import (
    ExperimentMetadata,
    QualityMetrics,
    LatencyMetrics,
    MemoryMetrics,
    SelectionMetrics,
    ExperimentMetrics,
    ExperimentResult,
)
from research.reproducibility import (
    get_git_commit,
    get_hardware_info,
    set_seed,
    save_experiment_bundle,
)


class BaseExperiment(abc.ABC):
    """Abstract base class for all scientific experiments in the project."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        dataset_name: str = "TUM",
        scene_name: str = "freiburg1_desk",
        seed: int = 42,
        output_base_dir: Optional[str] = None,
    ):
        self.name = name
        self.config = config
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        self.seed = seed
        self.timestamp = datetime.now().isoformat()
        
        # Determine output directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = output_base_dir or os.path.join(project_root, "results", self.name)
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base, f"run_{time_tag}_seed{seed}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = ExperimentMetadata(
            name=self.name,
            version="1.0.0",
            git_commit=get_git_commit(),
            timestamp=self.timestamp,
            seed=self.seed,
            dataset=self.dataset_name,
            scene=self.scene_name,
            frame_range=[0, 0],
            hardware=get_hardware_info(),
            config=self.config,
        )
        
        self.metrics = ExperimentMetrics()
        self.start_wall_time = 0.0

    def setup(self) -> None:
        """Standardized pre-experiment setup."""
        set_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.start_wall_time = time.time()

    @abc.abstractmethod
    def run(self) -> ExperimentResult:
        """Core experimental execution logic to be implemented by child classes."""
        pass

    def teardown(self, markdown_report: str = "") -> ExperimentResult:
        """Standardized post-experiment metrics finalization and bundle export."""
        elapsed_sec = time.time() - self.start_wall_time
        
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            self.metrics.memory.gpu_vram_peak_mb = round(peak_mb, 2)
            self.metrics.memory.allocated_mb = round(alloc_mb, 2)
            
        result = ExperimentResult(experiment=self.metadata, metrics=self.metrics)
        save_experiment_bundle(result, self.run_dir, markdown_report)
        return result
