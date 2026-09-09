"""Phase 6 Batch Sampler: GroupedBatchSampler.

Guarantees that candidates belonging to the same conditional decision problem
g = (scene_id, frame_id, tuple(sorted(selected_gaussian_ids)))
are NEVER split across training batches.
"""
from typing import Union, Dict, List, Optional
import numpy as np
import torch

from .phase6_dataset import GroupedBatchSampler

__all__ = ["GroupedBatchSampler"]
