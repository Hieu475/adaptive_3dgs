"""Base dataset class for RGB-D sequences."""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path


class BaseDataset(ABC):
    """Base dataset interface for RGB-D sequences.
    
    Each item returns a dict with:
        'rgb': (H, W, 3) float tensor in [0, 1]
        'depth': (H, W) float tensor in meters
        'pose': (4, 4) camera-to-world transform (optional)
        'intrinsics': (3, 3) camera intrinsic matrix
    """
    
    def __init__(self, data_path: str, depth_scale: float = 1000.0):
        self.data_path = Path(data_path)
        self.depth_scale = depth_scale
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass
    
    @property
    @abstractmethod
    def intrinsics(self) -> torch.Tensor:
        """Return (3, 3) intrinsic matrix."""
        pass
    
    @property
    def image_size(self) -> tuple:
        """Return (H, W) image dimensions."""
        item = self[0]
        return item['rgb'].shape[:2]
