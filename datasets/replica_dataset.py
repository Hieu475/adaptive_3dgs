"""Replica dataset loader.

Replica dataset structure:
    <scene>/
        results/
            frame000000.jpg
            depth000000.png
            ...
        traj.txt
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .base_dataset import BaseDataset


class ReplicaDataset(BaseDataset):
    """Loader for Replica RGB-D sequences.
    
    Supports the iMAP/NICE-SLAM Replica format with:
    - RGB images as JPG/PNG
    - Depth images as 16-bit PNG (depth in mm)
    - Camera trajectory as traj.txt
    """
    
    # Default Replica intrinsics (from iMAP/NICE-SLAM)
    DEFAULT_FX = 600.0
    DEFAULT_FY = 600.0
    DEFAULT_CX = 599.5
    DEFAULT_CY = 339.5
    DEFAULT_W = 1200
    DEFAULT_H = 680
    
    def __init__(
        self,
        data_path: str,
        depth_scale: float = 6553.5,  # Replica depth scale
        max_frames: Optional[int] = None,
        stride: int = 1,
    ):
        super().__init__(data_path, depth_scale)
        self.max_frames = max_frames
        self.stride = stride
        
        # Discover frames
        self.rgb_files: List[Path] = []
        self.depth_files: List[Path] = []
        self.poses: List[torch.Tensor] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Discover and index all frames."""
        results_dir = self.data_path / 'results'
        
        if not results_dir.exists():
            # Try alternative structure: direct frame files
            results_dir = self.data_path
        
        # Find RGB files
        rgb_pattern_jpg = sorted(results_dir.glob('frame*.jpg'))
        rgb_pattern_png = sorted(results_dir.glob('frame*.png'))
        rgb_files = rgb_pattern_jpg if rgb_pattern_jpg else rgb_pattern_png
        
        if not rgb_files:
            # Try color_* pattern
            rgb_files = sorted(results_dir.glob('color_*.png')) or sorted(results_dir.glob('rgb_*.png'))
        
        # Find depth files
        depth_files = sorted(results_dir.glob('depth*.png')) or sorted(results_dir.glob('depth_*.png'))
        
        # Apply stride
        rgb_files = rgb_files[::self.stride]
        depth_files = depth_files[::self.stride]
        
        # Limit frames
        if self.max_frames is not None:
            rgb_files = rgb_files[:self.max_frames]
            depth_files = depth_files[:self.max_frames]
        
        self.rgb_files = rgb_files
        self.depth_files = depth_files[:len(rgb_files)]
        
        # Load trajectory
        traj_file = self.data_path / 'traj.txt'
        if traj_file.exists():
            self.poses = self._load_trajectory(traj_file)
        else:
            # Generate identity poses
            self.poses = [torch.eye(4) for _ in range(len(self.rgb_files))]
    
    def _load_trajectory(self, traj_file: Path) -> List[torch.Tensor]:
        """Load camera trajectory from traj.txt (4x4 matrices)."""
        poses = []
        with open(traj_file, 'r') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 4):
            if i + 4 > len(lines):
                break
            mat = []
            for j in range(4):
                row = [float(x) for x in lines[i + j].strip().split()]
                mat.append(row)
            pose = torch.tensor(mat, dtype=torch.float32)
            poses.append(pose)
        
        # Apply stride
        poses = poses[::self.stride]
        if self.max_frames is not None:
            poses = poses[:self.max_frames]
        
        return poses
    
    def __len__(self) -> int:
        return len(self.rgb_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a frame."""
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")
        
        # Load RGB
        rgb_np = cv2.imread(str(self.rgb_files[idx]))
        if rgb_np is None:
            raise FileNotFoundError(f"Cannot read {self.rgb_files[idx]}")
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb_np).float() / 255.0  # (H, W, 3)
        
        # Load depth
        if idx < len(self.depth_files):
            depth_np = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_UNCHANGED)
            if depth_np is None:
                depth = torch.zeros(rgb.shape[0], rgb.shape[1])
            else:
                depth = torch.from_numpy(depth_np.astype(np.float32)) / self.depth_scale
        else:
            depth = torch.zeros(rgb.shape[0], rgb.shape[1])
        
        # Get pose
        pose = self.poses[idx] if idx < len(self.poses) else torch.eye(4)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'pose': pose,
            'intrinsics': self.intrinsics,
        }
    
    @property
    def intrinsics(self) -> torch.Tensor:
        return torch.tensor([
            [self.DEFAULT_FX, 0.0, self.DEFAULT_CX],
            [0.0, self.DEFAULT_FY, self.DEFAULT_CY],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
