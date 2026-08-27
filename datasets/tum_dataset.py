"""TUM RGB-D dataset loader.

TUM RGB-D Benchmark structure:
    <sequence>/
        rgb/
            <timestamp>.png
        depth/
            <timestamp>.png
        associations.txt or rgb.txt + depth.txt + groundtruth.txt

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from .base_dataset import BaseDataset


class TUMDataset(BaseDataset):
    """Loader for TUM RGB-D benchmark sequences."""
    
    # TUM default intrinsics (Freiburg 1)
    DEFAULT_FX = 517.3
    DEFAULT_FY = 516.5
    DEFAULT_CX = 318.6
    DEFAULT_CY = 255.3
    
    def __init__(
        self,
        data_path: str,
        depth_scale: float = 5000.0,  # TUM depth scale
        max_frames: Optional[int] = None,
        stride: int = 1,
        camera: str = 'freiburg1',
    ):
        super().__init__(data_path, depth_scale)
        self.max_frames = max_frames
        self.stride = stride
        
        # Set intrinsics based on camera
        if camera == 'freiburg2':
            self.fx, self.fy = 520.9, 521.0
            self.cx, self.cy = 325.1, 249.7
        elif camera == 'freiburg3':
            self.fx, self.fy = 535.4, 539.2
            self.cx, self.cy = 320.1, 247.6
        else:  # freiburg1
            self.fx, self.fy = self.DEFAULT_FX, self.DEFAULT_FY
            self.cx, self.cy = self.DEFAULT_CX, self.DEFAULT_CY
        
        self.associations: List[Tuple[str, str]] = []
        self.poses: List[torch.Tensor] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load associations and ground truth."""
        assoc_file = self.data_path / 'associations.txt'
        
        if assoc_file.exists():
            self._load_associations(assoc_file)
        else:
            # Auto-associate using timestamps
            self._auto_associate()
        
        # Load ground truth
        gt_file = self.data_path / 'groundtruth.txt'
        if gt_file.exists():
            self._load_groundtruth(gt_file)
        else:
            self.poses = [torch.eye(4) for _ in range(len(self.associations))]
    
    def _load_associations(self, assoc_file: Path):
        """Load pre-computed associations."""
        with open(assoc_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    rgb_file = parts[1]
                    depth_file = parts[3]
                    self.associations.append((rgb_file, depth_file))
        
        self.associations = self.associations[::self.stride]
        if self.max_frames:
            self.associations = self.associations[:self.max_frames]
    
    def _auto_associate(self):
        """Auto-associate RGB and depth by timestamp proximity."""
        rgb_dir = self.data_path / 'rgb'
        depth_dir = self.data_path / 'depth'
        
        if not rgb_dir.exists() or not depth_dir.exists():
            return
        
        rgb_files = sorted(rgb_dir.glob('*.png'))
        depth_files = sorted(depth_dir.glob('*.png'))
        
        # Simple pairing by index
        for i, (rf, df) in enumerate(zip(rgb_files, depth_files)):
            self.associations.append((str(rf.relative_to(self.data_path)), 
                                     str(df.relative_to(self.data_path))))
        
        self.associations = self.associations[::self.stride]
        if self.max_frames:
            self.associations = self.associations[:self.max_frames]
    
    def _load_groundtruth(self, gt_file: Path):
        """Load ground truth poses (tx ty tz qx qy qz qw format)."""
        gt_data = []
        with open(gt_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = [float(x) for x in line.strip().split()]
                if len(parts) >= 8:
                    gt_data.append(parts)
        
        # Convert to 4x4 matrices
        for data in gt_data[::self.stride][:len(self.associations)]:
            tx, ty, tz = data[1], data[2], data[3]
            qx, qy, qz, qw = data[4], data[5], data[6], data[7]
            
            # Quaternion to rotation matrix
            R = self._quat_to_rot(qw, qx, qy, qz)
            T = torch.eye(4)
            T[:3, :3] = R
            T[:3, 3] = torch.tensor([tx, ty, tz])
            self.poses.append(T)
        
        # Pad if needed
        while len(self.poses) < len(self.associations):
            self.poses.append(torch.eye(4))
    
    @staticmethod
    def _quat_to_rot(w, x, y, z) -> torch.Tensor:
        R = torch.tensor([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ], dtype=torch.float32)
        return R
    
    def __len__(self) -> int:
        return len(self.associations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV required: pip install opencv-python")
        
        rgb_path = self.data_path / self.associations[idx][0]
        depth_path = self.data_path / self.associations[idx][1]
        
        rgb_np = cv2.imread(str(rgb_path))
        if rgb_np is None:
            raise FileNotFoundError(f"Cannot read {rgb_path}")
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb_np).float() / 255.0
        
        depth_np = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_np is not None:
            depth = torch.from_numpy(depth_np.astype(np.float32)) / self.depth_scale
        else:
            depth = torch.zeros(rgb.shape[0], rgb.shape[1])
        
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
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
