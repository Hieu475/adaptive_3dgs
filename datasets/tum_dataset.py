"""
TUM RGB-D dataset loader.
"""
from .base_dataset import BaseDataset

class TUMDataset(BaseDataset):
    """Loader for TUM sequences."""
    def __init__(self, data_path: str):
        self.data_path = data_path
        # TODO: Load dataset index (associate.py logic)
        
    def __len__(self) -> int:
        return 0
        
    def __getitem__(self, idx: int):
        return None
