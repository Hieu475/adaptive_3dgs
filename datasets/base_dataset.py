"""
Base dataset class for RGB-D sequences.
"""
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Base dataset interface."""
    
    @abstractmethod
    def __len__(self) -> int:
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int):
        pass
