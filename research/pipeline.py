import torch
from .gaussian_repr import GaussianModel
from .tracker import ICPTracker
from .scheduler import BudgetScheduler

class OnlineReconstructionPipeline:
    """
    Main online reconstruction pipeline for Adaptive 3D Gaussian Splatting.
    """
    def __init__(self):
        self.gaussian_model = GaussianModel()
        self.tracker = ICPTracker()
        self.scheduler = BudgetScheduler(gpu_budget_ms=33.3) # 30 FPS target
        
    def initialize(self, first_rgb: torch.Tensor, first_depth: torch.Tensor, pose: torch.Tensor):
        """
        Initializes the model from the first RGB-D frame.
        """
        # TODO: Initialize Gaussians from first frame
        pass

    def process_frame(self, rgb: torch.Tensor, depth: torch.Tensor):
        """
        Full per-frame pipeline: 
        track -> render -> compute errors -> densify -> schedule -> optimize -> prune
        """
        # TODO: Online tracking and mapping step
        pass

    def get_gaussian_map(self) -> GaussianModel:
        """
        Returns the current state of the 3D Gaussian Splatting map.
        """
        return self.gaussian_model
