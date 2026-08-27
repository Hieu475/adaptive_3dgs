"""
OpenGL viewer stub with camera controller.
"""
import numpy as np

class CameraController:
    """Handles WASD and mouse inputs for camera movement."""
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.pitch = 0.0

    def update(self, dt: float):
        # TODO: Process inputs
        pass

class OpenGLViewer:
    """Main viewer class."""
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.camera = CameraController()

    def run(self):
        """Starts the viewer loop."""
        # TODO: Initialize window and start loop
        pass
