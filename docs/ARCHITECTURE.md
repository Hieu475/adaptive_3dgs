# Architecture Overview

This document describes the high-level architecture of the Adaptive 3DGS framework.

## Components

- **Scene Representation**: C++ & CUDA implementations for managing 3D Gaussians.
- **Rendering Engine**: Tile-based rasterizer in CUDA.
- **Tracking & Mapping**: Real-time camera tracking and scene map update.
- **Adaptive Scheduler**: Allocates computational budget dynamically.
