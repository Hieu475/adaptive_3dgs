"""Setup script for Adaptive 3DGS with CUDA extensions.

Install:
    pip install -e .           # Python-only (CPU)
    pip install -e . --cuda    # With CUDA extensions

Or explicitly:
    python setup.py develop
"""
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def get_cuda_extensions():
    """Build CUDA extension if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping CUDA extensions")
            return []
    except ImportError:
        return []
    
    cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
    cuda_sources = [
        os.path.join(cuda_dir, f) for f in os.listdir(cuda_dir)
        if f.endswith('.cu')
    ]
    
    if not cuda_sources:
        return []
    
    # Add a binding file
    binding_file = os.path.join(cuda_dir, 'bindings.cpp')
    if os.path.exists(binding_file):
        cuda_sources.append(binding_file)
    
    return [
        CUDAExtension(
            name='adaptive_3dgs._C',
            sources=cuda_sources,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                ],
            },
        )
    ]


setup(
    name='adaptive_3dgs',
    version='0.1.0',
    description='Adaptive 3D Gaussian Splatting for Real-Time Online Reconstruction',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0',
        'numpy',
        'pyyaml',
        'opencv-python',
        'tqdm',
    ],
    extras_require={
        'eval': ['lpips', 'scikit-image'],
        'viewer': ['PyOpenGL', 'glfw'],
        'dev': ['pytest', 'black', 'flake8'],
    },
    ext_modules=get_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension} if get_cuda_extensions() else {},
)
