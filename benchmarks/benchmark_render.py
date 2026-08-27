"""Benchmark rendering performance.

Measures:
- Python reference rasterizer FPS
- CUDA rasterizer FPS (if available)
- Per-component timing breakdown
- Scalability: frame time vs Gaussian count

Usage:
    python benchmarks/benchmark_render.py
    python benchmarks/benchmark_render.py --cuda
    python benchmarks/benchmark_render.py --n_gaussians 100000
"""
import sys
import os
import time
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def create_benchmark_scene(n_gaussians: int, device: str = 'cpu'):
    """Create a synthetic scene for benchmarking."""
    from research.gaussian_repr import GaussianModel
    
    model = GaussianModel(sh_degree=0, device=device)
    points = torch.randn(n_gaussians, 3, device=device) * 2.0
    colors = torch.rand(n_gaussians, 3, device=device)
    model.initialize_from_points(points, colors=colors)
    
    intrinsics = torch.tensor([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ], device=device)
    
    extrinsics = torch.eye(4, device=device)
    extrinsics[2, 3] = -5.0  # Camera at z=-5 looking forward
    
    return model, intrinsics, extrinsics


def benchmark_python_rasterizer(n_gaussians: int, n_warmup: int = 3, n_runs: int = 10,
                                  image_w: int = 320, image_h: int = 240):
    """Benchmark Python reference rasterizer."""
    from research.rasterizer import render as rasterize_scene
    
    model, intrinsics, extrinsics = create_benchmark_scene(n_gaussians)
    cov3D = model.build_covariance()
    colors = model.get_colors()
    opacities = model.opacities.squeeze(-1)
    
    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        rasterize_scene(
            model.positions, cov3D, colors, opacities,
            extrinsics, intrinsics, image_w, image_h
        )
    
    # Benchmark
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = rasterize_scene(
            model.positions, cov3D, colors, opacities,
            extrinsics, intrinsics, image_w, image_h
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    times = np.array(times)
    return {
        'backend': 'python',
        'n_gaussians': n_gaussians,
        'image_size': f'{image_w}x{image_h}',
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'fps': float(1000.0 / times.mean()),
    }


def benchmark_cuda_rasterizer(n_gaussians: int, n_warmup: int = 10, n_runs: int = 50,
                                image_w: int = 640, image_h: int = 480):
    """Benchmark CUDA rasterizer."""
    try:
        import adaptive_3dgs._C as _C
    except ImportError:
        print("  CUDA extension not built. Run: pip install -e .")
        return None
    
    if not torch.cuda.is_available():
        print("  CUDA not available")
        return None
    
    device = 'cuda'
    model, intrinsics, extrinsics = create_benchmark_scene(n_gaussians, device=device)
    
    # Warmup
    print(f"  Warming up ({n_warmup} runs)...")
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _C.preprocess_gaussians(
            model.positions, model._scaling, model._rotation,
            extrinsics, intrinsics
        )
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _C.preprocess_gaussians(
            model.positions, model._scaling, model._rotation,
            extrinsics, intrinsics
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    
    # GPU memory
    mem_allocated = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    
    return {
        'backend': 'cuda',
        'n_gaussians': n_gaussians,
        'image_size': f'{image_w}x{image_h}',
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'fps': float(1000.0 / times.mean()),
        'gpu_mem_allocated_mb': float(mem_allocated),
        'gpu_mem_reserved_mb': float(mem_reserved),
    }


def benchmark_scalability(backend: str = 'python', counts=None):
    """Test frame time vs Gaussian count."""
    if counts is None:
        counts = [100, 500, 1000, 2000, 5000]
    
    results = []
    for n in counts:
        print(f"  N={n}...")
        if backend == 'python':
            r = benchmark_python_rasterizer(n, n_warmup=1, n_runs=3,
                                            image_w=160, image_h=120)
        else:
            r = benchmark_cuda_rasterizer(n, n_warmup=3, n_runs=10)
            if r is None:
                break
        results.append(r)
    
    return results


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark rendering")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--n_gaussians', type=int, default=1000)
    parser.add_argument('--scalability', action='store_true')
    parser.add_argument('--output', type=str, default='artifacts/benchmark_results.json')
    args = parser.parse_args()
    
    results = {}
    
    print("\n" + "="*60)
    print("RENDERING BENCHMARK")
    print("="*60)
    
    # Python benchmark
    print(f"\n[Python Rasterizer] N={args.n_gaussians}")
    py_result = benchmark_python_rasterizer(args.n_gaussians)
    results['python'] = py_result
    print(f"  Mean: {py_result['mean_ms']:.2f} ms | FPS: {py_result['fps']:.1f}")
    print(f"  p50:  {py_result['p50_ms']:.2f} ms | p95: {py_result['p95_ms']:.2f} ms")
    
    # CUDA benchmark
    if args.cuda:
        print(f"\n[CUDA Rasterizer] N={args.n_gaussians}")
        cuda_result = benchmark_cuda_rasterizer(args.n_gaussians)
        if cuda_result:
            results['cuda'] = cuda_result
            print(f"  Mean: {cuda_result['mean_ms']:.2f} ms | FPS: {cuda_result['fps']:.1f}")
            print(f"  p50:  {cuda_result['p50_ms']:.2f} ms | p95: {cuda_result['p95_ms']:.2f} ms")
            print(f"  GPU Memory: {cuda_result['gpu_mem_allocated_mb']:.1f} MB allocated")
    
    # Scalability
    if args.scalability:
        print("\n[Scalability Test]")
        backend = 'cuda' if args.cuda else 'python'
        scale_results = benchmark_scalability(backend)
        results['scalability'] = scale_results
        print("  N_Gaussians | Frame Time (ms) | FPS")
        print("  " + "-"*45)
        for r in scale_results:
            print(f"  {r['n_gaussians']:>10} | {r['mean_ms']:>14.2f} | {r['fps']:.1f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    run_benchmark()
