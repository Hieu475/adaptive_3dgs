/**
 * @file statistics.cu
 * @brief CUDA kernels for collecting per-Gaussian statistics.
 * 
 * Collects:
 * - Per-Gaussian rendering contribution (alpha * transmittance)
 * - Visibility count
 * - Screen-space area
 * - Mean color/depth error
 * 
 * These feed into the importance estimator and budget scheduler.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void compute_per_gaussian_stats_kernel(
    int N_pixels,
    int N_gaussians,
    const int* __restrict__ pixel_gaussian_map,  // (H*W,) dominant Gaussian per pixel
    const float* __restrict__ color_errors,       // (H*W,)
    const float* __restrict__ depth_errors,       // (H*W,)
    float* __restrict__ gaussian_color_err_sum,   // (N_gaussians,) atomic add target
    float* __restrict__ gaussian_depth_err_sum,   // (N_gaussians,)
    int* __restrict__ gaussian_pixel_count         // (N_gaussians,)
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= N_pixels) return;
    
    int g_idx = pixel_gaussian_map[pixel_idx];
    if (g_idx < 0 || g_idx >= N_gaussians) return;
    
    // Atomic accumulation
    atomicAdd(&gaussian_color_err_sum[g_idx], color_errors[pixel_idx]);
    atomicAdd(&gaussian_depth_err_sum[g_idx], depth_errors[pixel_idx]);
    atomicAdd(&gaussian_pixel_count[g_idx], 1);
}


__global__ void normalize_gaussian_stats_kernel(
    int N_gaussians,
    float* __restrict__ color_err_mean,   // (N,) in-place: sum -> mean
    float* __restrict__ depth_err_mean,   // (N,)
    const int* __restrict__ pixel_count   // (N,)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_gaussians) return;
    
    int count = pixel_count[idx];
    if (count > 0) {
        float inv_count = 1.0f / (float)count;
        color_err_mean[idx] *= inv_count;
        depth_err_mean[idx] *= inv_count;
    }
}
