/**
 * @file depth_render.cu
 * @brief CUDA kernel for RTG-SLAM style surface-aware depth rendering.
 * 
 * Ray-plane intersection:
 *   θ = (p_G - t_cam) · n_G / (d_ray · n_G)
 *   p_hit = t_cam + θ · d_ray
 *   D(u) = ||p_hit - t_cam||₂
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length3(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}


__global__ void depth_render_surface_aware_kernel(
    int width,
    int height,
    const float3* __restrict__ gaussian_centers,   // (N, 3) world space
    const float3* __restrict__ gaussian_normals,   // (N, 3) world space
    const int* __restrict__ dominant_gaussian_idx,  // (H*W,) dominant Gaussian per pixel
    const float3* __restrict__ ray_dirs,            // (H*W, 3) world-space ray directions
    float3 camera_pos,                              // camera center in world space
    float* __restrict__ depth_out,                  // (H*W,) output depth
    int* __restrict__ valid_out                     // (H*W,) output validity
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= width * height) return;
    
    int g_idx = dominant_gaussian_idx[pixel_idx];
    if (g_idx < 0) {
        depth_out[pixel_idx] = 0.0f;
        valid_out[pixel_idx] = 0;
        return;
    }
    
    float3 p_G = gaussian_centers[g_idx];
    float3 n_G = gaussian_normals[g_idx];
    float3 d_ray = ray_dirs[pixel_idx];
    
    // θ = (p_G - t_cam) · n_G / (d_ray · n_G)
    float3 diff = p_G - camera_pos;
    float numerator = dot3(diff, n_G);
    float denominator = dot3(d_ray, n_G);
    
    if (fabsf(denominator) < 1e-6f || numerator / denominator <= 0.0f) {
        depth_out[pixel_idx] = 0.0f;
        valid_out[pixel_idx] = 0;
        return;
    }
    
    float theta = numerator / denominator;
    
    // p_hit = t_cam + θ · d_ray
    float3 p_hit = make_float3(
        camera_pos.x + theta * d_ray.x,
        camera_pos.y + theta * d_ray.y,
        camera_pos.z + theta * d_ray.z
    );
    
    // D(u) = ||p_hit - t_cam||₂
    float3 diff_hit = p_hit - camera_pos;
    depth_out[pixel_idx] = length3(diff_hit);
    valid_out[pixel_idx] = 1;
}
