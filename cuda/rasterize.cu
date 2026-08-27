/**
 * @file rasterize.cu
 * @brief CUDA tile-based rasterization with alpha compositing and early termination.
 * 
 * Each CUDA block handles one 16x16 tile.
 * Shared memory is used to stage batches of Gaussians for reuse across pixels.
 * 
 * Early termination: T < epsilon (1e-4) stops processing for saturated pixels.
 * 
 * Performance considerations:
 * - SoA memory layout for coalesced reads
 * - Shared memory bank conflict avoidance: 16x16 tile = 256 threads
 * - Warp-level synchronization for early termination
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)  // 256 threads
#define GAUSSIAN_BATCH 32  // Gaussians loaded into shared memory per iteration


__global__ void rasterize_color_kernel(
    int width,
    int height,
    int n_tiles_x,
    const int* __restrict__ tile_offsets,    // (n_tiles + 1,) prefix sum
    const int* __restrict__ sorted_indices,  // sorted Gaussian indices per tile
    const float* __restrict__ means2d,       // (N, 2)
    const float* __restrict__ conics,        // (N, 3) (a, b, c)
    const float* __restrict__ colors,        // (N, 3)
    const float* __restrict__ opacities,     // (N,)
    const float* __restrict__ depths,        // (N,)
    float* __restrict__ out_color,           // (H, W, 3)
    float* __restrict__ out_depth,           // (H, W)
    float* __restrict__ out_transmission     // (H, W)
) {
    // Tile coordinates
    int tile_id = blockIdx.x;
    int tile_y = tile_id / n_tiles_x;
    int tile_x = tile_id % n_tiles_x;
    
    // Pixel within tile
    int local_id = threadIdx.x;
    int local_y = local_id / TILE_SIZE;
    int local_x = local_id % TILE_SIZE;
    
    // Global pixel coordinates
    int px = tile_x * TILE_SIZE + local_x;
    int py = tile_y * TILE_SIZE + local_y;
    
    bool inside = (px < width) && (py < height);
    float pixel_x = (float)px + 0.5f;
    float pixel_y = (float)py + 0.5f;
    
    // Per-pixel state
    float T = 1.0f;  // Transmittance
    float rendered_r = 0.0f, rendered_g = 0.0f, rendered_b = 0.0f;
    float rendered_depth = 0.0f;
    bool done = !inside;
    
    // Shared memory for batched Gaussian loading
    __shared__ float2 s_means2d[GAUSSIAN_BATCH];
    __shared__ float3 s_conics[GAUSSIAN_BATCH];
    __shared__ float3 s_colors[GAUSSIAN_BATCH];
    __shared__ float s_opacity[GAUSSIAN_BATCH];
    __shared__ float s_depth[GAUSSIAN_BATCH];
    
    // Range of Gaussians for this tile
    int start = tile_offsets[tile_id];
    int end = tile_offsets[tile_id + 1];
    int n_gaussians = end - start;
    
    // Process in batches
    for (int batch_start = 0; batch_start < n_gaussians; batch_start += GAUSSIAN_BATCH) {
        // Check if entire block is done (warp-level early termination)
        int block_done = __syncthreads_and(done ? 1 : 0);
        if (block_done) break;
        
        int batch_end = min(batch_start + GAUSSIAN_BATCH, n_gaussians);
        int batch_size = batch_end - batch_start;
        
        // Collaboratively load batch into shared memory
        if (local_id < batch_size) {
            int g_idx = sorted_indices[start + batch_start + local_id];
            s_means2d[local_id] = make_float2(means2d[g_idx*2], means2d[g_idx*2+1]);
            s_conics[local_id] = make_float3(conics[g_idx*3], conics[g_idx*3+1], conics[g_idx*3+2]);
            s_colors[local_id] = make_float3(colors[g_idx*3], colors[g_idx*3+1], colors[g_idx*3+2]);
            s_opacity[local_id] = opacities[g_idx];
            s_depth[local_id] = depths[g_idx];
        }
        __syncthreads();
        
        // Process each Gaussian in the batch
        if (!done) {
            for (int j = 0; j < batch_size; j++) {
                float dx = pixel_x - s_means2d[j].x;
                float dy = pixel_y - s_means2d[j].y;
                
                float a = s_conics[j].x;
                float b = s_conics[j].y;
                float c = s_conics[j].z;
                
                // Mahalanobis distance: -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
                float power = -0.5f * (a * dx*dx + 2.0f * b * dx*dy + c * dy*dy);
                
                if (power > 0.0f) continue;  // Outside Gaussian
                
                float alpha = fminf(0.99f, s_opacity[j] * expf(power));
                if (alpha < 1.0f / 255.0f) continue;  // Too transparent
                
                // Alpha compositing
                float weight = T * alpha;
                rendered_r += weight * s_colors[j].x;
                rendered_g += weight * s_colors[j].y;
                rendered_b += weight * s_colors[j].z;
                rendered_depth += weight * s_depth[j];
                
                T *= (1.0f - alpha);
                
                // Early termination
                if (T < 1e-4f) {
                    done = true;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (inside) {
        int pixel_idx = py * width + px;
        out_color[pixel_idx * 3 + 0] = rendered_r;
        out_color[pixel_idx * 3 + 1] = rendered_g;
        out_color[pixel_idx * 3 + 2] = rendered_b;
        out_depth[pixel_idx] = rendered_depth;
        out_transmission[pixel_idx] = T;
    }
}


torch::Tensor rasterize_forward_cuda(
    int width, int height,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths
) {
    // Placeholder: full tile assignment + sort would go here
    // For now, return empty output sized correctly
    auto options = torch::TensorOptions().device(means2d.device()).dtype(torch::kFloat32);
    return torch::zeros({height, width, 3}, options);
}
