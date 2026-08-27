/**
 * @file binning.cu
 * @brief CUDA kernels for tile binning.
 * 
 * Assigns each Gaussian to the set of 16x16 tiles it overlaps.
 * Uses atomic operations for thread-safe counting.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 16


__global__ void count_tiles_per_gaussian(
    int N,
    const float* __restrict__ means2d,   // (N, 2)
    const int* __restrict__ radii,       // (N,)
    const int* __restrict__ valid,       // (N,)
    int image_width,
    int image_height,
    int* __restrict__ tile_counts        // (N,) output: number of tiles per Gaussian
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !valid[idx]) {
        if (idx < N) tile_counts[idx] = 0;
        return;
    }
    
    float cx = means2d[idx * 2];
    float cy = means2d[idx * 2 + 1];
    int r = radii[idx];
    
    int n_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
    int n_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
    
    int tx_min = max(0, (int)((cx - r) / TILE_SIZE));
    int tx_max = min(n_tiles_x - 1, (int)((cx + r) / TILE_SIZE));
    int ty_min = max(0, (int)((cy - r) / TILE_SIZE));
    int ty_max = min(n_tiles_y - 1, (int)((cy + r) / TILE_SIZE));
    
    tile_counts[idx] = (tx_max - tx_min + 1) * (ty_max - ty_min + 1);
}


__global__ void assign_gaussians_to_tiles(
    int N,
    const float* __restrict__ means2d,
    const int* __restrict__ radii,
    const int* __restrict__ valid,
    const int* __restrict__ offsets,       // (N,) prefix sum of tile_counts
    int image_width,
    int image_height,
    int* __restrict__ tile_gaussian_ids,   // output: (total_pairs,) Gaussian index
    int* __restrict__ tile_ids_out         // output: (total_pairs,) tile index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || !valid[idx]) return;
    
    float cx = means2d[idx * 2];
    float cy = means2d[idx * 2 + 1];
    int r = radii[idx];
    
    int n_tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
    int n_tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
    
    int tx_min = max(0, (int)((cx - r) / TILE_SIZE));
    int tx_max = min(n_tiles_x - 1, (int)((cx + r) / TILE_SIZE));
    int ty_min = max(0, (int)((cy - r) / TILE_SIZE));
    int ty_max = min(n_tiles_y - 1, (int)((cy + r) / TILE_SIZE));
    
    int offset = offsets[idx];
    int count = 0;
    for (int ty = ty_min; ty <= ty_max; ty++) {
        for (int tx = tx_min; tx <= tx_max; tx++) {
            int tile_id = ty * n_tiles_x + tx;
            tile_gaussian_ids[offset + count] = idx;
            tile_ids_out[offset + count] = tile_id;
            count++;
        }
    }
}
