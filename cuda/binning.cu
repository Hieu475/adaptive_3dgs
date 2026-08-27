/**
 * @file binning.cu
 * @brief CUDA kernel stubs for tile binning (mapping Gaussians to image tiles).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @brief Kernel to compute overlapping tiles for each Gaussian.
 * TODO: Implement tile intersection logic.
 */
__global__ void compute_keys_kernel(
    int num_gaussians,
    const float2* means2d,
    const int* radii,
    int2 grid,
    uint64_t* out_keys,
    uint32_t* out_values,
    int* out_num_keys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    // TODO: Determine which tiles the Gaussian covers and emit key-value pairs
}
