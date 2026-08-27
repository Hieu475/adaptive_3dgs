/**
 * @file radix_sort.cu
 * @brief CUDA stub for depth sorting using CUB radix sort.
 */

#include <cuda_runtime.h>
// #include <cub/cub.cuh>

namespace adaptive3dgs {

/**
 * @brief Sorts key-value pairs using CUB.
 * TODO: Implement actual CUB invocation.
 */
void sort_gaussians(
    uint64_t* keys_in,
    uint64_t* keys_out,
    uint32_t* values_in,
    uint32_t* values_out,
    int num_elements
) {
    // TODO: Allocate temporary storage and call cub::DeviceRadixSort::SortPairs
}

} // namespace adaptive3dgs
