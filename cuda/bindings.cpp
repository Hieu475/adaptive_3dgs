/**
 * @file bindings.cpp
 * @brief PyTorch C++ bindings for CUDA kernels.
 * 
 * Exposes CUDA kernels to Python via pybind11/torch.
 */

#include <torch/extension.h>

// Forward declarations of CUDA kernel wrapper functions
torch::Tensor preprocess_gaussians_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix
);

torch::Tensor rasterize_forward_cuda(
    int width, int height,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths
);

torch::Tensor radix_sort_cuda(
    const torch::Tensor& keys,
    const torch::Tensor& values
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Adaptive 3DGS CUDA kernels";
    
    m.def("preprocess_gaussians", &preprocess_gaussians_cuda,
          "Preprocess 3D Gaussians: project, compute 2D cov, cull",
          py::arg("positions"),
          py::arg("scales"),
          py::arg("rotations"),
          py::arg("view_matrix"),
          py::arg("proj_matrix"));
    
    m.def("rasterize_forward", &rasterize_forward_cuda,
          "Tile-based rasterization with alpha compositing",
          py::arg("width"),
          py::arg("height"),
          py::arg("means2d"),
          py::arg("conics"),
          py::arg("colors"),
          py::arg("opacities"),
          py::arg("depths"));
    
    m.def("radix_sort", &radix_sort_cuda,
          "GPU radix sort for depth ordering",
          py::arg("keys"),
          py::arg("values"));
}
