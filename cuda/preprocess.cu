/**
 * @file preprocess.cu
 * @brief CUDA kernels for Gaussian preprocessing.
 * 
 * Operations:
 * 1. Build 3D covariance from scale + rotation: Σ = R·S·S^T·R^T
 * 2. Transform to camera space
 * 3. Project to 2D (Jacobian-based covariance projection)
 * 4. Frustum culling
 * 5. Compute screen-space conics and radii
 * 
 * Memory layout: SoA (Structure of Arrays) for coalesced access.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <cmath>


__device__ void quaternion_to_matrix(
    float w, float x, float y, float z,
    float* R  // 3x3 output, row-major
) {
    float norm = rsqrtf(w*w + x*x + y*y + z*z + 1e-8f);
    w *= norm; x *= norm; y *= norm; z *= norm;
    
    R[0] = 1.f - 2.f*(y*y + z*z);  R[1] = 2.f*(x*y - w*z);        R[2] = 2.f*(x*z + w*y);
    R[3] = 2.f*(x*y + w*z);        R[4] = 1.f - 2.f*(x*x + z*z);  R[5] = 2.f*(y*z - w*x);
    R[6] = 2.f*(x*z - w*y);        R[7] = 2.f*(y*z + w*x);        R[8] = 1.f - 2.f*(x*x + y*y);
}


__device__ void compute_cov3d(
    const float* scale,  // 3 elements (already exp'd)
    const float* R,      // 3x3 rotation matrix
    float* cov           // 6 elements (upper triangle of symmetric 3x3)
) {
    // M = R @ S where S = diag(scale)
    // M[i][j] = R[i][j] * scale[j]
    float M[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M[i*3+j] = R[i*3+j] * scale[j];
        }
    }
    
    // Σ = M @ M^T, store upper triangle
    // cov[0] = Σ[0][0], cov[1] = Σ[0][1], cov[2] = Σ[0][2]
    // cov[3] = Σ[1][1], cov[4] = Σ[1][2], cov[5] = Σ[2][2]
    cov[0] = M[0]*M[0] + M[1]*M[1] + M[2]*M[2];
    cov[1] = M[0]*M[3] + M[1]*M[4] + M[2]*M[5];
    cov[2] = M[0]*M[6] + M[1]*M[7] + M[2]*M[8];
    cov[3] = M[3]*M[3] + M[4]*M[4] + M[5]*M[5];
    cov[4] = M[3]*M[6] + M[4]*M[7] + M[5]*M[8];
    cov[5] = M[6]*M[6] + M[7]*M[7] + M[8]*M[8];
}


__global__ void preprocess_gaussians_kernel(
    int N,
    const float* __restrict__ positions,    // (N, 3) SoA: x,y,z contiguous
    const float* __restrict__ log_scales,   // (N, 3)
    const float* __restrict__ rotations,    // (N, 4) quaternion w,x,y,z
    const float* __restrict__ view_matrix,  // (4, 4) world-to-camera
    const float* __restrict__ intrinsics,   // fx, fy, cx, cy
    float* __restrict__ means2d,            // (N, 2) output
    float* __restrict__ conics,             // (N, 3) output (a, b, c of inverse 2D cov)
    int* __restrict__ radii,                // (N,) output
    float* __restrict__ depths_out,         // (N,) output
    int* __restrict__ valid                 // (N,) output: 1 if visible, 0 otherwise
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Load position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];
    
    // Transform to camera space: p_cam = R_view @ p_world + t_view
    float cx = view_matrix[0]*px + view_matrix[1]*py + view_matrix[2]*pz + view_matrix[3];
    float cy = view_matrix[4]*px + view_matrix[5]*py + view_matrix[6]*pz + view_matrix[7];
    float cz = view_matrix[8]*px + view_matrix[9]*py + view_matrix[10]*pz + view_matrix[11];
    
    // Near-plane culling
    if (cz < 0.1f) {
        valid[idx] = 0;
        radii[idx] = 0;
        return;
    }
    
    depths_out[idx] = cz;
    
    // Perspective projection
    float fx = intrinsics[0], fy = intrinsics[1];
    float cx_intr = intrinsics[2], cy_intr = intrinsics[3];
    float inv_z = 1.0f / cz;
    
    float u = fx * cx * inv_z + cx_intr;
    float v = fy * cy * inv_z + cy_intr;
    means2d[idx * 2 + 0] = u;
    means2d[idx * 2 + 1] = v;
    
    // Build rotation matrix from quaternion
    float R[9];
    quaternion_to_matrix(
        rotations[idx*4+0], rotations[idx*4+1],
        rotations[idx*4+2], rotations[idx*4+3], R
    );
    
    // Compute 3D covariance
    float scale[3] = {
        expf(log_scales[idx*3+0]),
        expf(log_scales[idx*3+1]),
        expf(log_scales[idx*3+2]),
    };
    float cov3d[6];
    compute_cov3d(scale, R, cov3d);
    
    // Jacobian of perspective projection
    float inv_z2 = inv_z * inv_z;
    float J[6];  // 2x3 Jacobian
    J[0] = fx * inv_z;   J[1] = 0.0f;         J[2] = -fx * cx * inv_z2;
    J[3] = 0.0f;         J[4] = fy * inv_z;   J[5] = -fy * cy * inv_z2;
    
    // W = view_matrix[:3,:3] (rotation part)
    // T = J @ W -> 2x3
    float W[9];
    for (int i = 0; i < 9; i++) W[i] = view_matrix[i / 3 * 4 + i % 3];
    
    float T[6]; // 2x3
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            T[i*3+j] = 0;
            for (int k = 0; k < 3; k++) {
                T[i*3+j] += J[i*3+k] * W[k*3+j];
            }
        }
    }
    
    // 2D covariance: Σ2D = T @ Σ3D @ T^T (2x2 symmetric)
    // First compute T @ Σ3D (2x3 @ 3x3 -> 2x3, using symmetric storage)
    float cov3d_full[9] = {
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    };
    
    float TS[6]; // 2x3
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            TS[i*3+j] = 0;
            for (int k = 0; k < 3; k++) {
                TS[i*3+j] += T[i*3+k] * cov3d_full[k*3+j];
            }
        }
    }
    
    // Then TS @ T^T -> 2x2
    float cov2d_00 = TS[0]*T[0] + TS[1]*T[1] + TS[2]*T[2] + 0.3f;
    float cov2d_01 = TS[0]*T[3] + TS[1]*T[4] + TS[2]*T[5];
    float cov2d_11 = TS[3]*T[3] + TS[4]*T[4] + TS[5]*T[5] + 0.3f;
    
    // Compute conic (inverse of 2D covariance)
    float det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01;
    if (det < 1e-6f) {
        valid[idx] = 0;
        radii[idx] = 0;
        return;
    }
    float inv_det = 1.0f / det;
    conics[idx*3+0] = cov2d_11 * inv_det;   // a
    conics[idx*3+1] = -cov2d_01 * inv_det;  // b
    conics[idx*3+2] = cov2d_00 * inv_det;   // c
    
    // Compute radius from eigenvalues
    float mid = 0.5f * (cov2d_00 + cov2d_11);
    float disc = sqrtf(fmaxf(0.0f, (cov2d_00 - cov2d_11) * (cov2d_00 - cov2d_11) * 0.25f + cov2d_01 * cov2d_01));
    float lambda_max = mid + disc;
    int r = (int)ceilf(3.0f * sqrtf(lambda_max));
    radii[idx] = r;
    valid[idx] = 1;
}


// PyTorch wrapper
torch::Tensor preprocess_gaussians_cuda(
    const torch::Tensor& positions,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix
) {
    int N = positions.size(0);
    auto options = torch::TensorOptions().device(positions.device()).dtype(torch::kFloat32);
    
    auto means2d = torch::zeros({N, 2}, options);
    auto conics = torch::zeros({N, 3}, options);
    auto radii_out = torch::zeros({N}, options.dtype(torch::kInt32));
    auto depths_out = torch::zeros({N}, options);
    auto valid = torch::zeros({N}, options.dtype(torch::kInt32));
    
    float intrinsics_host[4] = {
        proj_matrix[0][0].item<float>(),
        proj_matrix[1][1].item<float>(),
        proj_matrix[0][2].item<float>(),
        proj_matrix[1][2].item<float>(),
    };
    auto intrinsics_d = torch::from_blob(intrinsics_host, {4}, torch::kFloat32).to(positions.device());
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    preprocess_gaussians_kernel<<<blocks, threads>>>(
        N,
        positions.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        rotations.contiguous().data_ptr<float>(),
        view_matrix.contiguous().data_ptr<float>(),
        intrinsics_d.data_ptr<float>(),
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        radii_out.data_ptr<int>(),
        depths_out.data_ptr<float>(),
        valid.data_ptr<int>()
    );
    
    return means2d;
}
