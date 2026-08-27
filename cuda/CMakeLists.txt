cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Adaptive3DGS_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB CUDA_SOURCES "*.cu")

add_library(adaptive_3dgs_cuda STATIC ${CUDA_SOURCES})
set_target_properties(adaptive_3dgs_cuda PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")

target_include_directories(adaptive_3dgs_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
