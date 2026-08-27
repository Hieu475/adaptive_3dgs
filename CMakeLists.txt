cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Adaptive3DGS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(cuda)
add_subdirectory(cpp)

# TODO: Add pybind11 integration to create a python module
