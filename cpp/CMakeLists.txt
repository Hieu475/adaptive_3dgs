cmake_minimum_required(VERSION 3.18)
project(Adaptive3DGS_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

file(GLOB CPP_SOURCES "*.cpp")

add_library(adaptive_3dgs_cpp STATIC ${CPP_SOURCES})
target_include_directories(adaptive_3dgs_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adaptive_3dgs_cpp PUBLIC adaptive_3dgs_cuda)
