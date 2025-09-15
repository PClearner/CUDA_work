cmake_minimum_required(VERSION 3.16)
project(Cudawork VERSION 0.1.0 LANGUAGES C CXX CUDA)
# project(Cudawork VERSION 0.1.0 LANGUAGES CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
# set(CUDA_FOUND TRUE)
# set(CUDA_VERSION_STRING ${CMAKE_CUDA_COMPILER_VERSION})
SET(CMAKE_CUDA_ARCHITECTURES 120)
# cmake code
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(./op/gemm)
add_subdirectory(./op/reduce)
add_subdirectory(./op/softmax)
add_subdirectory(./op/relu)
add_subdirectory(./op/rmsnorm)
add_subdirectory(./op/mha)
# CMAKE_CUDA_ARCHITECTURES