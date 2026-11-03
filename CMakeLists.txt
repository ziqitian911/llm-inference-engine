# minimum version required
cmake_minimum_required(VERSION 3.20)

# project name and language (CUDA is optional, enabled below if available)
project(mini_inference_engine CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main 
    src/main.cpp 
    src/tensor.cpp 
    src/gemm.cpp 
    src/ops.cpp 
    src/loader.cpp
    src/threadpool.cpp)
    
add_executable(bench 
    src/bench.cpp 
    src/tensor.cpp 
    src/gemm.cpp
    src/threadpool.cpp)

add_executable(bench_lmhead
    src/bench_lmhead.cpp
    src/tensor.cpp
    src/gemm.cpp
    src/threadpool.cpp)

target_include_directories(main PRIVATE include)
target_include_directories(bench PRIVATE include)
target_include_directories(bench_lmhead PRIVATE include)

# Optional CUDA backend: build GPU kernels + cuBLAS only if a CUDA compiler exists.
option(ENABLE_CUDA "Build the CUDA backend if a CUDA compiler is available" ON)
include(CheckLanguage)
check_language(CUDA)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 120)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    foreach(target main bench)
        target_sources(${target} PRIVATE src/cuda/gemm.cu)
        set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES 120)
        target_link_libraries(${target} PRIVATE CUDA::cublas)
        target_compile_definitions(${target} PRIVATE USE_CUDA)
    endforeach()

    message(STATUS "CUDA found: building GPU backend")
else()
    message(STATUS "CUDA backend disabled or not found: CPU-only build")
endif()