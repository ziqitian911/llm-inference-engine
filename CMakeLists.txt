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
    src/threadpool.cpp
    src/gpt2.cpp
    src/sampling.cpp)
    
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

# Unit tests: download GoogleTest at configure time
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Optional CUDA backend: build GPU kernels + cuBLAS only if a CUDA compiler exists.
option(ENABLE_CUDA "Build the CUDA backend if a CUDA compiler is available" ON)
include(CheckLanguage)
check_language(CUDA)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 120)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    foreach(target main bench)
        target_sources(${target} PRIVATE src/cuda/gemm.cu src/cuda/cuda_tensor.cu)
        set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES 120)
        target_link_libraries(${target} PRIVATE CUDA::cublas)
        target_compile_definitions(${target} PRIVATE USE_CUDA)
    endforeach()

    # CUDA unit tests (only built when CUDA is available)
    add_executable(test_cuda_tensor
        tests/test_cuda_tensor.cpp
        src/tensor.cpp
        src/cuda/cuda_tensor.cu)
    target_include_directories(test_cuda_tensor PRIVATE include)
    set_target_properties(test_cuda_tensor PROPERTIES CUDA_ARCHITECTURES 120)
    target_link_libraries(test_cuda_tensor PRIVATE GTest::gtest_main CUDA::cudart)
    gtest_discover_tests(test_cuda_tensor)


    message(STATUS "CUDA found: building GPU backend")
else()
    message(STATUS "CUDA backend disabled or not found: CPU-only build")
endif()