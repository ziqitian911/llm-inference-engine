#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * @brief Warp a CUDA call; throw with file:line and the error message if it fails.
 *
 * A marco (not a function) so __FILE__ / __LINE__ point at the caller.
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, bytes));
 */
#define CUDA_CHECK(call)                                               \
    do                                                                 \
    {                                                                  \
        cudaError_t err_ = (call);                                     \
        if (err_ != cudaSuccess)                                       \
        {                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" +     \
                                     std::to_string(__LINE__) + ": " + \
                                     cudaGetErrorString(err_));        \
        }                                                              \
    } while (0)
