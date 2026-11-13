#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "tensor.h"

/**
 * @brief A float tensor whose data lives in GPU memory
 *
 * Owns its memory through unique_ptr + cudaFree (Rule of Zero):
 * no destructor / copy / move written by hand.
 * Copy is deleted (unique_ptr can't be copied), move is allowed.
 * The CPU must never dereference data_: it is a GPU address.
 */
class CudaTensor
{
private:
    std::unique_ptr<float, decltype(&cudaFree)> data_; // GPU address, freed by cudaFree
    std::vector<int> shape_;
    int count_; // element count

public:
    /**
     * @brief Allocate GPU memory for the given shape, contents uninitialized.
     * @param shape e.g. {1024, 768}
     */
    explicit CudaTensor(const std::vector<int> &shape);

    /**
     * @brief Upload a CPU tensor to GPU memory (same shape, data copied).
     * @param tensor the CPU tensor to copy from.
     */
    explicit CudaTensor(const Tensor &tensor);

    /**
     * @brief Raw GPU address of the data, for passing to kernels / cudaMemcpy.
     * @return writable device pointer. Never dereference it on the CPU.
     */
    float *data()
    {
        return data_.get();
    }

    /**
     * @brief Read-only GPU address of the data (for const tensors).
     * @return read-only device pointer. Never dereference it on the CPU.
     */
    const float *data() const
    {
        return data_.get();
    }

    /**
     * @brief Number of elements in the tensor.
     * @return product of all dimensions.
     */
    int size() const
    {
        return count_;
    }

    Tensor to_host() const;
};
