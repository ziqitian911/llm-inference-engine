#include "cuda_tensor.h"
#include "cuda_check.h"

CudaTensor::CudaTensor(const std::vector<int> &shape)
    : data_(nullptr, &cudaFree), shape_(shape) // start empty, but already knows its deleter
{
    // element count = product of all dimensions
    count_ = 1;
    for (int n : shape_)
    {
        count_ *= n;
    }

    /*
    cudaMalloc writes the GPU address into a plain pointer (C API),
    so catch it first, then hand ownership to unique_ptr
    */
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, count_ * sizeof(float)));
    data_.reset(d_data);
}

CudaTensor::CudaTensor(const Tensor &tensor)
    : CudaTensor(tensor.shape())
{
    CUDA_CHECK(cudaMemcpy(data_.get(), tensor.begin(), count_ * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor CudaTensor::to_host() const
{
    Tensor out(shape_);

    CUDA_CHECK(cudaMemcpy(out.begin(), data_.get(), count_ * sizeof(float), cudaMemcpyDeviceToHost));

    return out;
}
