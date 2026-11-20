#include "cuda_ops.h"
#include "cuda_tensor.h"
#include "cuda_check.h"

#include <cmath>
#include <numbers>
#include <stdexcept>

/**
 * @brief One thread per element: x[i] = gelu(x[i]).
 * @param data device pointer, read and overwritten in place.
 * @param n number of elements; extra threads in the last block do nothing.
 */
__global__ void gelu_kernel(float *data, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const float k = std::sqrt(2.0f / std::numbers::pi_v<float>);

    if (i < n)
    {
        const float x = data[i];
        data[i] = 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }
}

/**
 * @brief One thread per element: a[i] += b[i].
 * @param a device pointer, read and overwritten in place.
 * @param b device pointer, read only.
 * @param n number of elements; extra threads in the last block do nothing.
 */
__global__ void residual_kernel(float *a, const float *b, const int n)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        a[i] += b[i];
    }
}

__host__ void gelu(CudaTensor &tensor)
{
    float *data = tensor.data();
    const int size = tensor.size();

    // 1D launch: enough 256-thread blocks to cover all elements (round up)
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(data, size);

    /*
    launch is asynchronous: this only catches launch errors (bad config etc.).
    no sync here; a later cudaMemcpy waits for the kernel automatically.
    */
    CUDA_CHECK(cudaGetLastError());
}

__host__ void residual(CudaTensor &a, const CudaTensor &b)
{
    // check dimension
    if (a.shape() != b.shape())
    {
        throw std::invalid_argument("residual: shapes must match");
    }

    float *d_a = a.data();
    const float *d_b = b.data();
    const int size = a.size();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    residual_kernel<<<blocks, threads>>>(d_a, d_b, size);

    CUDA_CHECK(cudaGetLastError());
}
