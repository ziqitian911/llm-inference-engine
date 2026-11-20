#include "cuda_ops.h"
#include "cuda_tensor.h"
#include "cuda_check.h"
#include "cublas_handle.h"

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

// threads per block for layernorm; must be a power of 2 (tree reduction halves it)
constexpr int LN_THREADS = 256;

/**
 * @brief One block per row: normalize row blockIdx.x, then scale by w and shift by b.
 *
 * Threads stride over the row (coalesced), each keeps a partial sum in a register,
 * then the block combines the partials in shared memory with a tree reduction.
 * Done twice: once for the mean, once for the variance.
 * @param in [T, d] input, device pointer.
 * @param out [T, d] output, device pointer.
 * @param w [d] gamma, shared by all rows.
 * @param b [d] beta, shared by all rows.
 * @param d row length (768 for GPT-2 small).
 * @param epsilon added to var before sqrt.
 */
__global__ void layernorm_kernel(
    const float *in,
    float *out,
    const float *w,
    const float *b,
    const int d,
    const float epsilon)
{
    // this block handles row blockIdx.x; tid = my seat in the block
    __shared__ float partial[LN_THREADS]; // one slot per thread, visible to the whole block
    const int tid = threadIdx.x;

    // ---- 1. mean ----
    // each thread sums its own elements: tid, tid + 256, tid + 512, ...
    // (neighbouring threads read neighbouring addresses -> coalesced)
    // sum is a register: private to this thread, so no race
    float sum = 0.0f;
    for (int i = tid; i < d; i += blockDim.x)
    {
        sum += in[blockIdx.x * d + i];
    }

    // hand my partial sum in, then wait until every thread has done so
    partial[tid] = sum;
    __syncthreads();

    // tree reduction: each round the first half adds the second half,
    // 256 -> 128 -> ... -> 1, so the total ends up in partial[0] after log2(256) = 8 rounds.
    // in one round every active thread writes a different slot -> no lock needed
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        // outside the if: every thread must reach it, or the block hangs.
        // next round reads what this round wrote, so wait for everyone
        __syncthreads();
    }

    const float mean = partial[0] / d;
    sum = 0.0f;
    // partial is reused below and partial[0] gets overwritten:
    // wait until every thread has read the mean first
    __syncthreads();

    // ---- 2. variance: same two steps, only what we add changes ----
    for (int i = tid; i < d; i += blockDim.x)
    {
        float x = in[blockIdx.x * d + i];
        sum += (x - mean) * (x - mean);
    }
    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    // safe to read: the last round above already synced, and nobody writes partial again
    const float var = partial[0] / d;

    // computed once per thread, not once per element
    const float std_dev = std::sqrt(var + epsilon);

    // ---- 3. normalize, then scale and shift ----
    // same strided split; every output has exactly one writer, so no sync needed.
    // in / out are [T, d] -> index includes the row;
    // w / b are [d], shared by all rows -> index is the column i only
    for (int i = tid; i < d; i += blockDim.x)
    {
        float x = in[blockIdx.x * d + i];
        out[blockIdx.x * d + i] = ((x - mean) / std_dev) * w[i] + b[i];
    }
}

/**
 * @brief One thread per element: out[i] += bias[column of i] (broadcast over rows).
 * @param a [M, N] output, device pointer, updated in place.
 * @param b [N] bias, device pointer; every row reuses the same N values.
 * @param n total elements, M * N.
 * @param N row length; i % N is the column of element i.
 */
__global__ void add_bias_kernel(float *a, const float *b, const int n, const int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        a[i] += b[i % N];
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

__host__ CudaTensor layernorm(
    const CudaTensor &input,
    const CudaTensor &weight,
    const CudaTensor &bias,
    const float epsilon)
{
    const int T = input.dim(0);
    const int d = input.dim(1);

    if (weight.size() != d || bias.size() != d)
    {
        throw std::invalid_argument("layernorm: weight and bias must have size d");
    }

    CudaTensor out({T, d});

    // one block per row, LN_THREADS threads cooperate on that row
    layernorm_kernel<<<T, LN_THREADS>>>(
        input.data(), out.data(), weight.data(), bias.data(), d, epsilon);

    CUDA_CHECK(cudaGetLastError());

    return out;
}

__host__ CudaTensor linear(
    const CudaTensor &input,
    const CudaTensor &weights,
    const CudaTensor &bias)
{

    // check dim
    if (input.dim(1) != weights.dim(0))
    {
        throw std::invalid_argument("linear: input cols must equal weights rows");
    }

    const int M = input.dim(0);
    const int K = input.dim(1);
    const int N = weights.dim(1);

    if (bias.size() != N)
    {
        throw std::invalid_argument("linear: bias size must equal weights cols.");
    }

    CudaTensor out({M, N});

    const float alpha = 1.0f, beta = 0.0f; // out = 1 * (input x weights) + 0 * out

    /*
    cuBLAS is column-major: our row-major [R, C] buffer looks like its transpose [C, R].
    So we ask it for out^T = weights^T x input^T, which it already "sees" for free:
    weights goes first, input second, M and N swapped.
    Each leading dimension = row length of that matrix in our row-major view.
    */
    cublasSgemm(
        get_cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        weights.data(), N,
        input.data(), K,
        &beta,
        out.data(), N);

    // same default stream as cuBLAS, so this runs after the multiply has finished
    const int threads = 256;
    const int blocks = (out.size() + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads>>>(out.data(), bias.data(), out.size(), N);
    CUDA_CHECK(cudaGetLastError());
    return out;
}
