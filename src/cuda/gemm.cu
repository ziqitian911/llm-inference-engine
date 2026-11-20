#include "gemm_cuda.h"
#include "cublas_handle.h"

#include <cublas_v2.h>

constexpr int TILE = 16;

__global__ void mtx_mul(float *a, float *b, float *c, const int M, const int N, const int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void blocked_mtx_mul(float *a, float *b, float *c, const int M, const int N, const int K)
{

    __shared__ float As[TILE][TILE], Bs[TILE][TILE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    float sum = 0;

    for (int t = 0; t < K / TILE; t++)
    {
        As[ty][tx] = a[row * K + (t * TILE + tx)];
        Bs[ty][tx] = b[(t * TILE + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N)
    {
        c[row * N + col] = sum;
    }
}

float cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K)
{

    float *d_a, *d_b, *d_c;
    float ms = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    cudaEventRecord(start);

    mtx_mul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ms;
}

float blocked_cuda_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K)
{

    float *d_a, *d_b, *d_c;
    float ms = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEventRecord(start);

    blocked_mtx_mul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, K);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ms;
}

float cublas_matmul(const float *a, const float *b, float *c, const int M, const int N, const int K)
{

    float *d_a, *d_b, *d_c;
    float ms = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    cudaEventRecord(start);

    cublasSgemm(
        get_cublas_handle(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        d_b,
        N,
        d_a,
        K,
        &beta,
        d_c,
        N);

    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ms;
}
