#include <gtest/gtest.h>

#include <vector>

#include "cuda_ops.h"
#include "cuda_tensor.h"
#include "ops.h"
#include "tensor.h"

/**
 * @brief GPU gelu (in place) must match the CPU gelu, which already matches PyTorch.
 *
 * 1000 elements: not a multiple of 256, so the last block has idle threads
 * (checks the i < n guard), and it needs 4 blocks (checks the index formula).
 */
TEST(CudaOps, GeluMatchesCpu)
{
    // inputs from -5 to 5: negatives, zero region, positives
    const int n = 1000;
    std::vector<float> values(n);
    for (int i = 0; i < n; i++)
    {
        values[i] = -5.0f + 10.0f * i / (n - 1);
    }
    Tensor cpu({1, n}, values);

    // reference answer: CPU version
    Tensor expected = gelu(cpu);

    // GPU version: upload, gelu in place, download
    CudaTensor gpu(cpu);
    gelu(gpu);
    Tensor actual = gpu.to_host();

    ASSERT_EQ(actual.shape(), expected.shape());

    // GPU tanh is not bit-identical to the CPU one, so allow a tiny error
    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "mismatch at index " << i;
    }
}

TEST(CudaOps, ResidualMatchesCpu)
{
    const int n = 1000;
    std::vector<float> values_1(n);
    std::vector<float> values_2(n);
    for (int i = 0; i < n; i++)
    {
        values_1[i] = -5.0f + 10.0f * i / (n - 1);
        values_2[i] = 5.0f - 10.0f * i / (n - 1);
    }
    Tensor cpu_1({1, n}, values_1);
    const Tensor cpu_2({1, n}, values_2);

    CudaTensor gpu_1(cpu_1);
    const CudaTensor gpu_2(cpu_2);

    cpu_1 += cpu_2;

    residual(gpu_1, gpu_2);

    Tensor actual = gpu_1.to_host();

    ASSERT_EQ(cpu_1.shape(), actual.shape());

    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(actual[i], cpu_1[i], 1e-5f) << "mismatch at index " << i;
    }
}
