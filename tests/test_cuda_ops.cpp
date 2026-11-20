#include <gtest/gtest.h>

#include <cmath>
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

/**
 * @brief GPU residual add (in place) must match the CPU operator+=.
 *
 * 1000 elements: not a multiple of 256, so the last block has idle threads
 * (checks the i < n guard), same as the gelu test.
 * a and b use different slopes so a + b is non-zero and varies per element:
 * a kernel that wrote 0, or only copied a or b, would fail.
 * Tolerance 1e-5: a single float add per element, so the results should be
 * (almost) bit-identical.
 */
TEST(CudaOps, ResidualMatchesCpu)
{
    const int n = 1000;
    std::vector<float> values_1(n);
    std::vector<float> values_2(n);
    for (int i = 0; i < n; i++)
    {
        values_1[i] = -5.0f + 10.0f * i / (n - 1); // -5 .. 5
        values_2[i] = 2.0f - 3.0f * i / (n - 1);   //  2 .. -1
    }
    Tensor cpu_1({1, n}, values_1);
    const Tensor cpu_2({1, n}, values_2);

    // upload before the CPU add, so the GPU starts from the original values
    CudaTensor gpu_1(cpu_1);
    const CudaTensor gpu_2(cpu_2);

    cpu_1 += cpu_2; // expected answer, computed on the CPU

    residual(gpu_1, gpu_2); // gpu_1 is overwritten in place

    Tensor actual = gpu_1.to_host();

    ASSERT_EQ(cpu_1.shape(), actual.shape());

    for (int i = 0; i < n; i++)
    {
        EXPECT_NEAR(actual[i], cpu_1[i], 1e-5f) << "mismatch at index " << i;
    }
}

/**
 * @brief GPU layernorm must match the CPU layernorm, which already matches PyTorch.
 *
 * Shape [4, 768] = GPT-2's "Hello, world!" case: 4 rows -> 4 blocks,
 * 768 = 3 x 256 so every thread strides 3 times.
 * w and b vary per column, so using the wrong index for them (e.g. tid
 * instead of the column) would show up past column 255.
 * Tolerance 1e-4: the GPU sums in a different order (tree) than the CPU
 * (left to right), so float rounding differs slightly.
 */
TEST(CudaOps, LayerNormMatchesCpu)
{
    const int T = 4;
    const int d = 768;

    std::vector<float> x(T * d);
    for (int i = 0; i < T * d; i++)
    {
        // deterministic, non-trivial values; each row has a different mean/scale
        x[i] = std::sin(0.01f * i) * (1.0f + i / d) + 0.1f * (i / d);
    }
    std::vector<float> wv(d);
    std::vector<float> bv(d);
    for (int j = 0; j < d; j++)
    {
        wv[j] = 0.5f + 0.001f * j;
        bv[j] = -0.2f + 0.0005f * j;
    }

    const Tensor cpu_x({T, d}, x);
    const Tensor cpu_w({d}, wv);
    const Tensor cpu_b({d}, bv);

    const Tensor expected = layernorm(cpu_x, cpu_w, cpu_b);

    const CudaTensor gpu_x(cpu_x);
    const CudaTensor gpu_w(cpu_w);
    const CudaTensor gpu_b(cpu_b);

    const Tensor actual = layernorm(gpu_x, gpu_w, gpu_b).to_host();

    ASSERT_EQ(expected.shape(), actual.shape());
    for (int i = 0; i < T * d; i++)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-4f) << "mismatch at index " << i;
    }
}
