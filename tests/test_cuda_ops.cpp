#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_check.h"
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

/**
 * @brief Build a [rows, cols] CPU tensor with deterministic, non-repeating values.
 *
 * sin of a non-integer step never lines up with rows/cols, so a wrong stride
 * or swapped dimension gives clearly different numbers.
 */
static Tensor make_tensor(const int rows, const int cols, const float scale, const float phase)
{
    std::vector<float> v(rows * cols);
    for (int i = 0; i < rows * cols; i++)
    {
        v[i] = scale * std::sin(0.37f * i + phase);
    }
    return Tensor({rows, cols}, v);
}

/**
 * @brief Run GPU linear and CPU linear on the same data and compare every element.
 *
 * The CPU linear already matches PyTorch (GPT-2 logits), so it is the reference.
 * Tolerance 1e-3: each output is a dot product of K terms; cuBLAS adds them in
 * a different order (and with FMA), so float rounding differs a little more
 * than for layernorm. Outputs here are O(1), so 1e-3 still catches real bugs
 * (a wrong ld, a swapped M/N or a missing bias is off by far more).
 */
static void check_linear(const int M, const int K, const int N)
{
    const Tensor cpu_x = make_tensor(M, K, 1.0f, 0.0f);
    const Tensor cpu_w = make_tensor(K, N, 0.05f, 1.0f); // GPT-2-like small weights
    std::vector<float> bv(N);
    for (int j = 0; j < N; j++)
    {
        bv[j] = 0.5f - 0.001f * j; // different per column: catches a wrong bias index
    }
    const Tensor cpu_b({N}, bv);

    const Tensor expected = linear(cpu_x, cpu_w, cpu_b);

    const CudaTensor gpu_x(cpu_x);
    const CudaTensor gpu_w(cpu_w);
    const CudaTensor gpu_b(cpu_b);
    const Tensor actual = linear(gpu_x, gpu_w, gpu_b).to_host();

    ASSERT_EQ(expected.shape(), actual.shape());
    for (int i = 0; i < M * N; i++)
    {
        EXPECT_NEAR(actual[i], expected[i], 1e-3f) << "mismatch at index " << i;
    }
}

/**
 * @brief Prefill shape of attn.c_attn: [4, 768] x [768, 2304].
 *
 * Non-square on purpose: with M != N != K, any mix-up of M/N/K or of the
 * leading dimensions in the cuBLAS call breaks the result (a square test
 * would hide it, as happened in M3).
 */
TEST(CudaOps, LinearMatchesCpuPrefill)
{
    check_linear(4, 768, 2304);
}

/**
 * @brief Decode shape of mlp.c_proj: [1, 3072] x [3072, 768].
 *
 * M = 1 is the GEMV case every decode step hits; K = 3072 is the longest
 * dot product in GPT-2.
 */
TEST(CudaOps, LinearMatchesCpuDecode)
{
    check_linear(1, 3072, 768);
}

/**
 * @brief Wrong shapes must throw before anything is launched.
 */
TEST(CudaOps, LinearRejectsBadShapes)
{
    const CudaTensor x(make_tensor(4, 768, 1.0f, 0.0f));
    const CudaTensor w(make_tensor(768, 2304, 1.0f, 0.0f));
    const CudaTensor bad_w(make_tensor(3072, 768, 1.0f, 0.0f));
    const CudaTensor b(Tensor({2304}, std::vector<float>(2304, 0.0f)));
    const CudaTensor bad_b(Tensor({3072}, std::vector<float>(3072, 0.0f)));

    EXPECT_THROW(linear(x, bad_w, b), std::invalid_argument); // K mismatch
    EXPECT_THROW(linear(x, w, bad_b), std::invalid_argument); // bias size != N
}

/**
 * @brief Copy token ids into a GPU buffer that frees itself (unique_ptr + cudaFree).
 *
 * In the engine this buffer is allocated once and reused; a test just needs one.
 */
static std::unique_ptr<int, decltype(&cudaFree)> upload_ids(const std::vector<int> &ids)
{
    int *raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, ids.size() * sizeof(int)));
    std::unique_ptr<int, decltype(&cudaFree)> d_ids(raw, &cudaFree);
    CUDA_CHECK(cudaMemcpy(raw, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    return d_ids;
}

/**
 * @brief Run GPU and CPU embedding on the same small tables and compare.
 *
 * Small tables (vocab 50, context 16) keep the test fast; d = 768 is GPT-2's width.
 * Exact match (EXPECT_FLOAT_EQ, not a tolerance): each output is a single float add
 * of the same two numbers on CPU and GPU, so there is no rounding difference.
 */
static void check_embedding(const std::vector<int> &ids, const int start_pos)
{
    const int vocab = 50;
    const int context = 16;
    const int d = 768;

    std::vector<float> te(vocab * d);
    std::vector<float> pe(context * d);
    for (int i = 0; i < vocab * d; i++)
    {
        te[i] = std::sin(0.13f * i); // every row different: a wrong row index shows up
    }
    for (int i = 0; i < context * d; i++)
    {
        pe[i] = 0.1f * std::cos(0.29f * i);
    }
    const Tensor cpu_wte({vocab, d}, te);
    const Tensor cpu_wpe({context, d}, pe);

    const Tensor expected = embedding(ids, cpu_wte, cpu_wpe, start_pos);

    const CudaTensor gpu_wte(cpu_wte);
    const CudaTensor gpu_wpe(cpu_wpe);
    const auto d_ids = upload_ids(ids);
    const int T = static_cast<int>(ids.size());

    const Tensor actual = embedding(d_ids.get(), T, gpu_wte, gpu_wpe, start_pos).to_host();

    ASSERT_EQ(expected.shape(), actual.shape());
    for (int i = 0; i < T * d; i++)
    {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "mismatch at index " << i;
    }
}

/**
 * @brief Prefill: 5 tokens from position 0, ids out of order and repeated.
 */
TEST(CudaOps, EmbeddingMatchesCpuPrefill)
{
    check_embedding({7, 3, 42, 0, 7}, 0);
}

/**
 * @brief Decode: one token at position 9.
 *
 * start_pos != 0 is the point: passing 0 to the kernel instead of start_pos
 * would read wpe row 0 and fail here (prefill alone would not notice).
 */
TEST(CudaOps, EmbeddingMatchesCpuDecode)
{
    check_embedding({42}, 9);
}

/**
 * @brief Bad tables or positions past the context must throw before launching.
 */
TEST(CudaOps, EmbeddingRejectsBadInputs)
{
    const int d = 768;
    const CudaTensor wte(Tensor({50, d}, std::vector<float>(50 * d, 0.0f)));
    const CudaTensor wpe(Tensor({16, d}, std::vector<float>(16 * d, 0.0f)));
    const CudaTensor narrow_wpe(Tensor({16, 64}, std::vector<float>(16 * 64, 0.0f)));
    const auto d_ids = upload_ids({1, 2, 3, 4});

    EXPECT_THROW(embedding(d_ids.get(), 4, wte, narrow_wpe), std::invalid_argument); // widths differ
    EXPECT_THROW(embedding(d_ids.get(), 4, wte, wpe, 13), std::invalid_argument);    // 13 + 4 > 16
}
