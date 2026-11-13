#include <gtest/gtest.h>

#include "cuda_tensor.h"
#include "tensor.h"

/**
 * @brief Upload a CPU tensor to the GPU, download it back, and check nothing changed.
 *
 * This is pure data movement (no math), so every value must match exactly.
 */
TEST(CudaTensor, UploadDownloadRoundTrip)
{
    // 1. a small CPU tensor with known values
    Tensor cpu({2, 3}, {1, 2, 3, 4, 5, 6});

    // 2. upload (constructor is explicit, so call it by name)
    CudaTensor gpu(cpu);

    // 3. download into a new CPU tensor
    Tensor back = gpu.to_host();

    // 4. shape first: if it is wrong, the loop below would read out of bounds, so stop here
    ASSERT_EQ(back.shape(), cpu.shape());

    for (int i = 0; i < cpu.size(); i++)
    {
        EXPECT_EQ(back[i], cpu[i]) << "mismatch at flat index " << i;
    }
}
