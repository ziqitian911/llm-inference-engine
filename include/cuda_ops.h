#pragma once

#include "cuda_tensor.h"

/**
 * @brief GELU (tanh approximation) on the GPU, in place.
 *
 * Overloads the CPU gelu(const Tensor&): same formula, but the data stays
 * in GPU memory and is overwritten (no extra allocation).
 * @param tensor GPU tensor of any shape; its values are replaced by gelu(x).
 */
__host__ void gelu(CudaTensor &tensor);
