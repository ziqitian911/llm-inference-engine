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

/**
 * @brief In-place residual add on the GPU: a[i] += b[i].
 * @param a GPU tensor, overwritten with a + b (the residual stream).
 * @param b GPU tensor with the same shape as a; read only.
 * @throws std::invalid_argument if the shapes differ.
 */
__host__ void residual(CudaTensor &a, const CudaTensor &b);

/**
 * @brief LayerNorm over the last dimension on the GPU (one block per row).
 * @param input [T, d] GPU tensor.
 * @param weight [d] gamma.
 * @param bias [d] beta.
 * @param epsilon added to the variance before sqrt.
 * @return new [T, d] GPU tensor; input is left unchanged (the residual stream still needs it).
 * @throws std::invalid_argument if weight or bias size != d.
 */
__host__ CudaTensor layernorm(const CudaTensor &input,
                              const CudaTensor &weight,
                              const CudaTensor &bias,
                              const float epsilon = 1e-5f);
