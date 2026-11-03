#pragma once
#include "tensor.h"
#include <vector>

Tensor relu(const Tensor &input);
Tensor gelu(const Tensor &input);

Tensor linear(const Tensor &input, const Tensor &weights, const Tensor &bias);
Tensor softmax(const Tensor &input);
Tensor embedding(const std::vector<int> &ids, const Tensor &wte, const Tensor &wpe);
Tensor layernorm(const Tensor &input, const Tensor &weight, const Tensor &bias, const float epsilon = 1e-5);
Tensor mlp(const Tensor &input, const Tensor &fc_weight, const Tensor &fc_bias, const Tensor &proj_weight, const Tensor &proj_bias);
