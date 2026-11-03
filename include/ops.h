#pragma once
#include "tensor.h"
#include <vector>

Tensor relu(const Tensor &input);
Tensor linear(const Tensor &input, const Tensor &weights, const Tensor &bias);
Tensor softmax(const Tensor &input);
Tensor embedding(const std::vector<int> &ids, const Tensor &wte, const Tensor &wpe);