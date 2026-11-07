#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "tensor.h"

void block(
    Tensor &input,
    const std::unordered_map<std::string, Tensor> &weight_map,
    const int layer);

Tensor forward(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map);

Tensor lm_head(const Tensor &input, const Tensor &wte);

std::vector<int> generate(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map,
    int n_tokens = 20);