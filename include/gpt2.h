#pragma once

#include <unordered_map>
#include <string>

#include "tensor.h"

void block(
    Tensor &input,
    const std::unordered_map<std::string, Tensor> &weight_map,
    const int layer);