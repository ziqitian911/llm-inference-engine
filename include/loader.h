#pragma once
#include "tensor.h"
#include <string>
#include <unordered_map>

Tensor load_tensor(const std::string &path, const std::vector<int> &shape);
std::unordered_map<std::string, Tensor> load_model(const std::string &path);
