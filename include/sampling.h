#pragma once

#include "tensor.h"

#include <random>

Tensor apply_temperature(const Tensor &logits, const float temperature = 1.0f);
Tensor apply_top_p(const Tensor &probs, const float p = 0.5);
Tensor apply_top_k(const Tensor &logits, const int k = 5);
int sample(const Tensor &probs, std::mt19937 &rng);