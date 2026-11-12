#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "tensor.h"

/**
 * @brief Keys and Values of every past token, for each layer.
 *
 * Preallocated to the full context length so new rows are written in place
 * instead of growing the tensors. Only the first len rows are valid.
 */
struct KVCache
{
    std::vector<Tensor> K;
    std::vector<Tensor> V;
    int len; // tokens cached so far, increased once after all layers have run

    explicit KVCache(int n_layer = 12, int d_model = 768, int context_length = 1024)
    {
        this->len = 0;
        this->K.reserve(n_layer);
        this->V.reserve(n_layer);

        for (int i = 0; i < n_layer; i++)
        {
            this->K.push_back(Tensor({context_length, d_model}));
            this->V.push_back(Tensor({context_length, d_model}));
        }
    }
};

void block(
    Tensor &input,
    const std::unordered_map<std::string, Tensor> &weight_map,
    const int layer,
    KVCache &cache);

Tensor forward(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map,
    KVCache &cache);

Tensor lm_head(const Tensor &input, const Tensor &wte);

// temperature == 0 means greedy decoding (argmax); otherwise sample
std::vector<int> generate(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map,
    int n_tokens = 20,
    float temperature = 0.0f,
    int top_k = 50,
    float top_p = 1.0f);