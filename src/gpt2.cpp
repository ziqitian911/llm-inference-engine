/**
 * @file gpt2.cpp
 * @brief GPT-2 model structure, built from the operators in ops.h
 */

#include "gpt2.h"
#include "ops.h"

#include <unordered_map>
#include <string>
#include <algorithm>

constexpr int n_layer = 12;

/**
 * @brief Run one transformer block, updating the residual stream in place.
 *
 * @param input residual stream of shape [T, d_model], modified in place
 * @param weight_map all model weights, keyed by tensor name
 * @param layer index of the block, from 0 to 11 for GPT-2 small
 * @throws std::out_of_range if a weight is missing from weight_map
 */
void block(Tensor &input, const std::unordered_map<std::string, Tensor> &weight_map, const int layer)
{
    const std::string prefix = "transformer.h." + std::to_string(layer) + ".";

    // n_head default equals 12
    // x += attention(ln_1(x))
    input += attention(
        layernorm(
            input,
            weight_map.at(prefix + "ln_1.weight"),
            weight_map.at(prefix + "ln_1.bias")),
        weight_map.at(prefix + "attn.c_attn.weight"),
        weight_map.at(prefix + "attn.c_attn.bias"),
        weight_map.at(prefix + "attn.c_proj.weight"),
        weight_map.at(prefix + "attn.c_proj.bias"));

    // x += mlp(ln_2(x))
    input += mlp(
        layernorm(
            input,
            weight_map.at(prefix + "ln_2.weight"),
            weight_map.at(prefix + "ln_2.bias")),
        weight_map.at(prefix + "mlp.c_fc.weight"),
        weight_map.at(prefix + "mlp.c_fc.bias"),
        weight_map.at(prefix + "mlp.c_proj.weight"),
        weight_map.at(prefix + "mlp.c_proj.bias"));
}

Tensor forward(const std::vector<int> &ids, const std::unordered_map<std::string, Tensor> &weight_map)
{

    Tensor out = embedding(ids, weight_map.at("transformer.wte.weight"), weight_map.at("transformer.wpe.weight"));

    // 12 blocks of GPT-2 small
    for (int i = 0; i < n_layer; i++)
    {
        block(out, weight_map, i);
    }

    out = layernorm(out, weight_map.at("transformer.ln_f.weight"), weight_map.at("transformer.ln_f.bias"));

    return out;
}

/**
 * @brief Compute next-token logits from the last row of the hidden states.
 *
 * lm_head shares its weights with wte (weight tying) and has no bias.
 *
 * @param input hidden states after ln_f, shape [T, d_model]
 * @param wte token embedding matrix, shape [vocab_size, d_model]
 * @return logits of shape [1, vocab_size]
 */
Tensor lm_head(const Tensor &input, const Tensor &wte)
{
    const int T = input.dim(0);       // amount of token
    const int d_model = input.dim(1); // dimension of model
    const int vocab_size = wte.dim(0);

    // last token shape is 1 x 768
    Tensor last({1, d_model}, {input.begin() + (T - 1) * d_model, input.end()});
    Tensor out({1, vocab_size});

    for (int i = 0; i < vocab_size; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < d_model; j++)
        {
            sum += last[j] * wte[i * d_model + j];
        }
        out[i] = sum;
    }

    return out;
}

std::vector<int> generate(const std::vector<int> &ids, const std::unordered_map<std::string, Tensor> &weight_map, int n_tokens)
{
    const Tensor &wte = weight_map.at("transformer.wte.weight");
    std::vector<int> out = ids;

    for (int i = 0; i < n_tokens; i++)
    {
        Tensor logits = lm_head(forward(out, weight_map), wte);

        out.push_back(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    return out;
}
