/**
 * @file gpt2.cpp
 * @brief GPT-2 model structure, built from the operators in ops.h
 */

#include "gpt2.h"
#include "ops.h"

#include <unordered_map>
#include <string>

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
