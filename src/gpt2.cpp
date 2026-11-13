/**
 * @file gpt2.cpp
 * @brief GPT-2 model structure, built from the operators in ops.h
 */

#include "gpt2.h"
#include "ops.h"
#include "sampling.h"

#include <unordered_map>
#include <string>
#include <algorithm>
#include <random>
#include <stdexcept>

constexpr int n_layer = 12;

/**
 * @brief Run one transformer block, updating the residual stream in place.
 *
 * @param input residual stream of shape [T, d_model], modified in place
 * @param weight_map all model weights, keyed by tensor name
 * @param layer index of the block, from 0 to 11 for GPT-2 small
 * @param cache KV cache; this block reads and writes cache.K[layer] and cache.V[layer]
 * @throws std::out_of_range if a weight is missing from weight_map
 */
void block(
    Tensor &input,
    const std::unordered_map<std::string, Tensor> &weight_map,
    const int layer,
    KVCache &cache)
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
        weight_map.at(prefix + "attn.c_proj.bias"),
        cache.K[layer],
        cache.V[layer],
        cache.len);

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

/**
 * @brief Run the new tokens through all blocks and the final layernorm.
 *
 * @param ids new token ids: the whole prompt for prefill, one token for decode
 * @param weight_map all model weights, keyed by tensor name
 * @param cache KV cache; cache.len is increased by ids.size() after all blocks
 * @return hidden states of shape [ids.size(), d_model]
 * @throws std::out_of_range if a weight is missing from weight_map
 */
Tensor forward(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map,
    KVCache &cache)
{

    Tensor out = embedding(
        ids,
        weight_map.at("transformer.wte.weight"),
        weight_map.at("transformer.wpe.weight"),
        cache.len);

    // 12 blocks of GPT-2 small
    for (int i = 0; i < n_layer; i++)
    {
        block(out, weight_map, i, cache);
    }

    cache.len += out.dim(0);

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
        // assumes d_model % 8 == 0
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
        const float *w = wte.begin() + i * d_model; // start of row i of wte

        for (int j = 0; j < d_model; j += 8)
        {
            sum0 += last[j] * w[j];
            sum1 += last[j + 1] * w[j + 1];
            sum2 += last[j + 2] * w[j + 2];
            sum3 += last[j + 3] * w[j + 3];
            sum4 += last[j + 4] * w[j + 4];
            sum5 += last[j + 5] * w[j + 5];
            sum6 += last[j + 6] * w[j + 6];
            sum7 += last[j + 7] * w[j + 7];
        }
        out[i] = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
    }

    return out;
}

/**
 * @brief Generate new tokens one at a time, appending each to the sequence.
 *
 * Step 0 (prefill) runs the whole prompt and fills the KV cache.
 * Every later step (decode) runs only the token generated in the previous
 * step; earlier tokens are seen through their cached K and V.
 *
 * @param ids prompt token ids
 * @param weight_map all model weights, keyed by tensor name
 * @param n_tokens number of new tokens to generate
 * @param temperature 0 means greedy (argmax); > 0 divides logits before sampling (higher = more random)
 * @param top_k keep only the k highest-scoring tokens before sampling
 * @param top_p keep the smallest set of tokens whose probabilities sum to >= p
 * @return prompt ids followed by the n_tokens generated ids
 * @throws std::out_of_range if a weight is missing from weight_map
 * @throws std::invalid_argument if the prompt plus n_tokens exceeds the context length
 */
std::vector<int> generate(
    const std::vector<int> &ids,
    const std::unordered_map<std::string, Tensor> &weight_map,
    int n_tokens,
    float temperature,
    int top_k,
    float top_p)
{

    const int context_length = weight_map.at("transformer.wpe.weight").dim(0);

    if (ids.size() + n_tokens > context_length)
    {
        throw std::invalid_argument("generate: prompt length + n_tokens exceeds the context length (1024)");
    }

    const Tensor &wte = weight_map.at("transformer.wte.weight");
    std::vector<int> out = ids;
    std::mt19937 rng(42); // fixed seed: same output every run
    KVCache cache;

    for (int i = 0; i < n_tokens; i++)
    {
        std::vector<int> input;

        if (i == 0)
        {
            input = out; // prefill: the whole prompt
        }
        else
        {
            input = {out.back()}; // decode: only the token generated last step
        }

        Tensor before_logits = forward(input, weight_map, cache);
        Tensor logits = lm_head(before_logits, wte);

        int next;
        if (temperature == 0.0f)
        {
            // greedy: take the highest score
            next = std::max_element(logits.begin(), logits.end()) - logits.begin();
        }
        else
        {
            // temperature -> top-k -> softmax -> top-p -> draw one
            Tensor probs = softmax(apply_top_k(apply_temperature(logits, temperature), top_k));
            next = sample(apply_top_p(probs, top_p), rng);
        }

        out.push_back(next);
    }

    return out;
}