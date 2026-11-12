#include "ops.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <numbers>
#include <limits>

#include "gemm.h"

/**
 * @brief Apply ReLU element-wise: out[i] = max(0, input[i]).
 * @param input the tensor to activate.
 * @return a new tensor with negative values set to 0.
 */
Tensor relu(const Tensor &input)
{
    Tensor out = input;
    int n = out.size();

    for (int i = 0; i < n; i++)
    {
        out[i] = std::max(.0f, out[i]);
    }

    return out;
}

Tensor gelu(const Tensor &input)
{
    Tensor out = input;
    const int n = out.size();
    const float k = std::sqrt(2.0f / std::numbers::pi_v<float>);

    for (int i = 0; i < n; i++)
    {
        float x = out[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }
    return out;
}

/**
 * @brief Fully-connected (Linear) layer: out = input * weights + bias.
 * @param input input activations, shape [M x in_features].
 * @param weights weight matrix, shape [in_features x out_features].
 * @param bias one bias per output neuron, length out_features.
 * @return the layer output, shape [M x out_features].
 */
Tensor linear(const Tensor &input, const Tensor &weights, const Tensor &bias)
{
    // check dim
    if (input.dim(1) != weights.dim(0))
    {
        throw std::invalid_argument("linear: input cols must equal weights rows");
    }

    // matmul
    Tensor out = matmul(input, weights);

    const int out_features = out.dim(1);

    for (int i = 0; i < out.size(); i++)
    {
        out[i] += bias[i % out_features];
    }

    return out;
}

/**
 * @brief Softmax: turn a vector of scores into a probability distribution.
 *
 * Numerically stable: subtracts the max before exp to avoid exp() overflow.
 * @param input the score vector.
 * @return a new tensor whose values are in [0, 1] and sum to 1.
 */
Tensor softmax(const Tensor &input)
{
    Tensor out = input;

    // treat any tensor as n_rows rows of row_len numbers (last dim is contiguous)
    const int row_len = out.dim(out.ndim() - 1);
    const int n_rows = out.size() / row_len;

    for (int r = 0; r < n_rows; r++)
    {
        float *row = out.begin() + r * row_len;

        const float max = *std::max_element(row, row + row_len);

        for (int c = 0; c < row_len; c++)
        {
            row[c] = std::exp(row[c] - max);
        }

        const float sum = std::accumulate(row, row + row_len, 0.0f);

        for (int c = 0; c < row_len; c++)
        {
            row[c] /= sum;
        }
    }

    return out;
}

/**
 * @brief Look up token and position embeddings and add them.
 *
 * Row t of the output = wte[ids[t]] (what the token is)
 *                     + wpe[start_pos + t] (where it is in the sequence).
 *
 * @param ids token ids to embed, T of them
 * @param wte token embedding table, shape [vocab_size, d_model]
 * @param wpe position embedding table, shape [context_length, d_model]
 * @param start_pos position of ids[0] in the whole sequence (0 for prefill, cache.len for decode)
 * @return embeddings of shape [T, d_model]
 */
Tensor embedding(
    const std::vector<int> &ids,
    const Tensor &wte,
    const Tensor &wpe,
    int start_pos)
{

    const int T = static_cast<int>(ids.size()); // amount of token
    const int d_model = wte.dim(1);             // dimension of the model

    Tensor out({T, d_model});

    for (int t = 0; t < T; t++)
    {
        int token_id = ids[t];
        for (int j = 0; j < d_model; j++)
        {
            // wte row = which token, wpe row = which position in the whole sequence
            out[t * d_model + j] = wte[token_id * d_model + j] + wpe[(t + start_pos) * d_model + j];
        }
    }

    return out;
}

/**
 * @brief Transpose a matrix, change it size from M * N to N * M
 *
 *
 * @return transposed matrix
 */
Tensor transpose(const Tensor &input)
{
    const int M = input.dim(0);
    const int N = input.dim(1);
    Tensor out({N, M});

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            out[n * M + m] = input[m * N + n];
        }
    }

    return out;
}

Tensor layernorm(const Tensor &input, const Tensor &weight, const Tensor &bias, const float epsilon)
{
    const int T = input.dim(0);
    const int d_model = weight.dim(0);

    Tensor out({T, d_model});

    for (int i = 0; i < T; i++)
    {
        float mean = std::accumulate(input.begin() + i * d_model, input.begin() + i * d_model + d_model, 0.0f) / d_model;

        // c
        float var = 0.0f;
        for (int j = 0; j < d_model; j++)
        {
            var += (input[i * d_model + j] - mean) * (input[i * d_model + j] - mean);
        }
        var /= d_model;
        float std_dev = std::sqrt(var + epsilon);

        for (int j = 0; j < d_model; j++)
        {
            out[i * d_model + j] = (input[i * d_model + j] - mean) / std_dev;
            out[i * d_model + j] = out[i * d_model + j] * weight[j] + bias[j];
        }
    }

    return out;
}

Tensor mlp(const Tensor &input, const Tensor &fc_weight, const Tensor &fc_bias, const Tensor &proj_weight, const Tensor &proj_bias)
{
    Tensor out = input;
    out = linear(out, fc_weight, fc_bias);
    out = gelu(out);
    out = linear(out, proj_weight, proj_bias);
    return out;
}

/**
 * @brief Multi-head causal self-attention with a KV cache.
 *
 * Only the T new tokens are passed in. Their K and V are appended to the cache,
 * and their Q attends to all len + T cached tokens.
 *
 * @param input new tokens after ln_1, shape [T, d_model]
 * @param attn_weight c_attn weight, shape [d_model, 3 * d_model] (Q | K | V)
 * @param attn_bias c_attn bias, shape [3 * d_model]
 * @param proj_weight c_proj weight, shape [d_model, d_model]
 * @param proj_bias c_proj bias, shape [d_model]
 * @param k_cache this layer's key cache, shape [context_length, d_model], written to
 * @param v_cache this layer's value cache, shape [context_length, d_model], written to
 * @param len number of tokens already in the cache before this call
 * @param n_head number of attention heads
 * @return attention output for the new tokens, shape [T, d_model]
 */
Tensor attention(
    const Tensor &input,
    const Tensor &attn_weight,
    const Tensor &attn_bias,
    const Tensor &proj_weight,
    const Tensor &proj_bias,
    Tensor &k_cache,
    Tensor &v_cache,
    int len,
    const int n_head)
{

    // calculate qkv
    Tensor qkv = linear(input, attn_weight, attn_bias);

    const int T = qkv.dim(0);         // rows of out tensor
    const int d_model = input.dim(1); // cols of out tensor

    Tensor Q({T, d_model});

    // split Q, K and V from qkv
    for (int i = 0; i < T; i++)
    {
        for (int j = 0; j < d_model; j++)
        {
            const int dst = i * d_model + j;
            const int src = i * 3 * d_model + j;
            // new token i goes right after the cached ones: row len + i
            const int cache_dst = (len + i) * d_model + j;

            Q[dst] = qkv[src];
            k_cache[cache_dst] = qkv[src + d_model];
            v_cache[cache_dst] = qkv[src + 2 * d_model];
        }
    }

    // multi-head

    Tensor out({T, d_model});
    const int head_dim = d_model / n_head; // cols of each head's Q, K and V
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float sqrt_d_k = std::sqrt((float)head_dim);

    // Q, K, V for each head
    for (int head = 0; head < n_head; head++)
    {
        // Q: only the T new tokens ask
        // K, V: all len + T tokens (old + new) are looked at
        Tensor Q_h({T, head_dim}), K_h({len + T, head_dim}), V_h({len + T, head_dim});

        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < head_dim; i++)
            {
                int dst = t * head_dim + i;
                int src = (t * d_model) + (head * head_dim) + i;
                Q_h[dst] = Q[src];
            }
        }

        // copy this head's columns of K and V for all len + T tokens from the cache
        for (int i = 0; i < len + T; i++)
        {
            for (int j = 0; j < head_dim; j++)
            {
                int cache_src = (i * d_model) + (head * head_dim) + j;
                int dst = i * head_dim + j;

                K_h[dst] = k_cache[cache_src];
                V_h[dst] = v_cache[cache_src];
            }
        }

        // dot product Q * K^T
        Tensor trans_K_h = transpose(K_h);
        Tensor scores = matmul(Q_h, trans_K_h);

        // scaling and mask
        for (int i = 0; i < scores.dim(0); i++)
        {
            for (int j = 0; j < scores.dim(1); j++)
            {
                // scaling
                const int dst = i * scores.dim(1) + j;
                scores[dst] /= sqrt_d_k;

                // causal mask: row i of Q is token len + i in the whole sequence,
                // so it cannot see columns j > len + i
                // in decode (T = 1) nothing is masked: the last token sees everything
                if (j > i + len)
                {
                    scores[dst] = neg_inf;
                }
            }
        }

        // softmax
        scores = softmax(scores);

        // dot product scores times V_h
        Tensor head_out = matmul(scores, V_h);

        // merge to out tensor
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < head_dim; i++)
            {
                const int src = t * head_dim + i;
                const int dst = (t * d_model) + (head * head_dim) + i;
                out[dst] = head_out[src];
            }
        }
    }

    return linear(out, proj_weight, proj_bias);
}