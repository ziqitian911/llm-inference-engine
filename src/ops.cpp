#include "ops.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>
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

    for (int i = 0; i < out.size(); i++)
    {
        out[i] += bias[i];
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

    const float max = *std::max_element(out.begin(), out.end());

    for (int i = 0; i < out.size(); i++)
    {
        out[i] = std::exp(out[i] - max);
    }

    float sum = std::accumulate(out.begin(), out.end(), 0.0f);

    for (int i = 0; i < out.size(); i++)
    {
        out[i] /= sum;
    }

    return out;
}

/**
 * @brief
 *
 * @param ids
 * @param wte word token embedding
 * @param wpe word position embedding
 */
Tensor embedding(const std::vector<int> &ids, const Tensor &wte, const Tensor &wpe)
{

    const int T = static_cast<int>(ids.size()); // amount of token
    const int d_model = wte.dim(1);             // dimension of the model

    Tensor result({T, d_model});

    for (int t = 0; t < T; t++)
    {
        int token_id = ids[t];
        for (int j = 0; j < d_model; j++)
        {
            result[t * d_model + j] = wte[token_id * d_model + j] + wpe[t * d_model + j];
        }
    }

    return result;
}
