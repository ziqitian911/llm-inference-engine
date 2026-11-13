#include "sampling.h"

#include <stdexcept>
#include <algorithm>
#include <vector>
#include <limits>
#include <functional>
#include <numeric>

Tensor apply_temperature(const Tensor &logits, const float temperature)
{

    //  temperature must be positive
    if (temperature <= 0)
    {
        throw std::invalid_argument("apply_temperature: temperature must be greater than 0.");
    }

    Tensor out = logits;

    for (int i = 0; i < out.size(); i++)
    {
        out[i] /= temperature;
    }

    return out;
}

Tensor apply_top_p(const Tensor &probs, const float p)
{
    if (p <= 0.0f || p > 1.0f)
    {
        throw std::invalid_argument("apply_top_p: p must be in (0, 1]");
    }

    const int vocab_size = probs.size();

    // sort token ids by probablity, largest first; probs it self is untouched
    std::vector<int> idx(vocab_size);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(
        idx.begin(),
        idx.end(),
        [&](int a, int b)
        { return probs[a] > probs[b]; });

    // keep the smallest set of tokens whose total probablity readhes p
    float cumulative = 0.0f;
    int n_keep = 0;
    while (n_keep < vocab_size)
    {
        cumulative += probs[idx[n_keep]];
        n_keep++;
        if (cumulative >= p)
        {
            break;
        }
    }

    // zero out the rest, then rescale the kept ones so they sum to 1
    Tensor out = probs;
    std::fill(out.begin(), out.end(), 0.0f);
    for (int i = 0; i < n_keep; i++)
    {
        out[idx[i]] = probs[idx[i]] / cumulative;
    }

    return out;
}

Tensor apply_top_k(const Tensor &logits, const int k)
{
    if (k <= 0)
    {
        throw std::invalid_argument("apply_top_k: k must be greater than 0");
    }

    const int vocalib_size = logits.size();

    // k covers every token: nothing to filter
    if (k >= vocalib_size)
    {
        return logits;
    }

    // find the k-th largest value on a copy, since nth_element reorders it
    std::vector<float> sorted(logits.begin(), logits.end());

    std::nth_element(
        sorted.begin(),
        sorted.begin() + (k - 1),
        sorted.end(),
        std::greater<float>());

    const float kth = sorted[k - 1];

    // drop everything below the k-th largest value
    const float neg_inf = -std::numeric_limits<float>::infinity();

    Tensor out = logits;
    for (int i = 0; i < out.size(); i++)
    {
        if (out[i] < kth)
        {
            out[i] = neg_inf;
        }
    }

    return out;
}

int sample(const Tensor &probs, std::mt19937 &rng)
{
    // picks index i with probability proportional to probs[i]
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}
