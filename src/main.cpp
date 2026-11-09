#include <chrono>
#include <iostream>
#include <vector>

#include "gpt2.h"
#include "loader.h"

// print ids on one line, ready to paste into the Python decoder
static void print_ids(const std::vector<int> &ids)
{
    for (int id : ids)
    {
        std::cout << id << " ";
    }
    std::cout << std::endl;
}

int main()
{
    auto m = load_model("model/gpt2.bin");
    const std::vector<int> prompt = {15496, 11, 995, 0}; // "Hello, world!"

    // greedy decoding: must match PyTorch exactly
    const std::vector<int> expected = {15496, 11, 995, 0, 198, 198, 40, 1101, 7926, 11, 475, 314,
                                       1101, 407, 1654, 644, 284, 466, 13, 198, 198, 40, 1101, 7926};

    auto start = std::chrono::steady_clock::now();
    std::vector<int> greedy = generate(prompt, m, 20);
    auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "greedy:  ";
    print_ids(greedy);
    std::cout << "matches PyTorch: " << (greedy == expected ? "yes" : "NO") << std::endl;
    std::cout << "time: " << seconds << " s, " << 20 / seconds << " tokens/s" << std::endl;

    // sampling with top_k = 1 leaves one candidate, so it must equal greedy
    std::vector<int> top1 = generate(prompt, m, 20, 0.8f, 1, 1.0f);
    std::cout << "top_k=1 equals greedy: " << (top1 == greedy ? "yes" : "NO") << std::endl;

    // real sampling (seed is fixed, so the output is the same every run)
    std::vector<int> sampled = generate(prompt, m, 40, 0.8f, 50, 0.9f);
    std::cout << "sampled: ";
    print_ids(sampled);

    return 0;
}
