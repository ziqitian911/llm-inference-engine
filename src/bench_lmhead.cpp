// Benchmark the GPT-2 lm_head matmul shape on CPU:
//
//     A [T, 768]  x  B [768, 50257]  ->  C [T, 50257]
//
// T = number of tokens processed in one forward pass:
//   T = 1        -> decode  (generating one new token at a time)
//   T = 16..512  -> prefill (feeding the whole prompt at once)
//
// B is 768 * 50257 * 4 bytes ~= 154 MB regardless of T, so small T reads a lot of
// memory for very little compute (memory-bound), large T reuses each weight T times
// (compute-bound).
//
// Same methodology as bench.cpp: correctness check, warm-up, min of K runs, cool-down.
// Values are all 1.0f: timing does not depend on the numbers, only on the shapes.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>
#include <functional>
#include <cmath>
#include <algorithm>
#include <string>

#include "tensor.h"
#include "gemm.h"

// Run fn once as warm-up (not timed), then K times; return the minimum seconds.
static double time_min(const std::function<void()> &fn, int K)
{
    fn();
    double best = 1e30;
    for (int r = 0; r < K; r++)
    {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

// True if two tensors match element-wise within a small tolerance.
static bool same(const Tensor &a, const Tensor &b)
{
    if (a.size() != b.size())
        return false;
    for (int i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) > 1e-3f)
            return false;
    return true;
}

int main()
{
    const int K = 5;              // timed runs per config (take the min)
    const int D = 768;            // GPT-2 small hidden size (K dim of the matmul)
    const int V = 50257;          // GPT-2 vocab size (N dim of the matmul)
    volatile float sink = 0;      // keep results "used" so the optimizer can't delete the work

    // B is the same for every T: build it once (~154 MB).
    Tensor B({D, V}, std::vector<float>((size_t)D * V, 1.0f));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "lm_head matmul: [T, " << D << "] x [" << D << ", " << V << "]\n\n";

    for (int T : {1, 16, 128, 512})
    {
        Tensor A({T, D}, std::vector<float>((size_t)T * D, 1.0f));

        // Reference: single-thread reorder (already verified against naive in bench.cpp;
        // naive itself is too slow at T=512).
        Tensor ref = matmul(A, B, MatmulMode::Reorder);
        double flops = 2.0 * T * D * V;

        auto run = [&](const std::string &name, MatmulMode mode)
        {
            Tensor C = matmul(A, B, mode);
            bool ok = same(C, ref);

            double sec = time_min(
                [&]
                {
                    Tensor t = matmul(A, B, mode);
                    sink = sink + t[0];
                },
                K);

            std::cout
                << "T=" << std::left << std::setw(4) << T
                << "  " << std::setw(13) << name << std::right
                << std::setw(10) << sec * 1e3 << " ms   "
                << std::setw(10) << flops / sec / 1e9 << " GFLOPS   "
                << "correct=" << (ok ? "yes" : "NO!!")
                << "\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        };

        run("reorder", MatmulMode::Reorder);
        run("threaded", MatmulMode::Threaded);
        run("threadedpool", MatmulMode::ThreadedPool);
        std::cout << "\n";
    }

    return 0;
}
