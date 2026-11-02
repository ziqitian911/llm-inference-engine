# mini-inference-engine

A neural-network inference engine **written from scratch in C++20**, with no deep-learning libraries. It loads an MLP trained in PyTorch and classifies MNIST digits with the same prediction as PyTorch. Its matrix-multiply (GEMM) core is then optimized step by step and measured at every step: **1.8 &rarr; 312 GFLOPS (~170&times;) on an Apple M4 Pro CPU**, plus hand-written CUDA kernels benchmarked against cuBLAS on an NVIDIA GPU.

## Benchmarks

Square N&times;N `float32` matrix multiply, GFLOPS = 2N&sup3; / time (higher is better).

### CPU — Apple M4 Pro, Release (`-O3`)

| Version | N=512 | N=1024 | N=2048 | Speedup @2048 |
|:---|---:|---:|---:|---:|
| Naive | 3.1 | 2.6 | 1.8 | 1&times; |
| Reorder + SIMD (NEON) | 33 | 35 | 33 | ~18&times; |
| Threaded | 159 | 248 | 312 | ~170&times; |
| Thread pool | 158 | 258 | 293 | ~160&times; |

### GPU — NVIDIA RTX 5070 Ti Laptop, kernel only

| Version | N=512 | N=1024 | N=2048 | % of cuBLAS @2048 |
|:---|---:|---:|---:|---:|
| Naive | 1587 | 1923 | 1911 | ~10% |
| Shared-memory tiled | 2136 | 2593 | 2625 | ~14% |
| cuBLAS | 9300 | 16677 | 18917 | 100% |

*Each number: one warm-up run, then the minimum of 5 timed runs. Every result is checked against the naive output. CPU rows time a whole `matmul()` call, including output allocation; GPU rows time the kernel only (cudaEvent), without host&harr;device copies. CPU and GPU were measured on different machines, so compare within a table, not across. See `src/bench.cpp`.*

## Key findings

**CPU**

- **Abstractions have hidden costs.** An indexing helper that heap-allocated on every element access made a 512&sup3; matmul take 33 s; flat indexing brought it to 0.37 s (~100&times;).
- **One keyword unlocked SIMD.** After reordering the loops (`ijk` &rarr; `ikj`) for sequential memory access, adding `__restrict__` told the compiler the matrices don't overlap, so it auto-vectorized the inner loop with NEON (confirmed with `-Rpass=loop-vectorize`): ~1.1 &rarr; ~32 GFLOPS.
- **Same optimization, opposite result.** Cache blocking was slower than plain reordering on the M4 Pro (27 vs. 33 GFLOPS at N=2048, 16 MB L2), but ~30% faster on an Intel Core Ultra 9 275HX. Optimizations are hardware-specific; measure on the target.
- **A thread pool isn't automatically faster.** On the M4 Pro, a persistent pool and spawning threads per call tie at every size. On the Intel machine the pool was ~1.9&times; faster at N=512, likely because thread startup is a large share of a ~1 ms job.

**GPU**

- **Aliasing, again.** Accumulating directly into the output array (`c[...] +=`) forced a global-memory write every iteration; accumulating in a local variable and writing once made the GPU matmul ~2.4&times; faster (end-to-end, N=2048). The compiler can't keep `c[...]` in a register because it can't prove the pointers don't alias: the same lesson as `__restrict__` on the CPU.
- **Shared-memory tiling: 1.4&times;, not 16&times;.** Tiling cuts global-memory reads by 16&times; in theory, but the kernel only got ~1.4&times; faster. I suspect the L1/L2 caches already absorbed much of that reuse in the naive kernel, and the bottleneck moved to shared memory (not yet verified with Nsight Compute).
- **cuBLAS is ~7&times; faster than my best kernel.** The main gap is register tiling (each thread computes many outputs), plus vectorized loads, double buffering, and avoiding shared-memory bank conflicts.
- **The kernel is not the whole story.** At N=2048, cuBLAS's kernel is ~7&times; faster than mine, but end-to-end only ~2&times;: host&harr;device copies take ~5 ms of its ~6 ms total. Real engines keep weights resident on the GPU.

## Quick start

Requires a C++20 compiler and CMake &ge; 3.20. CUDA is optional.

```bash
# 1) train the MLP and export weights + a test image (one-time; needs Python + PyTorch)
pip install torch torchvision numpy
python python/export.py          # writes model/*.bin

# 2) build and run
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/main                     # prints the logits and the predicted digit
./build/bench                    # GEMM benchmark
```

The CUDA backend is built automatically when a CUDA compiler is found. To force a CPU-only build, add `-DENABLE_CUDA=OFF`.

## How it works

```
28×28 image ──▶ Tensor ──▶ Linear(784→128) ──▶ ReLU ──▶ Linear(128→10) ──▶ Softmax ──▶ argmax ──▶ digit
```

- **`Tensor`**: an N-dimensional array over a raw heap buffer with row-major strides. It manages its own memory (RAII + Rule of Five) on purpose, to practice C++ resource management.
- **Operators**: `ops.*` (ReLU, Linear, Softmax) and `gemm.*` (matmul with selectable modes: Naive / Reorder / Blocked / Threaded / ThreadedPool, plus CUDA / BlockedCUDA / CUBLAS when built with CUDA). The thread pool (`threadpool.*`) is hand-written with `std::mutex`, `std::condition_variable` and a task queue.
- **Weights**: `python/export.py` trains the MLP in PyTorch and dumps each tensor as raw `float32`; `loader.*` reads them back with `std::ifstream`. PyTorch stores `Linear` weights as `[out, in]`, so they are transposed on export to match the C++ GEMM.

## Limitations

- The tiled CUDA kernel assumes matrix sizes are multiples of 16 (no partial tiles).
- Each GPU call allocates device memory and copies data both ways; a real engine would keep weights on the device.
- Benchmarks use square matrices; the cuBLAS call was not separately tested on non-square shapes.
- `float32` only, one model (MNIST MLP), no batching, operator fusion, or quantization.

## Related work

- Simon Boehm, [*How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*](https://siboehm.com/articles/22/CUDA-MMM) and [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA): a much deeper walk through the same CUDA GEMM path.

## Tech

C++20 · CMake · `std::thread` / `mutex` / `condition_variable` / `latch` · ARM NEON (auto-vectorized) · CUDA · cuBLAS · PyTorch (training/export only)
