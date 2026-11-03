#include <iostream>
#include "tensor.h"
#include "gemm.h"
#include "ops.h"
#include "loader.h"
#include "threadpool.h"

int main(int argc, char const *argv[])
{

    Tensor fc1_w = load_tensor("model/fc1_weight.bin", {784, 128});
    Tensor fc1_b = load_tensor("model/fc1_bias.bin", {1, 128});
    Tensor fc2_w = load_tensor("model/fc2_weight.bin", {128, 10});
    Tensor fc2_b = load_tensor("model/fc2_bias.bin", {1, 10});

    Tensor img = load_tensor("model/test_image.bin", {1, 784});

    Tensor h = linear(img, fc1_w, fc1_b);
    h = relu(h);
    Tensor out = linear(h, fc2_w, fc2_b);
    Tensor probs = softmax(out);

    out.print();

    float *it = std::max_element(probs.begin(), probs.end());
    int pred = it - probs.begin();

    std::cout << pred << std::endl;
    auto m = load_model("model/gpt2.bin");

    std::cout << m.size() << std::endl;
    std::cout << m["transformer.wte.weight"].dim(0) << std::endl;
    std::cout << m["transformer.wte.weight"].dim(1) << std::endl;
    std::cout << m["transformer.h.0.attn.c_attn.weight"].dim(0) << std::endl;
    std::cout << m["transformer.h.0.attn.c_attn.weight"].dim(1) << std::endl;
    std::cout << m["transformer.ln_f.bias"].dim(0) << std::endl;

    // value check: first 5 floats of wte, compare with PyTorch's wte.weight[0, :5]
    const Tensor &wte = m.at("transformer.wte.weight");
    for (int i = 0; i < 5; i++)
    {
        std::cout << wte[i] << " ";
    }
    std::cout << std::endl;

    // embedding check: "Hello, world!" -> ids [15496, 11, 995, 0]
    const Tensor &wpe = m.at("transformer.wpe.weight");
    Tensor x = embedding({15496, 11, 995, 0}, wte, wpe);
    const int d_model = x.dim(1);

    std::cout << "embedding shape: [" << x.dim(0) << ", " << d_model << "]" << std::endl;
    for (int t = 0; t < x.dim(0); t++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << x[t * d_model + j] << " ";
        }
        std::cout << std::endl;
    }

    // layernoem check
    const Tensor &ln1_w = m.at("transformer.h.0.ln_1.weight");
    const Tensor &ln1_b = m.at("transformer.h.0.ln_1.bias");
    Tensor y = layernorm(x, ln1_w, ln1_b);

    std::cout << "layernorm shape: [" << y.dim(0) << ", " << y.dim(1) << "]" << std::endl;
    for (int t = 0; t < y.dim(0); t++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << y[t * d_model + j] << " ";
        }
        std::cout << std::endl;
    }

    // gelu check: expected
    // -0.0036 -0.0454 -0.1588 -0.1543 0.0000 0.3457 0.8412 1.9546 2.9964
    Tensor g = gelu(Tensor({9}, {-3.f, -2.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 2.f, 3.f}));
    for (int i = 0; i < g.size(); i++)
    {
        std::cout << g[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
