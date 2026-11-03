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

    return 0;
}
