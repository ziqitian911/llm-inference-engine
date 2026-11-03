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
    // expected (first 3 of each row, from PyTorch wte[id] + wpe[pos]):
    //   -0.0875 -0.3301  0.0152
    //    0.0355 -0.0567 -0.0626
    //   -0.1446  0.0671  0.0601
    //   -0.1104 -0.1131  0.1386
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

    return 0;
}
