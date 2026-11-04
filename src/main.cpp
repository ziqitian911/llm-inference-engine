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

    // mlp check: y (ln_1 output) -> block 0's mlp
    Tensor z = mlp(
        y,
        m.at("transformer.h.0.mlp.c_fc.weight"),
        m.at("transformer.h.0.mlp.c_fc.bias"),
        m.at("transformer.h.0.mlp.c_proj.weight"),
        m.at("transformer.h.0.mlp.c_proj.bias"));

    std::cout << "mlp shape: [" << z.dim(0) << ", " << z.dim(1) << "]" << std::endl;
    for (int t = 0; t < z.dim(0); t++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << z[t * d_model + j] << " ";
        }
        std::cout << std::endl;
    }

    // attention check: y (ln_1 output) -> block 0's attn
    // expected (first 3 per row), depends on what attention() returns:
    //   qkv / Q: 0.4701 -0.0775 0.5531 | 0.2831 -0.2919 0.6766 | 0.3628 0.7346 0.0941 | 0.2790 -0.3308 0.7370
    //   K:      -1.2526  2.3200 0.1722 | -1.6482 3.0222 1.2789 | -2.7801 3.8824 1.4420 | -2.0379 2.7107 1.9940
    //   V:       0.1637  0.2042 -0.0436 | -0.0800 0.0181 -0.0534 | 0.0671 -0.0090 0.1121 | -0.1263 -0.0219 - 0.1731
    Tensor a = attention(
        y,
        m.at("transformer.h.0.attn.c_attn.weight"),
        m.at("transformer.h.0.attn.c_attn.bias"),
        m.at("transformer.h.0.attn.c_proj.weight"),
        m.at("transformer.h.0.attn.c_proj.bias"),
        12);

    std::cout << "attention shape: [" << a.dim(0) << ", " << a.dim(1) << "]" << std::endl;
    for (int t = 0; t < a.dim(0); t++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << a[t * a.dim(1) + j] << " ";
        }
        std::cout << std::endl;
    }

    printf("Transpose test\n");

    Tensor({2, 3}, {1, 2, 3, 4, 5, 6}).print();

    transpose(Tensor({2, 3}, {1, 2, 3, 4, 5, 6})).print();

    return 0;
}
