#include "loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <utility>

Tensor load_tensor(const std::string &path, const std::vector<int> &shape)
{
    Tensor t(shape);

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("cannot open " + path);
    }

    file.read(reinterpret_cast<char *>(t.begin()), t.size() * sizeof(float));

    return t;
}

std::unordered_map<std::string, Tensor> load_model(const std::string &path)
{

    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("cannot open " + path);
    }

    std::unordered_map<std::string, Tensor> result;

    std::int32_t tensor_amount;
    file.read(reinterpret_cast<char *>(&tensor_amount), sizeof(std::int32_t));

    for (std::int32_t i = 0; i < tensor_amount; i++)
    {

        std::int32_t name_len;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(std::int32_t));

        std::string name(name_len, '\0');
        file.read(reinterpret_cast<char *>(name.data()), name_len);

        std::int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(std::int32_t));

        std::vector<int> shape;

        for (std::int32_t j = 0; j < dim; j++)
        {
            std::int32_t dim_size;
            file.read(reinterpret_cast<char *>(&dim_size), sizeof(std::int32_t));
            shape.push_back(static_cast<int>(dim_size));
        }

        Tensor tensor(shape);

        file.read(reinterpret_cast<char *>(tensor.begin()), tensor.size() * sizeof(float));

        result.emplace(std::move(name), std::move(tensor));
    }

    return result;
}
