/**
 * @file loader.cpp
 * @brief Load model weights from binary file into a tensor.
 */

#include "loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <utility>

/**
 * @brief Load one raw float32 binary file into a tensor.
 *
 * The file holds only float data, so the caller must provide the shape.
 *
 * @param path path to the binary file
 * @param shape shape of tensor to create
 * @return tensor filled with file's data
 * @throws std::runtime_error if the file cannot be opened
 * @note The number of bytes read is not checked,
 *       so a file that is too short leaves the rest of the tensor uninitialized.
 */
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

/**
 * @brief Load all tensors of a model from one binary weight file.
 *
 * Tensors are stored back to back, with no offset table, so the file is read sequentially
 * from start to end.
 *
 * @param path path to the weight file
 * @return map from tensor name to tensor
 * @throws std::runtime_error if the file cannot be opened
 * @note Read errors are not checked, so a truncated file is not detected.
 */
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

        // read the tensor name
        std::int32_t name_len;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(std::int32_t));

        std::string name(name_len, '\0');
        file.read(reinterpret_cast<char *>(name.data()), name_len);

        // read the shape
        std::int32_t dim;
        file.read(reinterpret_cast<char *>(&dim), sizeof(std::int32_t));

        std::vector<int> shape;

        for (std::int32_t j = 0; j < dim; j++)
        {
            std::int32_t dim_size;
            file.read(reinterpret_cast<char *>(&dim_size), sizeof(std::int32_t));
            shape.push_back(static_cast<int>(dim_size));
        }

        // read the data straight into the tensor's buffer
        Tensor tensor(shape);

        file.read(reinterpret_cast<char *>(tensor.begin()), tensor.size() * sizeof(float));

        // move instead of copy: the tensor owns a large buffer
        result.emplace(std::move(name), std::move(tensor));
    }

    return result;
}
