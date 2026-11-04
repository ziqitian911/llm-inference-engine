#pragma once
#include <vector>

class Tensor
{
private:
    int count; // element count
    float *data;
    std::vector<int> shape;
    std::vector<int> strides;

public:
    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor(const std::vector<int> &shape, const std::vector<float> &values);
    ~Tensor();

    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;
    float &operator[](int i);
    float operator[](int i) const;

    int size() const;
    int dim(int i) const;
    int ndim() const;

    float &at(const std::vector<int> &idx);
    float at(const std::vector<int> &idx) const;
    float *begin();
    float *end();
    const float *begin() const;
    const float *end() const;

    void print() const;
};