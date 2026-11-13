#pragma once
#include <vector>

class Tensor
{
private:
    int count_; // element count
    float *data_;
    std::vector<int> shape_;
    std::vector<int> strides_;

public:
    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor(const std::vector<int> &shape, const std::vector<float> &values);
    ~Tensor();

    Tensor &operator+=(const Tensor &other);
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    /**
     * @brief Access the i-th element by flat index (row-major).
     * @param i flat index, 0 .. size()-1.
     * @return reference to the element, so it can be read or written.
     */
    float &operator[](int i)
    {
        return this->data_[i];
    }

    /**
     * @brief Read the i-th element by flat index (row-major).
     * @param i flat index, 0 .. size()-1.
     * @return a copy of the element (read only).
     */
    float operator[](int i) const
    {
        return this->data_[i];
    }

    int size() const;
    int dim(int i) const;
    int ndim() const;

    /**
     * @brief Shape of the tensor, e.g. {50257, 768}.
     * @return read-only reference to the shape (no copy).
     */
    const std::vector<int> &shape() const
    {
        return this->shape_;
    }

    float &at(const std::vector<int> &idx);
    float at(const std::vector<int> &idx) const;
    float *begin();
    float *end();
    const float *begin() const;
    const float *end() const;

    void print() const;
};

Tensor operator+(const Tensor &a, const Tensor &b);