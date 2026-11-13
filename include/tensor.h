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

    float &at(const std::vector<int> &idx);
    float at(const std::vector<int> &idx) const;
    float *begin();
    float *end();
    const float *begin() const;
    const float *end() const;

    void print() const;

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

    /**
     * @brief Number of elements in the tensor
     *
     * @return product of all dimensions
     */
    int size() const
    {
        return this->count_;
    }

    /**
     * @brief Length of one dimension.
     * @param i which dimension (0-based).
     * @return shape[i], the size of dimension i.
     */
    int dim(int i) const
    {
        return shape_[i];
    }

    int ndim() const
    {
        return static_cast<int>(this->shape_.size());
    }

    /**
     * @brief Shape of the tensor, e.g. {50257, 768}.
     * @return read-only reference to the shape (no copy).
     */
    const std::vector<int> &shape() const
    {
        return this->shape_;
    }
};

Tensor operator+(const Tensor &a, const Tensor &b);