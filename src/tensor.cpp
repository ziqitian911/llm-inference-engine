#include "tensor.h"
#include <utility>
#include <algorithm>
#include <iostream>

/**
 * @brief default constructor
 */
Tensor::Tensor()
{
    this->count = 0;
    this->data = nullptr;
}

/**
 * @brief Build a tensor of the given shape and allocate its buffer
 * @param shape dimensions, e.g. {28, 28} for a 28x28 image
 */
Tensor::Tensor(const std::vector<int> &shape)
{
    this->shape = shape;
    this->count = 1;

    for (int num : shape)
    {
        this->count *= num;
    }

    this->strides.assign(shape.size(), 1);

    for (int i = (int)shape.size() - 2; i >= 0; i--)
    {
        this->strides[i] = this->strides[i + 1] * shape[i + 1];
    }

    this->data = new float[count];
}

/**
 * @brief Copy constructor (deep copy)
 *
 * Allocates a separate buffer and copies the values,
 * so two Tensors never share one pointer - sharing would double-free on destruction
 * @param other tensor to copy from
 */
Tensor::Tensor(const Tensor &other)
{
    this->count = other.count;
    this->shape = other.shape;
    this->strides = other.strides;

    this->data = new float[this->count];

    // for (int i = 0; i < count; i++)
    // {
    //     this->data[i] = other.data[i];
    // }

    std::copy(other.data, other.data + count, this->data);
}

/**
 * @brief Move constructor: takes ownership of other buffer in O(1)
 *
 * Steals other's data pointer, then nulls out other's pointer
 * so other's destructor runs delete[] nullptr (a no-op) instead of freeing the buffer we own now
 * Marked noexcept so containers like std::vector will use it
 * @param other tensor to move from; left empty afterwards
 */
Tensor::Tensor(Tensor &&other) noexcept
{
    this->data = other.data;
    this->shape = std::move(other.shape);
    this->strides = std::move(other.strides);
    this->count = other.count; // move is useless for int
    other.data = nullptr;
    other.count = 0;
}

/**
 * @brief Build a tensor of the given shape and fill it with the given values.
 * @param shape dimensions, e.g. {2, 3} for a 2x3 tensor.
 * @param values initial values in row-major order (size must equal the element count).
 */
Tensor::Tensor(const std::vector<int> &shape, const std::vector<float> &values) : Tensor(shape)
{
    std::copy(values.begin(), values.end(), this->data);
}

/**
 * @brief Destructor: frees the owned heap buffer (RAII)
 */
Tensor::~Tensor()
{
    delete[] this->data;
}

/**
 * @brief Copy assignment (deep copy)
 *
 * Frees our current buffer, then allocates a new one and copies other's
 * values, so the two Tensors stay independent.
 * Guards against self-assignment
 * @param other tensor to copy from (left unchanged)
 * @return reference to *this, so assignments can chain (a = b = c)
 */
Tensor &Tensor::operator=(const Tensor &other)
{
    // t1 = t2;
    // t1 = t1;
    if (this == &other)
    {
        return *this;
    }

    delete[] this->data;
    this->count = other.count;
    this->shape = other.shape;
    this->strides = other.strides;
    this->data = new float[this->count];

    for (int i = 0; i < count; i++)
    {
        this->data[i] = other.data[i];
    }

    return *this;
}

/**
 * @brief Move assignment: take over other's buffer in O(1)
 *
 * Frees our current buffer, steals other's pointer, then nulls out other's
 * pointer so other's destructor won't free the buffer we now own
 * Guards against self-assignment
 * Noexcept so containers will use it
 * @param other tensor to move from (left empty afterwards)
 * @return reference to *this, so assignments can chain
 */
Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    // t1 = Tensor({1, 2});
    // t1 = std::move(t2);
    if (this == &other)
    {
        return *this;
    }

    delete[] this->data;
    this->count = other.count;
    this->shape = std::move(other.shape);
    this->strides = std::move(other.strides);
    this->data = other.data;
    other.data = nullptr;
    other.count = 0;
    return *this;
}

/**
 * @brief Access the i-th element by flat index (row-major).
 * @param i flat index, 0 .. size()-1.
 * @return reference to the element, so it can be read or written.
 */
float &Tensor::operator[](int i)
{
    return this->data[i];
}

/**
 * @brief Read the i-th element by flat index (row-major).
 * @param i flat index, 0 .. size()-1.
 * @return a copy of the element (read only).
 */
float Tensor::operator[](int i) const
{
    return this->data[i];
}

/**
 * @brief Number of elements in the tensor
 *
 * @return product of all dimensions
 */
int Tensor::size() const
{
    return this->count;
}

/**
 * @brief Length of one dimension.
 * @param i which dimension (0-based).
 * @return shape[i], the size of dimension i.
 */
int Tensor::dim(int i) const
{
    return shape[i];
}

/**
 * @brief Access element by multi-dim index;
 *
 * @param idx
 * @return const float for read only
 */
float Tensor::at(const std::vector<int> &idx) const
{
    int flat = 0;

    for (size_t i = 0; i < idx.size(); i++)
    {
        flat += idx[i] * this->strides[i];
    }

    return this->data[flat];
}

// begin()/end() expose the buffer as an iterator range, so STL algorithms
// (std::max_element, std::accumulate, ...) and range-for work on a Tensor.

/** @brief Iterator to the first element (read/write). */
float *Tensor::begin()
{
    return data;
}

/** @brief Iterator one past the last element (read/write). */
float *Tensor::end()
{
    return data + count;
}

/** @brief Iterator to the first element (read only, for const Tensors). */
const float *Tensor::begin() const
{
    return data;
}

/** @brief Iterator one past the last element (read only, for const Tensors). */
const float *Tensor::end() const
{
    return data + count;
}

/**
 * @brief Access element by multi-dim index;
 *
 * @param idx
 * @return float reference so you can read or write
 */
float &Tensor::at(const std::vector<int> &idx)
{
    int flat = 0;

    for (size_t i = 0; i < idx.size(); i++)
    {
        flat += idx[i] * this->strides[i];
    }

    return this->data[flat];
}

/**
 * @brief Print the tensor as a 2D grid (assumes exactly 2 dimensions).
 */
void Tensor::print() const
{
    for (int i = 0; i < this->dim(0); i++)
    {
        for (int j = 0; j < this->dim(1); j++)
        {
            std::cout << this->at({i, j}) << " ";
        }
        std::cout << "\n";
    }
}
