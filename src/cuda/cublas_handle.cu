#include "cublas_handle.h"

#include <memory>

cublasHandle_t get_cublas_handle()
{
    static std::unique_ptr<cublasContext, decltype(&cublasDestroy)> p(
        []()
        {
            cublasHandle_t handle;
            cublasCreate(&handle);
            return handle;
        }(),
        &cublasDestroy);

    return p.get();
}