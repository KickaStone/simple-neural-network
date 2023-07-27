#include <vector>
#include <gtest/gtest.h>
#include "../common.h"

#include <cublas_v2.h>
#include <iostream>

using namespace std;

cublasHandle_t cublasH;
cudaStream_t stream;

using data_type = double;

TEST(test, gbmv){
    const int m = 2;
    const int n = 2;
    const int lda = m;

    const std::vector<data_type> A = {1.0, 3.0, 2.0, 4.0};
    const std::vector<data_type> x = {5.0, 6.0};
    std::vector<data_type> y(m, 0);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;
    const int incx = 1;
    const int incy = 1;
    const int kl = 0;
    const int ku = 1;

    data_type *d_A = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
    printf("=====\n");

    printf("x\n");
    print_vector(x.size(), x.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        cublasDgbmv(cublasH, transa, m, n, kl, ku, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(y.data(), d_y, sizeof(data_type) * y.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   y = | 27.0 24.0 |
     */

    printf("y\n");
    print_vector(y.size(), y.data());
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

}
