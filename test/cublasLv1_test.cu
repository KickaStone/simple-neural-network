
#include <cublas_v2.h>

#include <cstdlib>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include "../common.h"

#include <gtest/gtest.h>

using data_type = double;
cublasHandle_t cublasH = NULL;
cudaStream_t stream = NULL;

/* cublas level 1 function */
/* cublasDamax */
TEST(cublas, amax){

    /**
     * A = [1.0 2.0 3.0; 4.0]
    */
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const int incx = 1;

    int result = 0.0;
    data_type *d_A = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasIdamax(cublasH, A.size(), d_A, incx, &result));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    EXPECT_EQ(result, 4);
}

/* cublasDamin */
TEST(cublas, amin){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};

    const int incx =1;

    int result = 5.0;
    data_type *d_A = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasIdamin(cublasH, A.size(), d_A, incx, &result));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(result, 1);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

/* sum : cublasDasum */
TEST(cublas, asum){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};

    const int incx =1;

    data_type *d_A = NULL;
    double result = 0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDasum(cublasH, A.size(), d_A, incx, &result));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_TRUE(result - 10 < 1e-6);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

/* vecMul cublasDaxpy */
TEST(cublas, axpy){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {1.0, 2.0, 3.0, 4.0};
    
    const int incx =1;
    const int incy =1;

    data_type alpha = 2.1;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDaxpy(cublasH, A.size(), &alpha, d_A, incx, d_B, incy));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    for(int i = 0; i < B.size(); ++i){
        EXPECT_TRUE(B[i] - (A[i] + alpha * A[i]) < 1e-6);
    }


    for(int i = 0; i < B.size(); ++i){
        EXPECT_TRUE(B[i] - (A[i] + alpha * A[i]) < 1e-6);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

/* copy : cublasDcopy */
TEST(cublas, copy){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B(A.size(), 0);
    
    const int incx = 1;
    const int incy = 1;

    data_type alpha = 2.1;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDcopy(cublasH, A.size(), d_A, incx, d_B, incy));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));


    for(int i = 0; i < B.size(); ++i){
        EXPECT_EQ(B[i], A[i]);
    }


    for(int i = 0; i < B.size(); ++i){
        EXPECT_TRUE(B[i] - (A[i] + alpha * A[i]) < 1e-6);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

/* also have complex number version: cublasDdotc */
TEST(cublas, dot){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};

    const int incx = 1;
    const int incy = 1;

    data_type result = 0;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDdot(cublasH, A.size(), d_A, incx, d_B, incy, &result));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    data_type res = 0;
    for(int i = 0; i < A.size(); i++){
        res += A[i] * B[i];
    }

    EXPECT_EQ(result, res);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}
    
TEST(cublas, nrm2){
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};

    const int incx = 1;

    data_type result = 0;
    data_type *d_A = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasDnrm2(cublasH, A.size(), d_A, incx, &result)); 
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
        nrm2(A) = sqrt(1*1 + 2*2 + 3*3 + 4*4)
    */
    EXPECT_TRUE(result - 5.477225575 < 1e-6);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}


TEST(cublas, rot){
    std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};

    const int incx = 1;
    const int incy = 1;

    const data_type c = 2.1;
    const data_type s = 1.2;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDrot(cublasH, A.size(), d_A, incx, d_B, incy, &c, &s));

    CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type)*A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type)*B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // [ A  B ] * rot(sin a = s, cos a = c) = [c*x + s*y -s*x + c* y]
    for(int i = 0; i < A.size(); i++){
        std::cout << A[i] << ' ';
    }
    std::cout << std::endl;

    for(int i = 0; i < B.size(); i++){
        std::cout << B[i] << ' ';
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}

/* https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasDrotg#cublas-t-rotg */
TEST(cublas, rotg){
    data_type A = 2.1;
    data_type B = 1.2;
    data_type c = 2.1;
    data_type s = 1.2;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUBLAS_CHECK(cublasDrotg(cublasH, &A, &B, &c, &s));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::cout << A << ' ' << B << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}


TEST(cublas, rotm){
    std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<data_type> param = {1.0, 5.0, 6.0, 7.0, 8.0};

    const int incx = 1;
    const int incy = 1;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDrotm(cublasH, A.size(), d_A, incx, d_B, incy, param.data()));

    CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type)*A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type)*B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    
    for(int i = 0; i < A.size(); i++){
        std::cout << A[i] << ' ';
    }
    std::cout << std::endl;

    for(int i = 0; i < B.size(); i++){
        std::cout << B[i] << ' ';
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}


TEST(cublas, rotmg){
    data_type A = 1.0;
    data_type B = 5.0;
    data_type X = 2.1;
    data_type Y = 1.2;
    std::vector<data_type> param = {1.0, 5.0, 6.0, 7.0, 8.0};

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUBLAS_CHECK(cublasDrotmg(cublasH, &A, &B, &X, &Y, param.data()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::cout << A << ' ' << B << ' ' << X << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}


TEST(cublas, scal){

    /**
     * A = [1.0 2.0 3.0; 4.0]
    */
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const int incx = 1;

    const data_type alpha = 2.2;
    data_type *d_A = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasDscal(cublasH, A.size(), &alpha, d_A, incx));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    for(int i = 0; i < A.size(); i++){
        std::cout << A[i] << ' ';
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

}

TEST(cublas, swap){
    std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};

    const int incx = 1;
    const int incy = 1;

    data_type *d_A = NULL;
    data_type *d_B = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));
    
    CUBLAS_CHECK(cublasDswap(cublasH, A.size(), d_A, incx, d_B, incy));

    CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type)*A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type)*B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for(int i = 0; i < A.size(); i++){
        std::cout << A[i] << ' ';
    }
    std::cout << std::endl;

    for(int i = 0; i < B.size(); i++){
        std::cout << B[i] << ' ';
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));
}