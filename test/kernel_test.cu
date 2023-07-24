#include <gtest/gtest.h>
#include "../mathkernel.cuh"


TEST(Test, TestMemset){
    double *d_a, *h_a;

    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(double) * 512));
    CUDA_CHECK(cudaMallocHost((void**)&h_a, sizeof(double) * 512));
    memset<<<2, 256>>>(d_a, 512, 1.0f);
    CUDA_CHECK(cudaMemcpy(h_a, d_a, sizeof(double) * 512, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 512; i++){
        ASSERT_EQ(h_a[i], 1.0f) << "i = " << i << " val= " << h_a[i];
    }
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFreeHost(h_a));
}

TEST(Test, matVerMul){
    double *mat;
    double *vec, *vec2;
    double *result, *h_result;

    double mat1[] = {1, 2, 3, 4, 5, 6}; // 2 * 3
    double vec1[] = {1, 2, 3};
    double h_vec2[] = {3, 4};

    CUDA_CHECK(cudaMalloc((void**)&mat, sizeof(double) * 6));
    CUDA_CHECK(cudaMalloc((void**)&vec, sizeof(double) * 3));
    CUDA_CHECK(cudaMalloc((void**)&vec2, sizeof(double) * 2));
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(double) * 2));
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(double) * 2));
    
    CUDA_CHECK(cudaMallocHost((void**)&h_result, sizeof(double) * 2));

    CUDA_CHECK(cudaMemcpy(mat, mat1, sizeof(double) * 6, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vec, vec1, sizeof(double) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vec2, h_vec2, sizeof(double) * 2, cudaMemcpyHostToDevice));

    matMulvec<<<1, 256>>>(mat, vec, result, 2, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_result, result, sizeof(double) * 2, cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result[0], 14);
    ASSERT_EQ(h_result[1], 32);

    printf("\033[34m[----------]\033[0m PASSED 1\n");
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaFree(result)); 
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(double) * 3));
    CUDA_CHECK(cudaMallocHost((void**)&h_result, sizeof(double) * 3));


    matMulvec<<<1, 256>>>(mat, vec2, result, 2, 3, true);     
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_result, result, sizeof(double) * 3, cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result[0], 19);
    ASSERT_EQ(h_result[1], 26);
    ASSERT_EQ(h_result[2], 33);

    printf("\033[34m[----------]\033[0m PASSED 2\n");
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(mat));
    CUDA_CHECK(cudaFree(vec));
    CUDA_CHECK(cudaFree(vec2));
}

TEST(Test, TestSigmoidZ2A){
    double *d_a, *h_a;
    double *d_z, *h_z;
    double *d_z_prime, *h_z_prime;

    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(double) * 5));
    CUDA_CHECK(cudaMalloc((void**)&d_z, sizeof(double) * 5));
    CUDA_CHECK(cudaMalloc((void**)&d_z_prime, sizeof(double) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_a, sizeof(double) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_z, sizeof(double) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_z_prime, sizeof(double) * 5));

    double z[] = {1, 2, 3, 4, 5};
    CUDA_CHECK(cudaMemcpy(d_z, z, sizeof(double) * 5, cudaMemcpyHostToDevice));
    sigmoid_ztoa<<<1, 5>>>(d_z, d_a, 5);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_a, d_a, sizeof(double) * 5, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 5; i++){
        EXPECT_EQ(fabs(h_a[i] - (1 / (1+exp(-i-1)))) < 1e-5, true);
    }



    sigmoid_z_prime<<<1, 5>>>(d_a, d_z_prime, 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_z_prime, d_z_prime, sizeof(double) * 5, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 5; i++){
        EXPECT_EQ(fabs(h_z_prime[i] - (h_a[i] * (1 - h_a[i]))) < 1e-5, true);
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_z_prime));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_z));
    CUDA_CHECK(cudaFreeHost(h_z_prime));
}

// Define a test for the cost_prime kernel
TEST(Test, cost_primeTest) {
    // Allocate memory for the input and output arrays
    int n = 10;
    double *a, *y, *da;
    cudaMalloc(&a, n * sizeof(double));
    cudaMalloc(&y, n * sizeof(double));
    cudaMalloc(&da, n * sizeof(double));

    // Initialize the input arrays
    double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double y_data[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    cudaMemcpy(a, a_data, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_data, n * sizeof(double), cudaMemcpyHostToDevice);

    // Call the cost_prime kernel
    cost_prime<<<(n + 255) / 256, 256>>>(a, y, da, n);
    cudaDeviceSynchronize();

    // Check the output array
    double expected[] = {-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0};
    double *actual = new double[n];
    cudaMemcpy(actual, da, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_double_EQ(expected[i], actual[i]);
    }

    // Free the memory for the input and output arrays
    cudaFree(a);
    cudaFree(y);
    cudaFree(da);
}


// Define a test for the vecMul function
TEST(Test, vecMulTest) {
    // Allocate memory for the input and output arrays
    int n = 10;
    double *a, *b, *c;
    cudaMalloc(&a, n * sizeof(double));
    cudaMalloc(&b, n * sizeof(double));
    cudaMalloc(&c, n * sizeof(double));

    // Initialize the input arrays
    double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double b_data[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    cudaMemcpy(a, a_data, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_data, n * sizeof(double), cudaMemcpyHostToDevice);

    // Call the vecMul function
    vecMul<<<1, n>>>(a, b, c, n);
    cudaDeviceSynchronize();

    // Check the output array
    double expected[] = {10.0, 18.0, 24.0, 28.0, 30.0, 30.0, 28.0, 24.0, 18.0, 10.0};
    double *actual = new double[n];
    cudaMemcpy(actual, c, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_double_EQ(expected[i], actual[i]);
    }

    // Free the memory for the input and output arrays
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

TEST(Test, copyTest) {
    // Allocate memory for the input and output arrays
    int n = 10;
    double *a, *b;
    cudaMalloc(&a, n * sizeof(double));
    cudaMalloc(&b, n * sizeof(double));

    // Initialize the input array
    double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    cudaMemcpy(a, a_data, n * sizeof(double), cudaMemcpyHostToDevice);

    // Call the copy kernel
    copy<<<(n + 255) / 256, 256>>>(b, a, n);
    cudaDeviceSynchronize();

    // Check the output array
    double expected[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double *actual = new double[n];
    cudaMemcpy(actual, b, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_double_EQ(expected[i], actual[i]);
    }

    // Free the memory for the input and output arrays
    cudaFree(a);
    cudaFree(b);
}
