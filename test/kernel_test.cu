#include <gtest/gtest.h>
#include "../mathkernel.cuh"


TEST(Test, TestMemset){
    float *d_a, *h_a;

    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * 512));
    CUDA_CHECK(cudaMallocHost((void**)&h_a, sizeof(float) * 512));
    memset<<<2, 256>>>(d_a, 512, 1.0f);
    CUDA_CHECK(cudaMemcpy(h_a, d_a, sizeof(float) * 512, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 512; i++){
        ASSERT_EQ(h_a[i], 1.0f) << "i = " << i << " val= " << h_a[i];
    }
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFreeHost(h_a));
}

TEST(Test, matVerMul){
    float *mat;
    float *vec, *vec2;
    float *result, *h_result;

    float mat1[] = {1, 2, 3, 4, 5, 6}; // 2 * 3
    float vec1[] = {1, 2, 3};
    float h_vec2[] = {3, 4};

    CUDA_CHECK(cudaMalloc((void**)&mat, sizeof(float) * 6));
    CUDA_CHECK(cudaMalloc((void**)&vec, sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void**)&vec2, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(float) * 2));
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(float) * 2));
    
    CUDA_CHECK(cudaMallocHost((void**)&h_result, sizeof(float) * 2));

    CUDA_CHECK(cudaMemcpy(mat, mat1, sizeof(float) * 6, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vec, vec1, sizeof(float) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vec2, h_vec2, sizeof(float) * 2, cudaMemcpyHostToDevice));

    matMulvec<<<1, 256>>>(mat, vec, result, 2, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_result, result, sizeof(float) * 2, cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result[0], 14);
    ASSERT_EQ(h_result[1], 32);

    printf("\033[34m[----------]\033[0m PASSED 1\n");
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaFree(result)); 
    CUDA_CHECK(cudaMalloc((void**)&result, sizeof(float) * 3));
    CUDA_CHECK(cudaMallocHost((void**)&h_result, sizeof(float) * 3));


    matMulvec<<<1, 256>>>(mat, vec2, result, 2, 3, true);     
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_result, result, sizeof(float) * 3, cudaMemcpyDeviceToHost));

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
    float *d_a, *h_a;
    float *d_z, *h_z;
    float *d_z_prime, *h_z_prime;

    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * 5));
    CUDA_CHECK(cudaMalloc((void**)&d_z, sizeof(float) * 5));
    CUDA_CHECK(cudaMalloc((void**)&d_z_prime, sizeof(float) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_a, sizeof(float) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_z, sizeof(float) * 5));
    CUDA_CHECK(cudaMallocHost((void**)&h_z_prime, sizeof(float) * 5));

    float z[] = {1, 2, 3, 4, 5};
    CUDA_CHECK(cudaMemcpy(d_z, z, sizeof(float) * 5, cudaMemcpyHostToDevice));
    sigmoid_ztoa<<<1, 5>>>(d_z, d_a, 5);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_a, d_a, sizeof(float) * 5, cudaMemcpyDeviceToHost));
    for(int i = 0; i < 5; i++){
        EXPECT_EQ(fabs(h_a[i] - (1 / (1+exp(-i-1)))) < 1e-5, true);
    }



    sigmoid_z_prime<<<1, 5>>>(d_a, d_z_prime, 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_z_prime, d_z_prime, sizeof(float) * 5, cudaMemcpyDeviceToHost));
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
    float *a, *y, *da;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&y, n * sizeof(float));
    cudaMalloc(&da, n * sizeof(float));

    // Initialize the input arrays
    float a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float y_data[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    cudaMemcpy(a, a_data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, y_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the cost_prime kernel
    cost_prime<<<(n + 255) / 256, 256>>>(a, y, da, n);
    cudaDeviceSynchronize();

    // Check the output array
    float expected[] = {-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0};
    float *actual = new float[n];
    cudaMemcpy(actual, da, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(expected[i], actual[i]);
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
    float *a, *b, *c;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));

    // Initialize the input arrays
    float a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float b_data[] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    cudaMemcpy(a, a_data, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the vecMul function
    vecMul<<<1, n>>>(a, b, c, n);
    cudaDeviceSynchronize();

    // Check the output array
    float expected[] = {10.0, 18.0, 24.0, 28.0, 30.0, 30.0, 28.0, 24.0, 18.0, 10.0};
    float *actual = new float[n];
    cudaMemcpy(actual, c, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(expected[i], actual[i]);
    }

    // Free the memory for the input and output arrays
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

TEST(Test, copyTest) {
    // Allocate memory for the input and output arrays
    int n = 10;
    float *a, *b;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));

    // Initialize the input array
    float a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    cudaMemcpy(a, a_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the copy kernel
    copy<<<(n + 255) / 256, 256>>>(b, a, n);
    cudaDeviceSynchronize();

    // Check the output array
    float expected[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float *actual = new float[n];
    cudaMemcpy(actual, b, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        EXPECT_FLOAT_EQ(expected[i], actual[i]);
    }

    // Free the memory for the input and output arrays
    cudaFree(a);
    cudaFree(b);
}
