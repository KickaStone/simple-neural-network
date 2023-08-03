//
// Created by JunchengJi on 7/31/2023.
//
#include <gtest/gtest.h>

#include "../layers/pooling.h"

TEST(pooling_test, MAX){
    double input[32] = {
            1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 17,
            4, 8, 12, 16,
            1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16
    };

    
    Pooling pooling(2, 4, 4, 2, 2, 0, PoolingType::MAX);
    double* output = pooling.forward(input);
    EXPECT_EQ(output[0], 6);
    EXPECT_EQ(output[2], 8);
    EXPECT_EQ(output[1], 14);
    EXPECT_EQ(output[3], 17);
    EXPECT_EQ(output[4], 6);
    EXPECT_EQ(output[6], 8);
    EXPECT_EQ(output[5], 14);
    EXPECT_EQ(output[7], 16);

    auto *output_grad = new double[8];
    std::fill(output_grad, output_grad + 4, 1);
    std::fill(output_grad + 4, output_grad + 8, 2);
    double* input_grad = pooling.backward(output_grad);
    EXPECT_EQ(input_grad[5], 1);
    EXPECT_EQ(input_grad[7], 1);
    EXPECT_EQ(input_grad[13], 1);
    EXPECT_EQ(input_grad[11], 1);

}

TEST(pooling_test, AVG){
    Pooling pooling(2, 4, 4, 2, 2, 0, PoolingType::AVG);
    double input[32] = {
            1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16,
            1, 5, 9, 13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16
    };
    double*output = pooling.forward(input);

    EXPECT_EQ(output[0], 3.5);
    EXPECT_EQ(output[2], 5.5);
    EXPECT_EQ(output[1], 11.5);
    EXPECT_EQ(output[3], 13.5);
    EXPECT_EQ(output[4], 3.5);
    EXPECT_EQ(output[6], 5.5);
    EXPECT_EQ(output[5], 11.5);
    EXPECT_EQ(output[7], 13.5);

    auto *output_grad = new double[8];
    std::fill(output_grad, output_grad + 4, 1);
    std::fill(output_grad + 4, output_grad + 8, 2);
    double* input_grad = pooling.backward(output_grad);
    for(int i = 0; i < 16; i++){
        EXPECT_EQ(input_grad[i], 0.25) << i << std::endl;
    }
    for(int i = 16; i < 32; i++){
        EXPECT_EQ(input_grad[i], 0.5) << i << std::endl;
    }

}

// TEST(pooling_test, MAX_2){
//     Pooling pooling(1, 2, 4, 2, 1, 1, PoolingType::MAX);
//     double input[8] = {
//             1, 5, 
//             2, 6, 
//             3, 7, 
//             4, 8,
//     };
//     double *output = pooling.forward(input);
//     EXPECT_EQ(output[0], 2);

//     // double* output_grad = new double[3]{1, 2, 3};
//     // double* input_grad = pooling.backward(output_grad);
//     // EXPECT_EQ(input_grad[0], 0);
// }