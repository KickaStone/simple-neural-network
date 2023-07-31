//
// Created by JunchengJi on 7/31/2023.
//
#include <gtest/gtest.h>

#include "../Pooling.h"

TEST(pooling_test, test1){
    double input[32] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
            13,14,15,16,
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12,
            13,14,15,16
    };


    Pooling pooling(2, 4, 4, 2, 0, 2, PoolingType::AVG, Activation::ActivationFunctionType::NONE);
    double* output = pooling.forward(input);
//    EXPECT_EQ(a[0], 6);
//    EXPECT_EQ(a[1], 8);
//    EXPECT_EQ(a[2], 14);
//    EXPECT_EQ(a[3], 16);
    EXPECT_EQ(output[0], 3.5);
    EXPECT_EQ(output[1], 5.5);
    EXPECT_EQ(output[2], 11.5);
    EXPECT_EQ(output[3], 13.5);
    EXPECT_EQ(output[4], 3.5);
    EXPECT_EQ(output[5], 5.5);
    EXPECT_EQ(output[6], 11.5);
    EXPECT_EQ(output[7], 13.5);

    auto *output_grad = new double[8];
    std::fill(output_grad, output_grad + 4, 1);
    std::fill(output_grad + 4, output_grad + 8, 2);
    double* input_grad = pooling.backward(output_grad);
//    EXPECT_EQ(input_grad[0], 0);
//    EXPECT_EQ(input_grad[1], 0);
//    EXPECT_EQ(input_grad[2], 0);
//    EXPECT_EQ(input_grad[3], 0);
//    EXPECT_EQ(input_grad[4], 0);
//    EXPECT_EQ(input_grad[5], 1);
//    EXPECT_EQ(input_grad[6], 0);
//    EXPECT_EQ(input_grad[7], 1);
//    EXPECT_EQ(input_grad[8], 0);
//    EXPECT_EQ(input_grad[9], 0);
//    EXPECT_EQ(input_grad[10], 0);
//    EXPECT_EQ(input_grad[11], 0);
//    EXPECT_EQ(input_grad[12], 0);
//    EXPECT_EQ(input_grad[13], 1);
//    EXPECT_EQ(input_grad[14], 0);
//    EXPECT_EQ(input_grad[15], 1);
    for(int i = 0; i < 16; i++){
        EXPECT_EQ(input_grad[i], 0.25) << i << std::endl;
    }
    for(int i = 16; i < 32; i++){
        EXPECT_EQ(input_grad[i], 0.5) << i << std::endl;
    }
}