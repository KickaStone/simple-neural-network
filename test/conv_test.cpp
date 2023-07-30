#include <gtest/gtest.h>

#include "../conv.h"


TEST(conv, cross_correlation){
    double img[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double kernel[4] = {1, 2, 3, 4};
    double output[4];
    Conv::cross_correlation(img, kernel, 3, 2, 1, output);
    double expected[4] = {37, 47, 67, 77};
    for(int i = 0; i < 4; i++){
        EXPECT_EQ(output[i], expected[i]);
    }


    Conv::correlation(img, kernel, 3, 2, 1, output);
    double expected2[4] = {23, 33, 53, 63};
    for(int i = 0; i < 4; i++){
        EXPECT_EQ(output[i], expected2[i]);
    }
}

TEST(conv, padding){
    double img[4] = {1, 2, 3, 4};
    double out[16] = {0};
    Conv::padding(img, 2, 1, out);
    double expected[16] = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    for(int i = 0; i < 16; i++){
        EXPECT_EQ(out[i], expected[i]);
    }
}

TEST(conv, conv_layer){
    Conv conv(
            9, // input size
            Activation::ActivationFunctionType::RELU, // activation function
            3, // Nx
            3, // Ny
            2, // kernel size
            1, // stride
            2, // n_kernel
            0 // padding
    );
    double img[9] = {1, 2, 3, 4,  5, 6, 7, 8, 9};
    conv.setInput(img);
    double kernel[4] = {1, 2, 3, 4};
    double kernel2[4] = {1, 0, 0, 0};
    conv.setKernel(0, kernel);
    conv.setKernel(1, kernel2);
    auto d = conv.forward(img);

    EXPECT_EQ(d[0], 37);
    EXPECT_EQ(d[1], 47);
    EXPECT_EQ(d[2], 67);
    EXPECT_EQ(d[3], 77);
    EXPECT_EQ(d[4], 1);
    EXPECT_EQ(d[5], 2);
    EXPECT_EQ(d[6], 4);
    EXPECT_EQ(d[7], 5);

    std::cout << "backward :" << std::endl;
    double grad[8] = {1, 1, 0, 0, 1, 1, 1, 1};
    auto d2 = conv.backward(grad);
    EXPECT_EQ(d2[0], 2);
    EXPECT_EQ(d2[1], 4);
    EXPECT_EQ(d2[2], 2);
    EXPECT_EQ(d2[3], 4);
    EXPECT_EQ(d2[5], 4);
    EXPECT_EQ(d2[6], 0);
    EXPECT_EQ(d2[7], 0);
    EXPECT_EQ(d2[8], 0);


}



