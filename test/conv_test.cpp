#include <gtest/gtest.h>

#include "../layers/conv.h"
#include "../utils/activation.h"

TEST(conv, conv_layer){
    // Conv conv(3, 3, 2, 1, 2, 0, Activation::ActivationFunctionType::RELU);
    Conv conv(
        1, 3, 3, 2, 2, 1, 0, Activation::ActivationFunctionType::RELU);
    
    double img[9] = { 1, 4, 7,
                      2, 5, 8,
                      3, 6, 9
    };
    // conv.setInput(img);
    double kernel[4] = {1, 3, 2, 4};
    double kernel2[4] = {1, 0, 0, 0};
    conv.setKernel(0, kernel);
    conv.setKernel(1, kernel2);
    auto d = conv.forward(img);

    EXPECT_EQ(d[0], 37);
    EXPECT_EQ(d[2], 47);
    EXPECT_EQ(d[1], 67);
    EXPECT_EQ(d[3], 77);
    EXPECT_EQ(d[4], 1);
    EXPECT_EQ(d[6], 2);
    EXPECT_EQ(d[5], 4);
    EXPECT_EQ(d[7], 5);

    std::cout << "backward :" << std::endl;
    double grad[8] = {1, 0, 1, 0, 1, 1, 1, 1};
    auto d2 = conv.backward(grad);
    EXPECT_EQ(d2[0], 2);
    EXPECT_EQ(d2[1], 4);
    EXPECT_EQ(d2[2], 0);
    EXPECT_EQ(d2[3], 4);
    EXPECT_EQ(d2[4], 8);
    EXPECT_EQ(d2[5], 0);
    EXPECT_EQ(d2[6], 2);
    EXPECT_EQ(d2[7], 4);
    EXPECT_EQ(d2[8], 0);
    std::cout << "Passed backward" << std::endl;
}

TEST(conv, test2){
    Conv conv(1, 2, 3, 1, 2, 1, 0, Activation::ActivationFunctionType::RELU);
    double img[6] = {1, 2, 3, 4, 5, 6};
    double kernel[4] = {1, 2, 3, 4};
    conv.setKernel(0, kernel);
    auto d = conv.forward(img);
    EXPECT_EQ(d[0], 37);
    EXPECT_EQ(d[1], 47);

    double *back = new double[2];
    back[0] = 1;
    back[1] = -1;
    auto d2 = conv.backward(back);
    EXPECT_EQ(d2[0], 2);
    EXPECT_EQ(d2[1], 4);
    EXPECT_EQ(d2[2], 0);
    EXPECT_EQ(d2[3], 4);
}
