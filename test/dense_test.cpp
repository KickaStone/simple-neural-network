//
// Created by JunchengJi on 7/30/2023.
//
#include <gtest/gtest.h>
#include <cmath>

#include "../layers/dense.h"

using namespace Activation;

Dense dense(2, 3, ActivationFunctions(ActivationFunctionType::SIGMOID));
TEST(DenseTest, Forward){
    double input[2] = {1, 2};
    double ww[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double bb[] = {0.1, 0.2, 0.3};
    dense.setW(ww);
    dense.setB(bb);
    double *output = dense.forward(input);

    double o1 = 1 / (1 + exp(-0.1 - 0.1 * 1 - 0.2 * 2));
    double o2 = 1 / (1 + exp(-0.2 - 0.3 * 1 - 0.4 * 2));
    double o3 = 1 / (1 + exp(-0.3 - 0.5 * 1 - 0.6 * 2));

    EXPECT_TRUE(std::abs(output[0] - o1) < 0.0001) << output[0] << std::endl;
    EXPECT_TRUE(std::abs(output[1] - o2) < 0.0001) << output[1] << std::endl;
    EXPECT_TRUE(std::abs(output[2] - o3) < 0.0001) << output[2] << std::endl;
}


TEST(DenseTest, Backward){
    double aa[] = {1, 2 ,3};
    double output_grad[] = {1, 2, 3};
    double ww[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double bb[] = {0.1, 0.2, 0.3};

    dense.setW(ww);
    dense.setB(bb);
    dense.setA(aa);
    dense.setInput(aa);

    double *grad = dense.backward(output_grad);
    double g1 = 0.1 * 0 + 0.3 * (-4) + 0.5 * (-18);
    double g2 = 0.2 * 0 + 0.4 * (-4) + 0.6 * (-18);
    EXPECT_TRUE(abs(grad[0] - g1) < 0.0001) << grad[0] << std::endl;
    EXPECT_TRUE(abs(grad[1] - g2) < 0.0001) << grad[1] << std::endl;

    std::cout << "Test dz: " << std::endl;

    double *dz = dense.getDz();

    EXPECT_EQ(dz[0], 0);
    EXPECT_EQ(dz[1], -4);
    EXPECT_EQ(dz[2], -18);

    double *db = dense.getDb();

    EXPECT_EQ(db[0], 0);
    EXPECT_EQ(db[1], -4);
    EXPECT_EQ(db[2], -18);

    double *dw = dense.getDw();

    EXPECT_EQ(dw[0], 0);
    EXPECT_EQ(dw[1], 0);
    EXPECT_EQ(dw[2], -4);
    EXPECT_EQ(dw[3], -8);
    EXPECT_EQ(dw[4], -18);
    EXPECT_EQ(dw[5], -36);
}