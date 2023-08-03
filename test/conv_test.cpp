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

TEST(conv, conv_layer2){
    double*img = new double[9];
    for(int i = 0; i < 9; i++){
        img[i] = i + 1;
    }

    Conv conv(
        1, 3, 3, 2, 2, 1, 1, Activation::ActivationFunctionType::RELU);
    
    double kernel[4] = {1, 3, 2, 4};
    double kernel2[4] = {1, 0, 0, 0};
    conv.setKernel(0, kernel);
    conv.setKernel(1, kernel2);
    auto d = conv.forward(img);
    
    std::cout << "forward :" << std::endl;
    std::cout << MatMap(d, 2, 2) << std::endl;

}

/*
void cross_correlation(const Eigen::Ref<const Mat> &data, Mat &kernel, Mat &output, int stride, int padding){
    // std::cout << "--------------------------------" << std::endl;
    // std::cout << "data: " << std::endl << data << std::endl;
    // std::cout << "kernel: " << std::endl << kernel << std::endl;
    int inputHeight  = data.rows();
    int inputWidth   = data.cols();
    int outputHeight = (inputHeight + 2 * padding - kernel.rows()) / stride + 1;
    int outputWidth  = (inputWidth + 2 * padding - kernel.cols()) / stride + 1;
    
    int expected_startRow;
    int expected_startCol;
    int expected_endRow;
    int expected_endCol;
    int startRow;
    int startCol;
    int endRow;
    int endCol;

    for(int i = 0; i < outputHeight; i++){
        for(int j = 0; j < outputWidth; j++){
            expected_startRow = i * stride - padding;
            expected_startCol = j * stride - padding;
            expected_endRow = expected_startRow + kernel.rows();
            expected_endCol = expected_startCol + kernel.cols();
            startRow = std::max(expected_startRow, 0);
            startCol = std::max(expected_startCol, 0);
            endRow = std::min(expected_endRow, inputHeight);
            endCol = std::min(expected_endCol, inputWidth);
            // std::cout <<  i <<  ',' << j << std::endl;
            if(startRow >= endRow || startCol >= endCol){
                output(i, j) = 0;
                continue;
            }else{
                Mat t1, t2;
                t1 = data.block(startRow, startCol, endRow - startRow, endCol - startCol);
                t2 = kernel.block(startRow - expected_startRow, startCol - expected_startCol, endRow - startRow, endCol - startCol);
                output(i, j) = (t1.array() * t2.array()).sum();
            } 
        }
    }
}

*/