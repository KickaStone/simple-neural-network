#include <gtest/gtest.h>

#include "../layers/conv.h"
#include "../utils/activation.h"

// TEST(conv, cross_correlation){
//     double img[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//     double kernel[4] = {1, 2, 3, 4};
//     double output[4];
//     convolution::cross_correlation(img, kernel, 3, 3, 2, 1, output);
//     double expected[4] = {37, 47, 67, 77};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(output[i], expected[i]);
//     }


//     convolution::correlation(img, kernel, 3, 3, 2, 1, output);
//     double expected2[4] = {23, 33, 53, 63};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(output[i], expected2[i]);
//     }
// }

// TEST(conv, convWithStride){
//     double img[16] = {
//             1, 2, 3, 4,
//             5, 6, 7, 8,
//             9,10,11,12,
//             13,14,15,16
//     };
//     double kernel[4];
//     std::fill(kernel, kernel + 4, 0.25);

//     double output[4];
//     convolution::cross_correlation(img, kernel, 4, 4, 2, 2, output);
//     double expected[4] = {3.5, 5.5, 11.5, 13.5};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(output[i], expected[i]);
//     }

// }

// TEST(conv, max_pooling){
//     double img[16] = {
//             1, 2, 3, 4,
//             5, 6, 7, 8,
//             9,10,11,12,
//             13,14,15,16
//     };
//     double output[4];
//     int record[4];
//     convolution::max_pooling(img, 4, 4, 2, 2, output, record);
//     double expected[4] = {6, 8, 14, 16};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(output[i], expected[i]);
//     }
//     double expected2[4] = {5, 7, 13, 15};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(record[i], expected2[i]);
//     }
// }

// TEST(conv, avg_pooling){
//     double img[16] = {
//             1, 2, 3, 4,
//             5, 6, 7, 8,
//             9,10,11,12,
//             13,14,15,16
//     };
//     double output[4];
//     convolution::avg_pooling(img, 4, 4, 2, 2, output);
//     double expected[4] = {3.5, 5.5, 11.5, 13.5};
//     for(int i = 0; i < 4; i++){
//         EXPECT_EQ(output[i], expected[i]);
//     }
// }


// TEST(conv, padding){
//     double img[4] = {1, 2, 3, 4};
//     double out[16] = {0};
//     convolution::padding(img, 2, 1, out);
//     double expected[16] = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
//     for(int i = 0; i < 16; i++){
//         EXPECT_EQ(out[i], expected[i]);
//     }
// }

TEST(conv, conv_layer){
//    Conv(int Nx, int Ny, int kernel_size,
//            int stride, int n_kernel, int padding, Activation::ActivationFunctionType type);
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
