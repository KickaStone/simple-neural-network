//
// Created by JunchengJi on 7/30/2023.
//

#include <gtest/gtest.h>
#include "../utils/mnist_loader.h"

TEST(MnistLoaderTest, LoadMnist)
{
    std::vector<double*> data;
    std::vector<int> label;
    load_mnist("E:\\Projects\\Cuda\\Network\\data\\train-images.idx3-ubyte", "E:\\Projects\\Cuda\\Network\\data\\train-labels.idx1-ubyte", data, label);
    EXPECT_EQ(data.size(), 60000);
    EXPECT_EQ(label.size(), 60000);

    // print the first image
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j)
            if(data[0][i * 28 + j] > 0.5)
                std::cout << "\033[34m1 \033[0m";
            else
                std::cout << "0 ";
        std::cout << std::endl;
    }
    std::cout << "label: " << label[0] << std::endl;
}