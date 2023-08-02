//
// Created by JunchengJi on 7/30/2023.
//

#include <gtest/gtest.h>

#include "../mlp.h"
#include "../utils/mnist_loader.h"

using namespace Activation;

TEST(MLP, train){
    const char* train_image_file = "E:/Projects/Cuda/Network/data/train-images.idx3-ubyte";
    const char* train_label_file = "E:/Projects/Cuda/Network/data/train-labels.idx1-ubyte";
    const char* test_image_file = "E:/Projects/Cuda/Network/data/t10k-images.idx3-ubyte";
    const char* test_label_file = "E:/Projects/Cuda/Network/data/t10k-labels.idx1-ubyte";
    std::vector<double*> train_data;
    std::vector<double*> test_data;
    std::vector<int> train_label, test_label;

    load_mnist(train_image_file, train_label_file, train_data, train_label);
    load_mnist(test_image_file, test_label_file, test_data, test_label);

    ASSERT_EQ(train_data.size(), 60000);
    ASSERT_EQ(train_label.size(), 60000);
    ASSERT_EQ(test_data.size(), 10000);
    ASSERT_EQ(test_label.size(), 10000);

    int num_layers = 4;
    std::vector<int> layers = {784, 50, 16, 10};
    MLP mlp(num_layers, layers);

    mlp.set_dataset(train_data, train_label, test_data, test_label);
    mlp.train(30, 10, 3.0);
    mlp.predict(test_data, test_label);
}