//
// Created by JunchengJi on 7/31/2023.
//
#include <gtest/gtest.h>
#include "../CNN.h"
#include "../layers/layer_type.h"
#include "../utils/mnist_loader.h"

using namespace Activation;

TEST(cnn_test, LeNet5){
    const char* train_image_file = "E:/Projects/Cuda/Network/data/train-images.idx3-ubyte";
    const char* train_label_file = "E:/Projects/Cuda/Network/data/train-labels.idx1-ubyte";
    const char* test_image_file = "E:/Projects/Cuda/Network/data/t10k-images.idx3-ubyte";
    const char* test_label_file = "E:/Projects/Cuda/Network/data/t10k-labels.idx1-ubyte";
    std::vector<double*> train_data, test_data;
    std::vector<int> train_label, test_label;

    load_mnist(train_image_file, train_label_file, train_data, train_label);
    load_mnist(test_image_file, test_label_file, test_data, test_label);

    ASSERT_EQ(train_data.size(), 60000);
    ASSERT_EQ(train_label.size(), 60000);
    ASSERT_EQ(test_data.size(), 10000);
    ASSERT_EQ(test_label.size(), 10000);

    CNN cnn(7, 10);
    cnn.addLayer(new Conv(1, 28, 28, 6, 5, 1, 0, ActivationFunctionType::SIGMOID));
    cnn.addLayer(new Pooling(6, 24, 24, 2, 2, 0, PoolingType::MAX));
    cnn.addLayer(new Conv(6, 12, 12, 16, 5, 1, 0, ActivationFunctionType::SIGMOID));
    cnn.addLayer(new Pooling(16, 8, 8, 2, 2, 0, PoolingType::MAX));
    cnn.addLayer(new Dense(4 * 4 * 16, 120, ActivationFunctionType::SIGMOID));
    cnn.addLayer(new Dense(120, 84, ActivationFunctionType::SIGMOID));
    cnn.addLayer(new Dense(84, 10, ActivationFunctionType::SIGMOID));
    cnn.setDataset(train_data, train_label, test_data, test_label);
    cnn.train(100, 0.2, 10);
}