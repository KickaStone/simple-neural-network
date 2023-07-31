//
// Created by JunchengJi on 7/31/2023.
//
#include <gtest/gtest.h>
#include "../CNN.h"
#include "../layer_type.h"
#include "../mnist_loader.h"

TEST(cnn_test, test1){

    const char* train_image_file = "data/train-images.idx3-ubyte";
    const char* train_label_file = "data/train-labels.idx1-ubyte";
    const char* test_image_file = "data/t10k-images.idx3-ubyte";
    const char* test_label_file = "data/t10k-labels.idx1-ubyte";
    std::vector<double*> train_data;
    std::vector<double*> test_data;
    std::vector<int> train_label, test_label;
    std::vector<double*> y;

    load_mnist(train_image_file, train_label_file, train_data, train_label);
    load_mnist(test_image_file, test_label_file, test_data, test_label);

    for(int l : train_label){
        auto* yy = new double[10];
        for(int i = 0; i < 10; i++)
            yy[i] = 0.0;
        yy[l] = 1.0;
        y.push_back(yy);
    }

    ASSERT_EQ(train_data.size(), 60000);
    ASSERT_EQ(train_label.size(), 60000);
    ASSERT_EQ(y.size(), 60000);
    ASSERT_EQ(test_data.size(), 10000);
    ASSERT_EQ(test_label.size(), 10000);

    CNN cnn(3, 10);
    cnn.addLayer(new Conv(784, Activation::ActivationFunctionType::SIGMOID, 28, 28, 5, 1, 20, 0));
    cnn.addLayer(new Pooling(20, 24, 24, 4, 0, 2, PoolingType::MAX, Activation::ActivationFunctionType::NONE));
    cnn.addLayer(new Dense(20 * 12 * 12,  30, Activation::ActivationFunctionType::SIGMOID));
    cnn.addLayer(new Dense(30, 10, Activation::ActivationFunctionType::SIGMOID));

    cnn.train(train_data, y, 10, 3.0, 10);
    cnn.predict(test_data, test_label);

}

