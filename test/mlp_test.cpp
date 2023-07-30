//
// Created by JunchengJi on 7/30/2023.
//

#include <gtest/gtest.h>

#include "../mlp.h"
#include "../mnist_loader.h"

using namespace Activation;

TEST(MLP, train){
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

    ASSERT_EQ(train_data.size(), 60000);
    ASSERT_EQ(train_label.size(), 60000);
    ASSERT_EQ(test_data.size(), 10000);
    ASSERT_EQ(test_label.size(), 10000);

    for(int l : train_label){
        auto* yy = new double[10];
        for(int i = 0; i < 10; i++)
            yy[i] = 0.0;
        yy[l] = 1.0;
        y.push_back(yy);
    }

    int num_layers = 3;
    std::vector<int> layers = {784, 30, 10};
    MLP mlp(num_layers, layers);

    mlp.train(1, 10, 3.0, train_data, y);
    mlp.predict(test_data, test_label);
}