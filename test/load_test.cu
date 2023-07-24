#include "../Mnist_helper.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../network.cuh"

using namespace std;


int n1 = 60000, n2 = 10000;
vector<double*> training_images(n1);
vector<double*> test_images(n2);
vector<int> training_labels(n1);
vector<double*> y(n1);
vector<int> test_labels(n2);

TEST(Mnist_helper, read_mnist_images) {

    load(training_images, training_labels, test_images, test_labels);

    for(int i = 0; i < n1; i++){
        y[i] = new double[10];
        for(int j = 0; j < 10; j++)
            y[i][j] = 0.0f;
        y[i][training_labels[i]] = 1.0f;
    }

    cout << "training_images size: " << training_images.size() << endl;
    cout << "training_labels size: " << training_labels.size() << endl;
}

TEST(Train_Test, handwriting_recognition){
    spdlog::set_level(spdlog::level::off);
    vector<int> layers = {30, 10};
    NeuralNetwork nn = NeuralNetwork(784, layers, 3.0);
    nn.SDG_train(training_images, y, 30, 10, test_images, test_labels);

    auto ouput = nn.forward(test_images[0], 784);
    for(int i = 0; i < 10; i++)
        printf("%.8lf ", ouput[i]);
    printf("\n");
}