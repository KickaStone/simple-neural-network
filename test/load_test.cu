#include "../Mnist_helper.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../network.cuh"

using namespace std;

int n1 = 60000, n2 = 10000;
vector<double *> training_images(n1);
vector<double *> test_images(n2);
vector<int> training_labels(n1);
vector<double *> y(n1);
vector<int> test_labels(n2);

TEST(Mnist_helper, read_mnist_images)
{

    load(training_images, training_labels, test_images, test_labels);

    for (int i = 0; i < n1; i++)
    {
        y[i] = new double[10];
        for (int j = 0; j < 10; j++)
            y[i][j] = 0.0f;
        y[i][training_labels[i]] = 1.0f;
    }

    cout << "training_images size: " << training_images.size() << endl;
    cout << "training_labels size: " << training_labels.size() << endl;
}

TEST(Train_Test, handwriting_recognition)
{

    vector<int> layers = {30, 10};
    NeuralNetwork nn = NeuralNetwork(784, layers);
    nn.setParams(3.0, 10, 1);
    nn.train(training_images, y, test_images, test_labels);
    // nn.save();
}