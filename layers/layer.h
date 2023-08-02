#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <iostream>
#include <utility>
#include <random>
#include <numeric>
#include <algorithm>
#include <limits>

#include "../utils/activation.h"
#include "../utils/eigen_helper.h"

class Layer{
    public:
    Layer(int inputSize, int outputSize, Activation::ActivationFunctionType TYPE): inputSize(inputSize), outputSize(outputSize){
        switch (TYPE) {
            case Activation::ActivationFunctionType::RELU:
                activation = Activation::relu;
                derivative = Activation::relu_derivative;
                break;
            case Activation::ActivationFunctionType::SIGMOID:
                activation = Activation::sigmoid;
                derivative = Activation::sigmoid_derivative;
                break;
            case Activation::ActivationFunctionType::TANH:
                activation = Activation::tanh;
                derivative = Activation::tanh_derivative;
                break;
            case Activation::ActivationFunctionType::NONE:
                activation = [] (double x) { return x; };
                derivative = [] (double x) { return 1.0; };
                break;
            default:
                throw "Activation function not supported";
        }
    }
    virtual ~Layer()= default;
    virtual double* forward(const double *input) = 0; // pure virtual function
    virtual double* backward(const double *grad) = 0; // pure virtual function
    virtual void update(double lr, int batchSize) = 0; // pure virtual function
    [[nodiscard]] int getOutputSize() const{ return outputSize; }
    [[nodiscard]] int getInputSize() const{ return inputSize; }
protected:
    int inputSize;
    int outputSize;
    std::function<double(double)> activation;
    std::function<double(double)> derivative;
};

#endif //NETWORK_LAYER_H