#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <utility>
#include <random>

#include "activation.h"

class Layer{
    public:
    Layer(int inputSize, int outputSize, Activation::ActivationFunctionType TYPE): inputSize(inputSize), outputSize(outputSize){
        switch (TYPE) {
            case Activation::ActivationFunctionType::RELU:
                activation = Activation::relu;
                activationDerivative = Activation::relu_derivative;
                break;
            case Activation::ActivationFunctionType::SIGMOID:
                activation = Activation::sigmoid;
                activationDerivative = Activation::sigmoid_derivative;
                break;
            case Activation::ActivationFunctionType::TANH:
                activation = Activation::tanh;
                activationDerivative = Activation::tanh_derivative;
                break;
            case Activation::ActivationFunctionType::NONE:
                activation = [] (double x) { return x; };
                activationDerivative = [] (double x) { return 1.0; };
                break;
            default:
                throw "Activation function not supported";
        }
    }
    virtual ~Layer()= default;
    virtual double* forward(double *input) = 0; // pure virtual function
    virtual double* backward(double *grad) = 0; // pure virtual function
    virtual void update(double lr, int batchSize) = 0; // pure virtual function
    [[nodiscard]] int getOutputSize() const{ return outputSize; }
    [[nodiscard]] int getInputSize() const{ return inputSize; }
protected:
    int inputSize;
    int outputSize;
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
};

#endif //NETWORK_LAYER_H